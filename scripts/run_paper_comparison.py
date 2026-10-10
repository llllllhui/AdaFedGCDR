"""Run bounded, reproducible FedGCDR/AdaFedGCDR comparisons.

Example: python -m scripts.run_paper_comparison --events ... --out ...
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np
import torch

from model.adafedgcdr.data import Case, build_split, file_sha256, load_events, manifest_to_dict, user_views
from model.adafedgcdr.experiment import (
    Catalog, FusionConfig, METHODS, _case_records, _method_summary,
    _user_means, paired_bootstrap_ci, sample_candidates,
)
from model.adafedgcdr.fusion import AdaptiveFusion
from model.adafedgcdr.paper import (
    fit_head, original_records, popularity_records, prefix_cases, prepare_rows, target_inputs,
    train_backbone, train_sources, quarantine_development_users,
)


def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def catalog_for(events, manifest, items, train_users):
    seen = defaultdict(set)
    counts = Counter()
    for e in events:
        if e.domain != manifest.target_domain:
            continue
        seen[e.user_id].add(e.item_id)
        if e.user_id in train_users and e.item_id in items:
            counts[e.item_id] += 1
    return Catalog(tuple(sorted(items)),{u:frozenset(v) for u,v in seen.items()},dict(counts))


def original_validator(cases, catalog, visible, device):
    selected = [c for c in cases if c.label_item_id in catalog.popularity and visible[c.user_id]]
    candidates = [sample_candidates(c,catalog,100,2026) for c in selected]
    if not selected:
        raise ValueError('No validation users for original FedGCDR')
    def validate(server, table, users, mapping):
        ids = {u:i for i,u in enumerate(users)}
        with torch.no_grad():
            candidate_idx = torch.tensor([[table[i] for i in row] for row in candidates],device=device)
            uid = torch.tensor([mapping[ids[c.user_id]][server.domain_name] for c in selected],device=device)
            scores = torch.einsum('bd,bcd->bc',server.U[uid],server.V[candidate_idx])
            order = torch.argsort(scores,dim=1,descending=True,stable=True)
            rank = (order==99).float().argmax(1)
            discount = 1/torch.log2(rank.double()+2)
            return torch.where(rank<10,discount,0.).mean().item()
    return validate


def head_results(base, embeddings, train_rows, val_rows, test_rows, fixed_rows, seed, device, args, output):
    config = FusionConfig(seed=2026,candidate_count=100,device=device)
    models = {}
    traces = {}
    for method in ('dynamic-only','reliability-only','full'):
        print(f'fit gate={method} train={len(train_rows)} val={len(val_rows)}',flush=True)
        model, trace = fit_head(method,train_rows,val_rows,seed,device,args.gate_epochs,args.patience)
        models[method], traces[method] = model, trace
        torch.save({'state':model.state_dict(),'trace':trace,'training_seed':seed},output/f'{method}.pt')
    default = AdaptiveFusion(len(base.domain_names)).to(device)
    records, fixed = {}, {}
    for method in METHODS:
        model = models.get(method,default)
        rows = [r for r in test_rows if r['k']>0] if method=='target-only' else test_rows
        frows = [r for r in fixed_rows if r['k']>0] if method=='target-only' else fixed_rows
        records[method] = _case_records(model,rows,config,method)
        fixed[method] = _case_records(model,frows,config,method)
    records['popularity']=popularity_records(test_rows,device)
    fixed['popularity']=popularity_records(fixed_rows,device)
    if embeddings is not None:
        records['FedGCDR-original'] = original_records(test_rows,embeddings,base.target_items,base.target_item_vectors,device)
        fixed['FedGCDR-original'] = original_records(fixed_rows,embeddings,base.target_items,base.target_item_vectors,device)
    robustness={}
    if embeddings is None and 'Clothing' in base.domain_names:
        j=base.domain_names.index('Clothing')
        eligible=sorted({r['user_id'] for r in test_rows if bool(r['source_mask'][j]) and int(r['source_mask'].sum())>=2})
        if len(eligible)>1:
            permutation={u:eligible[(i+1)%len(eligible)] for i,u in enumerate(eligible)}
            for condition in ['remove-Clothing','shuffle-Clothing']:
                changed=[]
                for row in test_rows:
                    if row['user_id'] not in permutation:continue
                    clone=dict(row)
                    clone['source_vec']=row['source_vec'].clone()
                    if condition=='remove-Clothing':
                        clone['source_mask']=row['source_mask'].clone()
                        clone['source_mask'][j]=False
                        clone['source_count']=row['source_count'].clone()
                        clone['source_count'][j]=0
                        clone['source_vec'][j]=0
                    else:
                        clone['source_vec'][j]=base.source_vectors[permutation[row['user_id']]][j]
                    changed.append(clone)
                clean=[r for r in test_rows if r['user_id'] in permutation]
                robustness[condition]={'users':len(eligible),'cases':len(changed),'records':{},'clean_records':{}}
                for method in ['full','dynamic-only','source-only']:
                    model=models.get(method,default)
                    robustness[condition]['records'][method]=_case_records(model,changed,config,method)
                    robustness[condition]['clean_records'][method]=_case_records(model,clean,config,method)
    return records,fixed,traces,robustness


def run_one(events,manifest,source,seed,protocol,k,args,out):
    directory = out/f'seed-{seed}'/(f'sparse-k{k}' if protocol=='sparse' else 'cold')
    directory.mkdir(parents=True,exist_ok=True)
    result_path = directory/'result.json'
    if result_path.exists():
        print(f'reuse completed {directory}',flush=True)
        return json.loads(result_path.read_text())
    views = user_views(events,manifest.target_domain)
    items, visible, train_users = target_inputs(views,manifest,protocol,k,args.item_core)
    catalog = catalog_for(events,manifest,set(items),train_users)
    kvals = [k] if protocol=='sparse' else [0,1,3,5,10]
    val_cases = prefix_cases(events,manifest,'validation',kvals)
    validator = original_validator(val_cases,catalog,visible,args.device) if protocol=='sparse' else None
    print(f'seed={seed} protocol={protocol} k={k} catalog={len(items)}',flush=True)
    base, embeddings, backbone_info = train_backbone(events,manifest,source,seed,args.device,protocol,k,
        args.warm_rounds,args.transfer_rounds,args.fine_rounds,args.patience,args.item_core,validator)
    torch.save({'base':base,'target_user_embeddings':embeddings,'info':backbone_info},directory/'backbone.pt')
    row_sets, audit = {}, {}
    for group in ['fusion','validation','test']:
        cases = prefix_cases(events,manifest,group,kvals)
        row_sets[group],audit[group] = prepare_rows(cases,base,catalog,protocol=='sparse')
    fixed_cases = prefix_cases(events,manifest,'test',kvals,True)
    if protocol=='sparse':
        # Constant cohort for the fixed-label trajectory: the first item is known.
        fixed_cases = [c for c in fixed_cases if views[c.user_id]['target_events'][0].item_id in base.target_items]
    fixed_rows, audit['fixed-label'] = prepare_rows(fixed_cases,base,catalog,protocol=='sparse')
    # A fixed label trajectory must also use identical negatives across k.
    for row in fixed_rows:
        candidates = sample_candidates(Case(row['user_id'],0,(),row['candidate_items'][-1]),catalog,100,2026)
        row['candidate_items'] = candidates
        row['candidate_item_vectors'] = torch.stack([base.target_item_vectors[base.target_items[i]] for i in candidates])
        row['candidate_pop'] = torch.tensor([float(catalog.popularity[i]) for i in candidates])
    records,fixed,traces,robustness = head_results(base,embeddings if protocol=='sparse' else None,
        row_sets['fusion'],row_sets['validation'],row_sets['test'],fixed_rows,seed,args.device,args,directory)
    checks = {'finite_item_vectors':bool(torch.isfinite(base.target_item_vectors).all()),
        'cold_target_users_isolated': set(embeddings).issubset(set(manifest.users['base'])) if protocol=='cold' else None,
        'candidate_count':100,'catalog_from_training_groups_only':True}
    candidate_digest = hashlib.sha256(json.dumps([(r['user_id'],r['k'],r['candidate_items']) for r in row_sets['test']],sort_keys=True).encode()).hexdigest()
    result = {'training_seed':seed,'protocol':protocol,'k_values':kvals,'audit':audit,'backbone':backbone_info,
        'candidate_sha256':candidate_digest,'checks':checks,'gate_training':traces,
        'methods':{m:_method_summary(m,r,2026,len(base.target_items)) for m,r in records.items()},
        'records':records,'fixed_label_records':fixed,'robustness':robustness,
        'source_domains_per_test_user':{r['user_id']:int(base.source_mask[r['user_id']].sum()) for r in row_sets['test']},
        'original_cold_note':'Original FedGCDR is not evaluated for unseen target users; no random target ID score substituted.' if protocol=='cold' else None}
    write_json(result_path,result)
    return result


def aggregate(out,args):
    complete=all(len(list((out/f'seed-{s}').glob(f'{p}*/result.json')))>=(len(args.k_values.split(',')) if p=='sparse' else 1) for s in args.seeds for p in args.protocols.split(','))
    report = ['# 四域受控对比实验','',
        '状态：全部指定阶段已完成。' if complete else '**状态：运行中；当前表格只包含已完成阶段，不能作为最终跨阶段对比结果。**', '',
        '本报告是固定计算预算的多种子实验，不构成原论文指标复现。少交互协议与严格冷启动协议分别报告。',
        '',f'目标域 {args.target}；训练种子 {args.seeds}；拆分/候选种子固定2026；候选数100。',
        f'每源域训练{args.source_rounds}轮；目标域普通GAT {args.warm_rounds}轮、迁移GAT {args.transfer_rounds}轮；MF最多{args.fine_rounds}轮；门控最多{args.gate_epochs}轮，验证早停patience={args.patience}。',
        '原版Client/Server/GAT/MLP未修改，未采用batched训练改写；两个推荐头共享每一组骨干与物品向量。',
        '稀疏协议允许C/D已见前缀参与训练，C/D下一条标签仅用于验证/测试。A/B完整目标历史是共同可用训练监督。冷启动协议仅A目标历史进入骨干、B训练门控、C验证、D测试。',
        '所有消融分别训练；固定预测标签实验始终预测第11条交互，并使用共同用户集。物品目录仅由对应训练组建立，目录外标签及无目录内前缀单独计数。',
        '每种子的用户均值再跨种子报告均值±样本标准差；配对区间先对相同用户跨种子平均，再按用户bootstrap，不把种子重复当成独立用户。','']
    summaries = {'_meta':{'complete':complete,'requested_seeds':args.seeds,'requested_protocols':args.protocols,'sparse_k_values':args.k_values}}
    for protocol in ['sparse','cold']:
        per_seed = {}
        source_groups={}
        robustness_by_seed={}
        for seed in args.seeds:
            paths = list((out/f'seed-{seed}').glob(f'{protocol}*/result.json'))
            if not paths:
                continue
            recs, fixed = defaultdict(list),defaultdict(list)
            for path in paths:
                result=json.loads(path.read_text())
                source_groups.update(result.get('source_domains_per_test_user',{}))
                if result.get('robustness'): robustness_by_seed[seed]=result['robustness']
                for m, rows in result['records'].items(): recs[m].extend(rows)
                for m, rows in result['fixed_label_records'].items(): fixed[m].extend(rows)
            per_seed[seed]=(recs,fixed)
        if not per_seed:
            continue
        summary={}
        report += [f'## {protocol} 协议','', '| 方法 | 用户等权 NDCG@10，均值 ± 标准差 |', '|---|---:|']
        for method in per_seed[next(iter(per_seed))][0]:
            ks=[int(k) for k in args.k_values.split(',')] if protocol=='sparse' else ([1,3,5,10] if method=='target-only' else [0,1,3,5,10])
            means=[]
            for recs,_ in per_seed.values():
                values=list(_user_means(recs[method],ks).values())
                if values: means.append(float(np.mean(values)))
            if not means: continue
            summary[method]={'mean':float(np.mean(means)),'std':float(np.std(means,ddof=1)) if len(means)>1 else None,'seed_values':means,'k_values':ks}
            sd=summary[method]['std']
            report.append(f'| {method} | {summary[method]["mean"]:.6f} ± {sd:.6f} |' if sd is not None else f'| {method} | {summary[method]["mean"]:.6f}（单种子） |')
        comparator='FedGCDR-original' if protocol=='sparse' else 'source-only'
        user_scores=defaultdict(lambda:defaultdict(list))
        for recs,_ in per_seed.values():
            ks=[int(k) for k in args.k_values.split(',')] if protocol=='sparse' else [0,1,3,5,10]
            for method in ['full',comparator]:
                for u,v in _user_means(recs[method],ks).items(): user_scores[method][u].append(v)
        common=sorted(set(user_scores['full'])&set(user_scores[comparator]))
        a=[float(np.mean(user_scores['full'][u])) for u in common]
        b=[float(np.mean(user_scores[comparator][u])) for u in common]
        ci=list(paired_bootstrap_ci(a,b,2026))
        delta=float(np.mean(np.array(a)-np.array(b))) if a else None
        report += ['',f'full 相对 {comparator}：配对用户数 {len(common)}，NDCG@10 差值 {delta}，用户bootstrap 95%区间 {ci}。','']
        summary['paired_comparison']={'baseline':comparator,'users':len(common),'mean_diff':delta,'ci':ci}
        stage_values = {}
        fixed_values = {}
        for label, index, target in [('下一条交互',0,stage_values),('固定第11条交互',1,fixed_values)]:
            report += [f'### {label}的分阶段 NDCG@10','', '| k | full，均值 ± 标准差 | 对照，均值 ± 标准差 | 用户数 |','|---:|---:|---:|---:|']
            for k in ([int(k) for k in args.k_values.split(',')] if protocol=='sparse' else [0,1,3,5,10]):
                values={m:[] for m in ['full',comparator]}
                counts=[]
                for entry in per_seed.values():
                    record_sets=entry[index]
                    for m in values:
                        subset=[r for r in record_sets[m] if r['k']==k]
                        if subset: values[m].append(float(np.mean([r['ndcg@10'] for r in subset])))
                    counts.append(len({r['user_id'] for r in record_sets['full'] if r['k']==k}))
                if not values['full']: continue
                target[str(k)]={'methods':{m:{'mean':float(np.mean(v)),'std':float(np.std(v,ddof=1)) if len(v)>1 else None} for m,v in values.items() if v},'user_counts':counts}
                formatted=[]
                for m in values:
                    result=target[str(k)]['methods'].get(m)
                    formatted.append('N/A' if result is None else f'{result["mean"]:.6f}' + (f' ± {result["std"]:.6f}' if result['std'] is not None else ''))
                report.append(f'| {k} | {formatted[0]} | {formatted[1]} | {counts[0]} |')
            report.append('')
        summary['stages']=stage_values
        summary['fixed_label_stages']=fixed_values
        if source_groups:
            groups_summary={}
            report += ['### 按可用源域数量分组','', '| 源域数 | full | 对照 | 用户数 |','|---:|---:|---:|---:|']
            for group in [1,2,3]:
                grouped={m:[] for m in ['full',comparator]}
                users=set()
                for recs,_ in per_seed.values():
                    for m in grouped:
                        subset=[r for r in recs[m] if source_groups.get(r['user_id'])==group]
                        means=_user_means(subset,[int(k) for k in args.k_values.split(',')] if protocol=='sparse' else [0,1,3,5,10])
                        users.update(means)
                        if means: grouped[m].append(float(np.mean(list(means.values()))))
                if not users:continue
                groups_summary[str(group)]={'users':len(users),'methods':{m:float(np.mean(v)) for m,v in grouped.items() if v}}
                report.append(f'| {group} | {groups_summary[str(group)]["methods"].get("full")} | {groups_summary[str(group)]["methods"].get(comparator)} | {len(users)} |')
            summary['source_groups']=groups_summary
            report.append('')
        if robustness_by_seed:
            stress={}
            report += ['### 源域敏感性检查','', '仅在拥有Clothing及另一个源域的相同用户上移除或打乱Clothing表示；不涉及新的原版训练，结果仅解释冻结表示融合的敏感性。','', '| 条件 | 方法 | 清洁输入 NDCG@10 | 干扰输入 NDCG@10 |','|---|---|---:|---:|']
            for condition in ['remove-Clothing','shuffle-Clothing']:
                stress[condition]={}
                for m in ['full','dynamic-only','source-only']:
                    clean_values,changed_values=[],[]
                    for conditions in robustness_by_seed.values():
                        if condition not in conditions:continue
                        clean=_user_means(conditions[condition]['clean_records'][m],[0,1,3,5,10])
                        changed=_user_means(conditions[condition]['records'][m],[0,1,3,5,10])
                        if clean:
                            clean_values.append(float(np.mean(list(clean.values()))))
                            changed_values.append(float(np.mean(list(changed.values()))))
                    if not clean_values:continue
                    stress[condition][m]={'clean':float(np.mean(clean_values)),'changed':float(np.mean(changed_values))}
                    report.append(f'| {condition} | {m} | {np.mean(clean_values):.6f} | {np.mean(changed_values):.6f} |')
            summary['robustness']=stress
            report.append('')
        summary['completed_seeds']=list(per_seed)
        summaries[protocol]=summary
    write_json(out/'summary.json',summaries)
    (out/'report.md').write_text('\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--events',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--target',default='Books')
    parser.add_argument('--seeds',default='2026,2027,2028')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--source-rounds',type=int,default=5)
    parser.add_argument('--warm-rounds',type=int,default=5)
    parser.add_argument('--transfer-rounds',type=int,default=5)
    parser.add_argument('--fine-rounds',type=int,default=20)
    parser.add_argument('--gate-epochs',type=int,default=50)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--item-core',type=int,default=1)
    parser.add_argument('--protocols',default='sparse,cold')
    parser.add_argument('--k-values',default='1,3,5,10')
    parser.add_argument('--development-users',help='JSON list of previously inspected test-user IDs; moved to validation')
    args=parser.parse_args()
    args.seeds=[int(s) for s in args.seeds.split(',')]
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    events=load_events(args.events)
    manifest=build_split(events,args.target,2026)
    quarantined=()
    if args.development_users:
        manifest,quarantined=quarantine_development_users(manifest,json.loads(Path(args.development_users).read_text()))
        if not manifest.users['test']:raise ValueError('No untouched test users remain')
    original_files=['model/fedgcdr/party.py','model/fedgcdr/model.py']
    metadata={'args':vars(args),'events_sha256':file_sha256(args.events),'split':manifest_to_dict(manifest),
        'original_code_sha256':{p:file_sha256(p) for p in original_files},'python':sys.version,
        'implementation_code_sha256':{p:file_sha256(p) for p in ['model/adafedgcdr/paper.py','model/adafedgcdr/experiment.py','scripts/run_paper_comparison.py']},
        'quarantined_development_users':list(quarantined),
        'development_users_sha256':file_sha256(args.development_users) if args.development_users else None,
        'torch':torch.__version__,'host':platform.node(),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'status':'running','completed':[]}
    status_path=out/'run.json'
    if status_path.exists():
        old=json.loads(status_path.read_text())
        if old['events_sha256']!=metadata['events_sha256'] or old['args']!=metadata['args'] or old['split']!=metadata['split'] or old.get('implementation_code_sha256')!=metadata['implementation_code_sha256']:
            raise ValueError('Resume config/data differs from the existing experiment')
    write_json(status_path,metadata)
    try:
        for seed in args.seeds:
            source_path=out/f'seed-{seed}'/'source.pt'
            if source_path.exists(): source=torch.load(source_path,map_location='cpu',weights_only=False)
            else:
                source=train_sources(events,manifest,seed,args.device,args.source_rounds)
                source_path.parent.mkdir(parents=True,exist_ok=True)
                torch.save(source,source_path)
            for protocol in args.protocols.split(','):
                kvals=[int(k) for k in args.k_values.split(',')] if protocol=='sparse' else [None]
                for k in kvals:
                    started=time.monotonic()
                    run_one(events,manifest,source,seed,protocol,k,args,out)
                    metadata['completed'].append({'seed':seed,'protocol':protocol,'k':k,'seconds':time.monotonic()-started})
                    write_json(status_path,metadata)
                    aggregate(out,args)
        metadata['status']='complete'
        metadata['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    except BaseException as exc:
        metadata['status']='failed'
        metadata['error']=repr(exc)
        raise
    finally:
        write_json(status_path,metadata)


if __name__=='__main__':
    main()
