"""Audit completed seed/cohort consistency and all predeclared comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from model.adafedgcdr.experiment import _user_means, paired_bootstrap_ci


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    args=parser.parse_args()
    root=Path(args.results)
    suite=json.loads((root/'suite.json').read_text())
    if suite['status']!='complete':raise ValueError('Suite is not complete')
    seeds=suite['args']['seeds']
    checks={}
    comparisons={}
    for protocol,ks in [('sparse',[1,3,5,10]),('cold',[0,1,3,5,10])]:
        data={}
        fingerprints={}
        for seed in seeds:
            records={}
            for path in sorted((root/f'seed-{seed}').glob(f'{protocol}*/result.json')):
                result=json.loads(path.read_text())
                stage=path.parent.name
                fingerprints.setdefault(stage,[]).append(result['candidate_sha256'])
                for method,rows in result['records'].items():records.setdefault(method,[]).extend(rows)
            data[seed]=records
        if not all(len(set(values))==1 and len(values)==len(seeds) for values in fingerprints.values()):
            raise ValueError('Candidates or cohorts changed across seeds')
        for seed,records in data.items():
            reference={(r['user_id'],r['k']) for r in records['full']}
            for method,rows in records.items():
                if method=='target-only':continue
                if {(r['user_id'],r['k']) for r in rows}!=reference:
                    raise ValueError(f'Unpaired cohort: {protocol} {seed} {method}')
        baselines=['FedGCDR-original'] if protocol=='sparse' else ['source-only','fixed-0.5','dynamic-only','reliability-only','popularity']
        result={}
        for baseline in baselines:
            gathered={m:{} for m in ['full',baseline]}
            for records in data.values():
                for method in gathered:
                    for user,value in _user_means(records[method],ks).items():gathered[method].setdefault(user,[]).append(value)
            common=sorted(set(gathered['full'])&set(gathered[baseline]))
            full=[float(np.mean(gathered['full'][u])) for u in common]
            base=[float(np.mean(gathered[baseline][u])) for u in common]
            result[baseline]={'users':len(common),'full_mean':float(np.mean(full)),'baseline_mean':float(np.mean(base)),
                'difference':float(np.mean(np.array(full)-np.array(base))),
                'paired_user_bootstrap_95ci':list(paired_bootstrap_ci(full,base,2026))}
        comparisons[protocol]=result
        checks[protocol]={'candidate_and_cohort_identical_across_seeds':True,'method_cohorts_paired':True,'stages':list(fingerprints)}
    cold=comparisons['cold']
    strongest=max(cold,key=lambda m:cold[m]['baseline_mean'])
    k0={m:{} for m in ['full','source-only']}
    for seed in seeds:
        result=json.loads((root/f'seed-{seed}'/'cold'/'result.json').read_text())
        for m in k0:
            for u,v in _user_means(result['records'][m],[0]).items():k0[m].setdefault(u,[]).append(v)
    k0_common=sorted(set(k0['full'])&set(k0['source-only']))
    k0_diff=float(np.mean([np.mean(k0['full'][u])-np.mean(k0['source-only'][u]) for u in k0_common]))
    effective=all(v['paired_user_bootstrap_95ci'][0]>0 for v in cold.values()) and k0_diff>=0
    audit={'checks':checks,'comparisons':comparisons,'strongest_cold_baseline':strongest,
        'cold_k0_vs_source_only_difference':k0_diff,'cold_effective_under_all_baseline_criterion':bool(effective),
        'interpretation':'Individual bootstrap intervals are reported for every prespecified baseline. Cold effectiveness requires a positive lower interval bound against all comparable five-stage baselines and no k=0 loss vs source-only; target-only is excluded from the five-stage comparison.'}
    (root/'primary-comparisons.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['\n## 最终配对审计','', '三个训练种子的候选集/用户一致，方法间评估用户成对一致。','', '| 协议 | 对照 | full−对照 NDCG@10 | 用户bootstrap 95%区间 |','|---|---|---:|---|']
    for protocol,results in comparisons.items():
        for method,values in results.items():text.append(f'| {protocol} | {method} | {values["difference"]:.6f} | {values["paired_user_bootstrap_95ci"]} |')
    text+=['',f'冷启动最强可比基线：{strongest}。所有可比基线配对区间下界均大于0且k=0不退化：{effective}。','', '此处为受控预算和已知物品条件下的离线结论，不是线上效果或原论文指标复现。']
    with (root/'report.md').open('a') as file:file.write('\n'.join(text)+'\n')
    print(json.dumps({'audit':'passed','cold_effective':bool(effective),'strongest_cold_baseline':strongest}))


if __name__=='__main__':main()
