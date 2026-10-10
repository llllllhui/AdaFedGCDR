import argparse
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model.adafedgcdr.data import load_events,manifest_from_dict
from model.adafedgcdr.paper import prefix_cases,prepare_rows
from scripts.run_paper_comparison import catalog_for
from model.adafedgcdr.experiment import _collate,fuse_user_vec
from model.adafedgcdr.fusion import AdaptiveFusion

torch.set_num_threads(2)
parser=argparse.ArgumentParser(description='Inspect saved paper checkpoints without retraining')
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--events',type=Path,required=True)
parser.add_argument('--out',type=Path,default=Path('artifacts/embedding-diagnostics.json'))
args=parser.parse_args()
root=args.results
events=load_events(args.events)
def scalar(x):return float(x.detach().cpu())
def corr(a,b):
 a=a.double();b=b.double();a=a-a.mean(-1,keepdim=True);b=b-b.mean(-1,keepdim=True)
 denom=a.norm(dim=-1)*b.norm(dim=-1)
 valid=denom>1e-12
 values=(a*b).sum(-1)/denom.clamp_min(1e-12)
 return float(values[valid].mean()) if valid.any() else None
def stats(x):
 x=x.detach().flatten().float()
 return {'mean':scalar(x.mean()),'std':scalar(x.std()),'p10':scalar(x.quantile(.1)),'p50':scalar(x.median()),'p90':scalar(x.quantile(.9))}
def pair_cos(x):
 x=F.normalize(x,dim=-1);n=len(x)
 return scalar((x.sum(0).square().sum()-n)/(n*(n-1))) if n>1 else None

report={}
for seed in [2026,2027,2028]:
 manifest=manifest_from_dict(json.loads((root/f'workers/seed-{seed}/run.json').read_text())['split'])
 for stage in ['sparse-k1','sparse-k10','cold']:
  p=root/f'seed-{seed}'/stage
  checkpoint=torch.load(p/'backbone.pt',map_location='cpu',weights_only=False)
  base=checkpoint['base'];uids=checkpoint['target_user_embeddings']
  state=torch.load(p/'full.pt',map_location='cpu',weights_only=False)['state']
  model=AdaptiveFusion(3);model.load_state_dict(state)
  config=json.loads((p/'result.json').read_text())
  catalog=catalog_for(events,manifest,set(base.target_items),set(base.target_train_users))
  cases=prefix_cases(events,manifest,'test',config['k_values'])
  rows,audit=prepare_rows(cases,base,catalog,stage!='cold')
  batch=_collate(rows,'cpu')
  with torch.no_grad():
   h,w,r,personal=model(batch['source_vec'],batch['source_mask'],batch['source_count'],batch['history_vec'],batch['history_mask'])
   n=batch['history_mask'].sum(1)
   ht=(batch['history_vec']*batch['history_mask'][...,None]).sum(1)/n.clamp_min(1)[:,None]
   hs=(r[...,None]*batch['source_vec']).sum(1)
   scores={m:torch.einsum('bd,bcd->bc',fuse_user_vec(m,model,batch)[0],batch['candidate_item_vectors']) for m in ['full','source-only','target-only']}
   if stage!='cold':
    u=torch.stack([uids[row['user_id']] for row in rows])
    scores['original']=torch.einsum('bd,bcd->bc',u,batch['candidate_item_vectors'])
   d={'audit':audit,'gate':{k:v.tolist() for k,v in state.items()},'tau':scalar(F.softplus(model.tau_raw)),
      'source_norm':stats(hs.norm(dim=-1)),'history_norm':stats(ht[n>0].norm(dim=-1)),
      'source_history_cos':stats(F.cosine_similarity(hs[n>0],ht[n>0],dim=-1)),
      'source_pairwise_cos':pair_cos(hs),
      'item_norm_pop_corr':corr(base.target_item_vectors.norm(dim=-1),torch.tensor([math.log1p(catalog.popularity[i]) for i in sorted(base.target_items)])),
      'scores':{},'stages':{}}
   for m,s in scores.items():
    d['scores'][m]={'correlation_with_log_popularity':corr(s,torch.log1p(batch['candidate_pop'])),
       'positive_minus_negative_mean':stats(s[:,-1]-s[:,:-1].mean(1)),
       'within_candidate_std':stats(s.std(1))}
   for k in config['k_values']:
    selected=batch['k']==k
    d['stages'][str(k)]={'cases':int(selected.sum()),'valid_history_mean':scalar(n[selected].float().mean()),
       'no_known_history_ratio':scalar((n[selected]==0).float().mean()),'target_weight_mean':scalar(w[selected].mean())}
   if stage!='cold':
    d['original_user_norm']=stats(u.norm(dim=-1))
    d['history_original_cos']=stats(F.cosine_similarity(ht,u,dim=-1))
    d['source_original_cos']=stats(F.cosine_similarity(hs,u,dim=-1))
  report[f'{seed}/{stage}']=d
  print(seed,stage,'score correlations',{m:round(s['correlation_with_log_popularity'],3) for m,s in d['scores'].items()},'gate',d['gate'],flush=True)
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print('saved',args.out)
