"""Launch independent training seeds concurrently; persist a recoverable status."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_paper_comparison import aggregate, write_json


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
    parser.add_argument('--development-users')
    args=parser.parse_args()
    args.seeds=[int(s) for s in args.seeds.split(',')]
    out=Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    children={}
    status={'status':'running','started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'args':vars(args),'workers':{}}
    write_json(out/'suite.json',status)
    for seed in args.seeds:
        worker=out/'workers'/f'seed-{seed}'
        worker.mkdir(parents=True,exist_ok=True)
        target=worker/f'seed-{seed}'
        target.mkdir(exist_ok=True)
        link=out/f'seed-{seed}'
        if not link.exists():link.symlink_to(target,target_is_directory=True)
        command=[sys.executable,'-u','-m','scripts.run_paper_comparison','--events',args.events,'--out',str(worker),'--seeds',str(seed)]
        if args.development_users:command.extend(['--development-users',args.development_users])
        for name in ['target','device','source_rounds','warm_rounds','transfer_rounds','fine_rounds','gate_epochs','patience','item_core','protocols','k_values']:
            command.extend(['--'+name.replace('_','-'),str(getattr(args,name))])
        log=(worker/'worker.log').open('w')
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        children[seed]=(child,log,worker)
        status['workers'][str(seed)]={'pid':child.pid,'status':'starting','command':command}
    last=None
    while True:
        active=False
        for seed,(child,log,worker) in children.items():
            record=status['workers'][str(seed)]
            state_path=worker/'run.json'
            if state_path.exists():
                state=json.loads(state_path.read_text())
                record['status']=state['status']
                record['completed']=state['completed']
                record['error']=state.get('error')
            code=child.poll()
            record['exit_code']=code
            if code is None:active=True
            else:
                record['status']='complete' if code==0 else 'failed'
            path=worker/'worker.log'
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-100000))
                tail=f.read().decode(errors='replace')
            phases=re.findall(r'(?m)^(?:source=|seed=|sparse |cold |fit gate=)[^\r\n]+',tail)
            record['phase']=phases[-1] if phases else None
        aggregate(out,args)
        status['elapsed_seconds']=time.time()-started
        status['status']='running' if active else ('complete' if all(c.poll()==0 for c,_,_ in children.values()) else 'failed')
        write_json(out/'suite.json',status)
        compact={seed:{'status':v['status'],'completed':len(v.get('completed',[])),'phase':v.get('phase')} for seed,v in status['workers'].items()}
        serialized=json.dumps(compact,sort_keys=True)
        if serialized!=last:
            print(serialized,flush=True)
            last=serialized
        if not active:break
        time.sleep(30)
    for child,log,_ in children.values():log.close()
    if status['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
