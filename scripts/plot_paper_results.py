"""Export scientific figures from completed, paired experiment summaries."""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw_curves(ax, protocol, fixed=False):
    values=protocol['fixed_label_stages' if fixed else 'stages']
    ks=sorted(int(k) for k in values)
    methods=[('full','AdaFedGCDR','#c64f47','o')]
    baseline=protocol['paired_comparison']['baseline']
    names={'FedGCDR-original':'FedGCDR (original)','source-only':'Source only','popularity':'Popularity','fixed-0.5':'Fixed 0.5','dynamic-only':'Count gate','reliability-only':'Reliability gate'}
    methods.append((baseline,names.get(baseline,baseline),'#476ca4','s'))
    for method,label,color,marker in methods:
        mean=[values[str(k)]['methods'][method]['mean'] for k in ks]
        sd=[values[str(k)]['methods'][method]['std'] or 0. for k in ks]
        ax.errorbar(ks,mean,yerr=sd,color=color,marker=marker,capsize=3,lw=1.7,label=label)
    ax.set_xticks(ks)
    ax.set_xlabel('Observed target interactions (k)')
    ax.set_ylabel('NDCG@10')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y',alpha=.22)
    ax.legend(frameon=False,fontsize=9)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    args=parser.parse_args()
    root=Path(args.results)
    summary=json.loads((root/'summary.json').read_text())
    if not summary['_meta']['complete']:raise ValueError('Refusing to plot incomplete comparisons')
    audit_path=root/'primary-comparisons.json'
    cold=copy.deepcopy(summary['cold'])
    if audit_path.exists():
        baseline=json.loads(audit_path.read_text())['strongest_cold_baseline']
        cold['paired_comparison']['baseline']=baseline
        for field,fixed in [('stages',False),('fixed_label_stages',True)]:
            for k,values in cold[field].items():
                seed_values=[]
                for seed in summary['_meta']['requested_seeds']:
                    result=json.loads((root/f'seed-{seed}'/'cold'/'result.json').read_text())
                    rows=result['fixed_label_records' if fixed else 'records'][baseline]
                    selected=[r['ndcg@10'] for r in rows if r['k']==int(k)]
                    if selected:seed_values.append(float(np.mean(selected)))
                values['methods'][baseline]={'mean':float(np.mean(seed_values)),'std':float(np.std(seed_values,ddof=1)) if len(seed_values)>1 else None}
    note='Synthetic smoke test only: ' if 'smoke' in root.name else ''
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    for fixed in [False,True]:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
        draw_curves(axes[0],summary['sparse'],fixed)
        draw_curves(axes[1],cold,fixed)
        axes[0].set_title('(a) Sparse observed-user protocol')
        axes[1].set_title('(b) Isolated-user cold-start protocol')
        fig.suptitle(note+('Fixed next label (11th interaction)' if fixed else 'Next-interaction recommendation'),fontsize=12)
        name='fixed-label-comparison' if fixed else 'main-comparison'
        fig.savefig(root/f'{name}.png',dpi=300)
        fig.savefig(root/f'{name}.pdf')
        plt.close(fig)
    methods=['popularity','source-only','target-only','fixed-0.5','dynamic-only','reliability-only','full']
    names=['Popularity','Source only','Target only*','Fixed 0.5','Count gate','Reliability gate','AdaFedGCDR']
    values=summary['cold']
    fig,ax=plt.subplots(figsize=(8.5,4),constrained_layout=True)
    means=[values[m]['mean'] for m in methods]
    sd=[values[m]['std'] or 0. for m in methods]
    ax.bar(np.arange(len(methods)),means,yerr=sd,capsize=3,color=['#a0a0a0']*6+['#c64f47'])
    ax.set_xticks(np.arange(len(methods)),names,rotation=22,ha='right')
    ax.set_ylabel('User-mean NDCG@10')
    ax.set_title(note+'Cold-start baselines and independently trained ablations')
    ax.grid(axis='y',alpha=.2)
    fig.supxlabel('*Target only averages k > 0; other methods include k = 0. Error bars: training-seed SD.',fontsize=8)
    fig.savefig(root/'cold-ablations.png',dpi=300)
    fig.savefig(root/'cold-ablations.pdf')
    plt.close(fig)
    (root/'figure-notes.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'error_bars':'sample standard deviation over independent training seeds','main_plot':'conditional on labels in training catalog; sparse additionally requires a known prefix','target_only':'k=0 excluded; aggregate not directly comparable to all-five-stage methods'},indent=2)+'\n')


if __name__=='__main__':main()
