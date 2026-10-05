#!/usr/bin/env python3
"""w11 - three clean versions of the Fig 1 ladder for the PI: A alone, D alone, A and D together.
Same reads, folds, seeds and bootstrap as w9 (numbers read from w9_results.csv, nothing recomputed).
A = centred, 128 PCs, L2 (the geometry of the original figure). D = same, each PC divided by its spread."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
BG='#fbfbf9'; INK='#1f2328'; SUB='#555b63'; GRID='#e6e6e3'; NULLC='#dcddd8'; GAP='#eef2fb'
COL={'A_pca128':'#2a78d6','D_pca128_wht':'#c0368c'}; MK={'A_pca128':'o','D_pca128_wht':'D'}
NAME={'A_pca128':'A · centred embedding (original)','D_pca128_wht':'D · centred + standardised embedding'}
plt.rcParams.update({'font.family':'DejaVu Sans','axes.edgecolor':'#cfd1cc','axes.labelcolor':SUB,
    'xtick.color':SUB,'ytick.color':SUB,'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG})
d=pd.read_csv('w9_results.csv'); d=d[d.fig=='fig1']
BANDS=['98-99','97-98','95-97','92-95','88-92','84-88','80-84','70-80']
NS=[2946,2281,4098,8593,22761,5689,282,177]; x=np.arange(8)
def get(v): s=d[d.variant==v].set_index('comparison'); return s.loc[BANDS], s.loc['null >=99 vs itself']
CAPTION=("Each point is a separate comparison: the reads in that identity band (novel) vs the same known reads —\n"
         "Ascomycota reads ≥99 % identical to a MIDORI record (2,000 drawn at random from 3,569). n = reads in the band;\n"
         "bands larger than 2,000 were randomly subsampled to 2,000. Point = median AUROC over 15 held-out splits ·\n"
         "error bar = 95 % bootstrap CI (200 resamples of reads) · grey band = null (known reads split against themselves) ·\n"
         "blue shading = barcode gap (95–99 %), where a new species is expected.")
def base(title,sub):
    fig,ax=plt.subplots(figsize=(11,7.2)); fig.subplots_adjust(top=.84,bottom=.30,left=.08,right=.97)
    ax.axvspan(-.5,2.5,color=GAP,zorder=0,lw=0)
    ax.grid(axis='y',color=GRID,lw=.8); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.axhline(.5,color='#8a8f96',ls='--',lw=1,zorder=2)
    ax.set_xticks(x,[f'{b} %\nn = {n:,}' for b,n in zip(BANDS,NS)],fontsize=10.5)
    ax.set_xlim(-.5,7.5); ax.set_ylim(.4,1.04); ax.set_yticks(np.arange(.4,1.01,.1))
    ax.set_ylabel('AUROC  (novel band vs known reads)',fontsize=11.5)
    ax.set_xlabel('% identity of the novel reads to their nearest MIDORI record   →  further from the known world',fontsize=11.5,labelpad=8)
    fig.text(.08,.955,title,fontsize=15.5,color=INK,va='top')
    fig.text(.08,.905,sub,fontsize=11,color=SUB,va='top')
    fig.text(.08,.035,CAPTION,fontsize=9.3,color=SUB,va='bottom',linespacing=1.5)
    return fig,ax
def draw(ax,v,dx=0,lab='above'):
    s,n=get(v); c=COL[v]
    ax.axhspan(n.null_lo,n.null_hi,color=NULLC,zorder=1,lw=0) if dx==0 else None
    ax.errorbar(x+dx,s.auroc,yerr=[s.auroc-s.ci_lo,s.ci_hi-s.auroc],color=c,lw=2.2,marker=MK[v],ms=8.5,
                capsize=4,capthick=1.5,elinewidth=1.5,mec=BG,mew=1.3,zorder=4,label=NAME[v])
    for xi,(a,lo,hi) in enumerate(zip(s.auroc,s.ci_lo,s.ci_hi)):
        if lab=='above': ax.text(xi+dx,hi+.014,f'{a:.2f}',ha='center',va='bottom',fontsize=10,fontweight='bold',color=c)
        else:            ax.text(xi+dx,lo-.014,f'{a:.2f}',ha='center',va='top',fontsize=10,fontweight='bold',color=c)
    return n
def save(fig,name):
    for e in ('pdf','png'): fig.savefig(f'figs/w11/{name}.{e}',dpi=200)
    plt.close(fig)
import os; os.makedirs('figs/w11',exist_ok=True)
fig,ax=base('A · Evo2 embedding, centred (original geometry)',
            'Brazilian soil eDNA, Ascomycota only — taxonomic composition held fixed')
n=draw(ax,'A_pca128'); save(fig,'ladder_A_centred')
fig,ax=base('D · Evo2 embedding, centred and standardised',
            'Same reads, splits and seeds as A — the only change: each of the 128 directions is divided by its spread')
n=draw(ax,'D_pca128_wht'); save(fig,'ladder_D_standardised')
fig,ax=base('A vs D · the same comparisons under the two geometries',
            'Brazilian soil eDNA, Ascomycota only — identical reads, splits and seeds; only the geometry differs')
sa,na=get('A_pca128'); sd,nd=get('D_pca128_wht')
ax.axhspan(min(na.null_lo,nd.null_lo),max(na.null_hi,nd.null_hi),color=NULLC,zorder=1,lw=0)
draw(ax,'A_pca128',dx=-.09,lab='below'); draw(ax,'D_pca128_wht',dx=.09,lab='above')
ax.legend(loc='upper left',frameon=False,fontsize=10.5)
save(fig,'ladder_A_vs_D_overlay')
print('ok'); print(get('D_pca128_wht')[1][['auroc','null_lo','null_hi']])
