#!/usr/bin/env python3
"""
w7 - figure variants. Run all, then choose.   python3 w7_figures.py [1|2|3|4|all]
Writes PNG (300 dpi) + PDF (vector) into figs/w7/ . Nothing here recomputes a result:
every number is read from w6_ladder_ascomycota.csv, w6_pergenus.csv, w1_c2c5_composition.csv.

DESIGN RULES APPLIED (same set for every panel, so they read as one system)
  one y-axis everywhere, never two scales   |   chance line drawn recessive, never as a series
  2 series max per panel, both direct-labelled AND in a legend, so identity is never colour-alone
  palette is a validated categorical pair - blue #2a78d6 / orange #eb6834 - CVD dE 24.7, all checks pass
  the scatter uses 3 slots (validated all-pairs) plus grey for everything else, with direct labels
"""
import sys, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT='figs/w7'; os.makedirs(OUT,exist_ok=True)
BLUE='#2a78d6'; ORANGE='#eb6834'; AQUA='#1baf7a'
INK='#0b0b0b'; INK2='#52514e'; MUTED='#8a8a85'; SURF='#fcfcfb'; GRID='#e3e3df'
plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID,'axes.linewidth':0.8,
    'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'axes.labelcolor':INK2,
    'figure.facecolor':SURF,'axes.facecolor':SURF,'savefig.facecolor':SURF,
    'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for ext in ('png','pdf'):
        fig.savefig(f'{OUT}/{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig); print(f'  wrote {OUT}/{name}.png / .pdf')
def chance(ax,x0,x1):
    ax.axhline(0.5,color=MUTED,lw=1,ls=(0,(4,3)),zorder=1)
    ax.text(x1,0.5,'  chance',va='center',ha='left',color=MUTED,fontsize=8)

ORDER=['98-99','97-98','95-97','92-95','88-92','84-88','80-84','70-80']
def titleblock(fig,ax,title,sub):
    ax.set_title(title,loc='left',fontsize=12.5,color=INK,pad=30)
    ax.text(0,1.045,sub,transform=ax.transAxes,fontsize=9,color=INK2,va='bottom')

def ladder():
    d=pd.read_csv('w6_ladder_ascomycota.csv').set_index('band').reindex(ORDER).dropna(how='all').reset_index()
    x=np.arange(len(d))
    # --- 1a
    fig,ax=plt.subplots(figsize=(9.0,4.8))
    ax.set_xlim(-0.6,len(d)-0.15); ax.set_ylim(0.20,1.06)
    ax.axvspan(-0.5,1.5,color='#eef3fb',zorder=0)
    ax.text(0.5,1.035,'barcode gap \u2014 a new species lives here',ha='center',va='center',
            fontsize=8.5,color=BLUE,zorder=5)
    ax.axhline(0.5,color=MUTED,lw=1,ls=(0,(4,3)),zorder=1)
    ax.text(-0.52,0.487,'chance',va='top',ha='left',color=MUTED,fontsize=8)
    ax.plot(x,d.auroc_emb,color=BLUE,lw=2,marker='o',ms=8,zorder=3,label='Evo2 embedding (held-out)')
    ax.plot(x,d.auroc_ppl,color=ORANGE,lw=2,marker='s',ms=8,zorder=3,label='Evo2 mean perplexity')
    ax.annotate('embedding',(x[2],d.auroc_emb.iloc[2]),xytext=(8,10),textcoords='offset points',
                color=BLUE,fontsize=9.5,fontweight='bold')
    ax.annotate('perplexity',(x[2],d.auroc_ppl.iloc[2]),xytext=(8,8),textcoords='offset points',
                color=ORANGE,fontsize=9.5,fontweight='bold')
    i=list(d.band).index('92-95')
    ax.annotate('perplexity inverts here \u2014\nreads further from the database\nlook MORE familiar to the model',
                (x[i],d.auroc_ppl.iloc[i]),xytext=(x[i]+0.18,0.635),textcoords='data',
                ha='left',va='center',color=ORANGE,fontsize=8.5,
                arrowprops=dict(arrowstyle='->',color=ORANGE,lw=1,shrinkB=6))
    ax.set_xticks(x); ax.set_xticklabels(d.band)
    ax.set_xlabel('% identity to the nearest full-MIDORI record   (further from the known world \u2192)',labelpad=8)
    ax.set_ylabel('AUROC vs reads at \u226599% identity')
    titleblock(fig,ax,'Evo2 sees species-level novelty in its embedding, not in its perplexity',
               'Brazilian soil eDNA, Ascomycota only \u2014 one library, one phylum, so taxonomic composition is held fixed')
    ax.grid(axis='y',color=GRID,lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    save(fig,'fig1a_ladder')
    # --- 1b
    fig,ax=plt.subplots(figsize=(9.0,4.4))
    w=0.38
    ax.set_xlim(-0.6,len(d)-0.4)
    ax.axvspan(-0.5,1.5,color='#eef3fb',zorder=0)
    ax.bar(x-w/2,d.auroc_emb-0.5,w-0.04,color=BLUE,label='Evo2 embedding (held-out)',zorder=3)
    ax.bar(x+w/2,d.auroc_ppl-0.5,w-0.04,color=ORANGE,label='Evo2 mean perplexity',zorder=3)
    ax.axhline(0,color=INK2,lw=1,zorder=4)
    ax.text(0.5,0.52,'barcode gap',ha='center',fontsize=8.5,color=BLUE)
    ax.annotate('below the line =\npoints the wrong way',(x[4]+w/2,d.auroc_ppl.iloc[4]-0.5),
                xytext=(6,-6),textcoords='offset points',ha='left',va='top',
                fontsize=8.5,color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(d.band)
    ax.set_ylabel('AUROC \u2212 0.5')
    ax.set_xlabel('% identity to the nearest full-MIDORI record',labelpad=8)
    titleblock(fig,ax,'Deviation from chance, by distance from the database',
               'the same numbers as the line chart \u2014 perplexity does not merely fail, it inverts')
    ax.grid(axis='y',color=GRID,lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='upper left',fontsize=9,bbox_to_anchor=(0.005,0.99))
    save(fig,'fig1b_ladder_deviation')

def genus():
    d=pd.read_csv('w6_pergenus.csv')
    bands=[b for b in ORDER if b in set(d.band)]
    gl=sorted(set(d.genus))
    fig,axes=plt.subplots(1,len(bands),figsize=(3.3*len(bands),2.9),sharey=True)
    if len(bands)==1: axes=[axes]
    for k,(ax,b) in enumerate(zip(axes,bands)):
        s=d[d.band==b].set_index('genus').reindex(gl).reset_index()
        y=np.arange(len(s))
        for j,row in s.iterrows():
            ax.plot([row.null_lo,row.null_hi],[j,j],color='#dcdcd7',lw=13,solid_capstyle='butt',zorder=1)
        ax.plot(s.emb,y,'o',color=BLUE,ms=10,zorder=3,label='Evo2 embedding (held-out)')
        ax.plot(s.ppl,y,'s',color=ORANGE,ms=9,zorder=3,label='Evo2 mean perplexity')
        for j,row in s.iterrows():
            ax.annotate(f'{row.emb:.2f}',(row.emb,j),xytext=(0,11),textcoords='offset points',
                        ha='center',fontsize=8.5,color=BLUE,fontweight='bold')
        ax.axvline(0.5,color=MUTED,lw=1,ls=(0,(4,3)),zorder=0)
        ax.set_yticks(y); ax.set_yticklabels(s.genus,style='italic',fontsize=10)
        ax.set_ylim(-0.6,len(s)-0.25); ax.set_xlim(0.15,1.0)
        ax.set_title(f'{b} % identity',fontsize=10,color=INK2,pad=6)
        ax.grid(axis='x',color=GRID,lw=0.6); ax.set_axisbelow(True)
        if k==1: ax.set_xlabel('AUROC vs that genus\u2019s own \u226599% reads',labelpad=6)
    fig.suptitle('Within a single fungal genus, the embedding separates new species from known ones',
                 x=0.005,ha='left',fontsize=12.5,y=1.33)
    fig.text(0.005,1.09,'grey band = null (that genus\u2019s \u226599% reads split against themselves)  \u00b7  '
             'dashed line = chance\nevery blue point clears its null  \u00b7  7 of the 9 orange points sit below chance',
             fontsize=9,color=INK2,ha='left',va='bottom')
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,frameon=False,fontsize=9.5,ncol=2,loc='lower center',bbox_to_anchor=(0.5,-0.30))
    save(fig,'fig2_within_genus')

def themap():
    Z=np.load('mapsuite_sample.npz',allow_pickle=True); X=Z['X128']; L=Z['labels']
    COI=['REAL:ref_anchor','REAL:midori','REAL:edna_brazil','REAL:edna_abyssal']
    OFF=['REAL:abyssal_off401','REAL:abyssal_off404','REAL:abyssal_off368']
    groups=[('real COI\n(reference + MIDORI + eDNA)',COI,BLUE),
            ('real off-target\n(bacterial / non-COI)',OFF,ORANGE),
            ('simulated 16S insert (C3)',['SIM:C3_insert'],AQUA)]
    def panel(P,tag,xl,yl,direct):
        fig,ax=plt.subplots(figsize=(7.0,6.0))
        other=~np.isin(L,sum([g[1] for g in groups],[]))
        ax.scatter(P[other,0],P[other,1],s=3,c='#dcdcd7',lw=0,alpha=.5,zorder=1,label='all other cohorts')
        for name,keys,c in groups:
            mk=np.isin(L,keys)
            ax.scatter(P[mk,0],P[mk,1],s=5,c=c,lw=0,alpha=.55,zorder=2,label=name.replace('\n',' '))
        if direct:
            for name,keys,c in groups:
                mk=np.isin(L,keys)
                ax.annotate(name,(np.median(P[mk,0]),np.median(P[mk,1])),fontsize=9.5,color=c,
                            fontweight='bold',ha='center',va='center',
                            bbox=dict(fc=SURF,ec='none',alpha=.82,pad=2.0),zorder=5)
            leg=ax.legend(frameon=False,fontsize=8,loc='lower right',markerscale=3,
                          handletextpad=.3,labelspacing=.3)
        else:
            leg=ax.legend(frameon=True,fontsize=9,loc='upper left',markerscale=4,
                          handletextpad=.4,labelspacing=.5,facecolor=SURF,edgecolor=GRID,framealpha=.95)
            for t,c in zip(leg.get_texts(),['#52514e',BLUE,ORANGE,AQUA]): t.set_color(c)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xticks([]); ax.set_yticks([])
        titleblock(fig,ax,f'Evo2 embedding map \u2014 {tag}',
                   'one shared space: 8 real classes (H200) and 34 simulated cohorts (H100), '
                   'co-embedded after a bit-exact anchor check')
        save(fig,f'fig3_map_{tag.lower()}')
    panel(X,'PCA','PC1  (79.9% of variance \u2014 this axis is embedding magnitude)','PC2  (6.1%)',False)
    try:
        import umap
        P=umap.UMAP(n_neighbors=25,min_dist=.1,random_state=20260920).fit_transform(X[:,:40])
        np.save('w8_umap_coords.npy',P)
        panel(P,'UMAP','UMAP 1','UMAP 2',True)
    except Exception as e:
        print('  UMAP skipped:',e)

w=sys.argv[1] if len(sys.argv)>1 else 'all'
if w in ('1','all'): ladder()
if w in ('2','all'): genus()
if w in ('3','all'): themap()
if w in ('4','all'): chimera()
print('done')
