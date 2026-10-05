#!/usr/bin/env python3
"""s29 figures: (A) ridgeline of all real classes, (B) all-pairs bells+AUROC matrix,
(C) per-position perplexity profile across simulated cohorts (style of the divergence figure)."""
import numpy as np, pandas as pd, csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Helvetica Neue','Helvetica','DejaVu Sans'],
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfcfb',
                     'axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb'})
INK='#0b0b0b'; INK2='#52514e'; MUTED='#898781'; GRID='#e1e0d9'; AXIS='#c3c2b7'
BLUE=['#86b6ef','#5598e7','#2a78d6','#1c5cab','#104281']          # validated ordinal ramp, COI family
ORANGE=['#f19a6b','#eb6834','#a33b12']                             # validated ordinal ramp, off-target family
GREY='#6f6d67'                                                     # neutral reference (achromatic by design)

B="../analysis_s28/scoring_set/"
p=pd.read_csv(B+'perplexity_final_2026-09-19.csv')
m=pd.read_csv(B+'real_scoring_set_2026-09-17_manifest.tsv',sep='\t',usecols=['read_id','classes'])
d=p.merge(m,left_on='Query_ID',right_on='read_id'); d['cls']=d['classes'].fillna('UNLABELLED')
real={c:g['Perplexity'].to_numpy() for c,g in d.groupby('cls') if len(g)>=100}

LABEL={'ref_anchor':'Reference anchors','midori':'MIDORI database COI','edna_brazil':'eDNA Brazil (soil)',
       'edna_abyssal':'eDNA abyssal (sediment)','brazil_stop':'eDNA Brazil, stop codons',
       'abyssal_off404':'Off-target 404 bp','abyssal_off401':'Off-target 401 bp','abyssal_off368':'Off-target 368 bp'}
FAM ={'ref_anchor':'ref','midori':'coi','edna_brazil':'coi','edna_abyssal':'coi','brazil_stop':'coi',
      'abyssal_off404':'off','abyssal_off401':'off','abyssal_off368':'off'}
COLOR={'ref_anchor':GREY,'midori':BLUE[0],'edna_brazil':BLUE[1],'edna_abyssal':BLUE[2],'brazil_stop':BLUE[3],
       'abyssal_off404':ORANGE[0],'abyssal_off401':ORANGE[1],'abyssal_off368':ORANGE[2]}

def rankdata_avg(a):
    n=a.size; o=np.argsort(a,kind='mergesort'); s=a[o]; r=np.empty(n); r[o]=np.arange(1,n+1)
    i=0
    while i<n:
        j=i
        while j+1<n and s[j+1]==s[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2
        i=j+1
    return r
def auroc(x,y):
    a=np.concatenate([x,y]); r=rankdata_avg(a); nx=x.size
    return (r[:nx].sum()-nx*(nx+1)/2)/(nx*y.size)
BINS=np.linspace(1.0,4.0,601)
def ovl(x,y):
    hx,_=np.histogram(x,bins=BINS,density=True); hy,_=np.histogram(y,bins=BINS,density=True)
    return float(np.minimum(hx,hy).sum()*(BINS[1]-BINS[0]))
def dens(x,lo=1.3,hi=3.9,nb=420,smooth=17):
    e=np.linspace(lo,hi,nb+1); h,_=np.histogram(x,bins=e,density=True)
    k=np.ones(smooth)/smooth; h=np.convolve(np.pad(h,(smooth//2,smooth//2),mode='edge'),k,mode='valid')[:nb]
    return (e[:-1]+e[1:])/2, h

order=sorted(real, key=lambda c: np.median(real[c]))

# ---------------- FIG A - ridgeline ----------------
fig,ax=plt.subplots(figsize=(12,7.6))
GAP=1.0; AMP=0.80
for i,c in enumerate(order):
    x=real[c]; xs,h=dens(x); h=h/h.max()*AMP
    base=i*GAP
    ax.fill_between(xs, base, base+h, color=COLOR[c], alpha=.90, lw=0, zorder=3+2*i)
    ax.plot(xs, base+h, color='#fcfcfb', lw=2.2, zorder=3+2*i)
    ax.plot(xs, base+h, color=COLOR[c], lw=1.3, zorder=4+2*i)
    med=np.median(x)
    ax.plot([med,med],[base,base+np.interp(med,xs,h)],color='#fcfcfb',lw=2.8,zorder=5+2*i)
    ax.plot([med,med],[base,base+np.interp(med,xs,h)],color=INK,lw=1.1,zorder=6+2*i)
    ax.axhline(base, color=GRID, lw=.7, zorder=1)
    ax.text(3.88, base+0.30, f"median {med:.3f}", ha='right', va='center', fontsize=9.5,
            color=INK2, zorder=40)
    ax.text(3.88, base+0.12, f"n = {x.size:,}", ha='right', va='center', fontsize=8.5,
            color=MUTED, zorder=40)
ax.set_xlim(1.3,3.92); ax.set_ylim(-0.12, len(order)*GAP+0.75)
ax.set_yticks([i*GAP+0.16 for i in range(len(order))])
ax.set_yticklabels([LABEL[c] for c in order], fontsize=11.5, color=INK)
for t,c in zip(ax.get_yticklabels(), order): t.set_color(INK)
ax.tick_params(axis='y', length=0, pad=10)
ax.spines['left'].set_visible(False); ax.spines['bottom'].set_color(AXIS)
ax.tick_params(axis='x', colors=MUTED, labelsize=10)
ax.set_xlabel('Mean per-base perplexity   exp(NLL)', fontsize=11.5, color=INK2, labelpad=8)
for v in [1.5,2.0,2.5,3.0,3.5]: ax.axvline(v,color=GRID,lw=.7,zorder=0)
ax.set_title('Evo2 perplexity separates off-target amplicons from COI —\nbut not novel COI from known COI',
             fontsize=15, color=INK, loc='left', pad=30, weight='medium')
ax.text(0,1.045,'Real eDNA scoring set · 200,898 unique 312-bp sequences · H200 run 2026-09-19 · area-normalised density · vertical rule = median',
        transform=ax.transAxes, fontsize=9, color=MUTED, ha='left')
leg=[('COI-like',BLUE[3]),('off-target',ORANGE[2]),('reference',GREY)]
for k,(lab,col) in enumerate(leg):
    ax.add_patch(Rectangle((1.35,len(order)*GAP+0.52-0.17*k),0.055,0.085,color=col,lw=0,clip_on=False))
    ax.text(1.42,len(order)*GAP+0.562-0.17*k,lab,va='center',ha='left',fontsize=10,color=INK2)
plt.tight_layout(); plt.savefig('figs/figA_ridgeline_real_classes.png',dpi=220,bbox_inches='tight')
plt.savefig('figs/figA_ridgeline_real_classes.pdf',bbox_inches='tight'); plt.close()
print("FIG A done")

# ---------------- FIG B - all-pairs matrix ----------------
K=len(order); SH=list(reversed(order))          # highest median first, reads top-left to bottom-right
fig,axes=plt.subplots(K,K,figsize=(17,15.5))
AU=np.zeros((K,K)); OV=np.zeros((K,K))
for i,ri in enumerate(SH):
    for j,cj in enumerate(SH):
        a=axes[i,j]; a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values(): sp.set_color(GRID); sp.set_lw(.8)
        if i==j:
            xs,h=dens(real[ri]); h=h/h.max()
            a.fill_between(xs,0,h,color=COLOR[ri],alpha=.9,lw=0); a.plot(xs,h,color=COLOR[ri],lw=1.2)
            a.set_xlim(1.3,3.9); a.set_ylim(0,1.35)
            a.text(.5,.90,LABEL[ri],transform=a.transAxes,ha='center',va='top',fontsize=9.5,color=INK,weight='medium')
            a.text(.5,.66,f"n={real[ri].size:,}",transform=a.transAxes,ha='center',va='top',fontsize=8,color=MUTED)
            a.set_facecolor('#f4f3ef')
        elif i<j:                                # upper triangle: the two densities overlaid
            for cc in (ri,cj):
                xs,h=dens(real[cc]); h=h/h.max()
                a.fill_between(xs,0,h,color=COLOR[cc],alpha=.55,lw=0)
                a.plot(xs,h,color=COLOR[cc],lw=1.3)
            a.set_xlim(1.3,3.9); a.set_ylim(0,1.18)
        else:                                    # lower triangle: the statistics
            au=auroc(real[ri],real[cj]); ov=ovl(real[ri],real[cj])
            AU[i,j]=au; OV[i,j]=ov
            sep=abs(au-0.5)*2                    # 0 = identical, 1 = perfectly separated
            shade=plt.cm.Blues(0.10+0.62*sep)
            a.set_facecolor(shade)
            a.text(.5,.62,f"{au:.3f}",transform=a.transAxes,ha='center',va='center',
                   fontsize=15,color=INK if sep<.55 else '#ffffff',weight='medium')
            a.text(.5,.27,f"OVL {ov:.2f}",transform=a.transAxes,ha='center',va='center',
                   fontsize=9,color=INK2 if sep<.55 else '#e8eef7')
        if i==K-1: a.set_xlabel(LABEL[cj].replace(' (','\n('),fontsize=8.5,color=INK2,labelpad=5)
        if j==0:   a.set_ylabel(LABEL[ri].replace(' (','\n('),fontsize=8.5,color=INK2,labelpad=5)
fig.suptitle('Every pair of classes: overlapping densities (upper) and separation statistics (lower)',
             fontsize=16,color=INK,x=0.07,ha='left',y=0.975,weight='medium')
fig.text(0.07,0.952,'AUROC = P(row sequence scores higher than column sequence); 0.5 = indistinguishable. '
                    'OVL = shared area under the two densities; 1.0 = identical. '
                    'Cell shade encodes |AUROC-0.5|. x-axis of every density panel: perplexity 1.3-3.9.',
         fontsize=9.5,color=MUTED,ha='left')
plt.tight_layout(rect=[0.01,0.01,0.99,0.94])
plt.savefig('figs/figB_pairwise_matrix.png',dpi=170,bbox_inches='tight')
plt.savefig('figs/figB_pairwise_matrix.pdf',bbox_inches='tight'); plt.close()
print("FIG B done")

# ---------------- FIG C - per-position profiles, simulated ----------------
Z=np.load('perbase_phase1_profiles.npz')
def ramp(n, dark='#0d366b', light='#86b6ef'):
    import matplotlib.colors as mc
    c1=np.array(mc.to_rgb(light)); c2=np.array(mc.to_rgb(dark))
    return [mc.to_hex(c1+(c2-c1)*t) for t in np.linspace(0,1,n)]

def profile_panel(cohorts, colors, labels, title, sub, fname, band=True):
    fig,ax=plt.subplots(figsize=(13.5,7.4))
    ax.add_patch(Rectangle((0,0),50,10,color='#efeee8',lw=0,zorder=0))
    ax.set_ylim(1.0,5.4)
    ax.text(25,1.06,'warm-up\nzone',ha='center',va='bottom',fontsize=9,color=MUTED,style='italic',zorder=2)
    ends=[]
    def roll(v,w=5):
        return np.convolve(np.pad(v,(w//2,w//2),mode='edge'),np.ones(w)/w,mode='valid')[:v.size]
    for c,col,lab in zip(cohorts,colors,labels):
        mu=np.exp(roll(Z[f"{c}__mean"]))      # exp(mean NLL) = geometric-mean per-base perplexity
        x=np.arange(1,mu.size+1)
        if band and (c==cohorts[0] or c==cohorts[-1]):   # bands only on the two extremes; 9 stacked IQRs is mud
            ax.fill_between(x,np.exp(roll(Z[f"{c}__q25"])),np.exp(roll(Z[f"{c}__q75"])),
                            color=col,alpha=.13,lw=0,zorder=2)
        ax.plot(x,mu,color=col,lw=1.9,zorder=4,solid_capstyle='round')
        ends.append((mu[250:300].mean(),col,lab,float(np.exp(Z[f"{c}__mean"][50:].mean()))))
    ends.sort(key=lambda t:-t[0])
    y0=ax.get_ylim()
    for k,(yv,col,lab,m) in enumerate(ends):
        ax.text(1.012, 0.93-0.052*k, f"{lab}", transform=ax.transAxes, color=col,
                fontsize=10.5, va='center', ha='left', weight='medium')
        ax.text(1.29, 0.93-0.052*k, f"{m:.2f}", transform=ax.transAxes, color=col,
                fontsize=10.5, va='center', ha='right')
    ax.text(1.012,0.985,'cohort', transform=ax.transAxes, color=MUTED, fontsize=9, ha='left')
    ax.text(1.29,0.985,'mean', transform=ax.transAxes, color=MUTED, fontsize=9, ha='right')
    ax.set_xlim(0,312); ax.set_ylim(1.0,5.4)
    ax.grid(axis='y',color=GRID,lw=.7); ax.set_axisbelow(True)
    ax.spines['bottom'].set_color(AXIS); ax.spines['left'].set_color(AXIS)
    ax.tick_params(colors=MUTED,labelsize=10)
    ax.set_xlabel('Position in read (bp)',fontsize=12,color=INK2,labelpad=8)
    ax.set_ylabel('Per-base perplexity  exp(NLL)',fontsize=12,color=INK2,labelpad=8)
    ax.set_title(title,fontsize=15,color=INK,loc='left',pad=26,weight='medium')
    ax.text(0,1.035,sub,transform=ax.transAxes,fontsize=9.5,color=MUTED,ha='left')
    plt.savefig(f'figs/{fname}.png',dpi=220,bbox_inches='tight')
    plt.savefig(f'figs/{fname}.pdf',bbox_inches='tight'); plt.close()

nov=['C0_REF']+[f'C6_pyvolve_Csc-T{i}' for i in range(1,9)]
profile_panel(nov,[GREY]+ramp(8),['Reference (C0)']+[f'Csc-T{i}' for i in range(1,9)],
  'Simulated novelty does NOT lift the profile uniformly: +0.33 nats in positions 51-150 vs +0.15 after,\na 2.2x front-loaded response at the deepest tier',
  'phase1 H100 · geometric-mean per-base perplexity, 2,500-read subsample per cohort · 5-bp rolling mean · IQR band on the two extreme cohorts only · mean column = positions 51-311',
  'figC1_profile_novelty_ladder')

con=['C0_REF','C1_illumina','C2_chimera_T2','C2_chimera_T6','C4_briggs_low','C4_briggs_high','C3_insert']
profile_panel(con,[GREY]+ramp(6,dark='#7a2408',light='#f19a6b'),
  ['Reference (C0)','C1 Illumina error','C2 chimera T2','C2 chimera T6','C4 damage low','C4 damage high','C3 16S insert'],
  'Each contamination type deforms the profile differently: damage uniformly (1.0x),\nthe 16S insert back-loaded (0.55x), chimeras DOWNWARD',
  'phase1 H100 · geometric-mean per-base perplexity, 2,500-read subsample per cohort · 5-bp rolling mean · IQR band on the two extreme cohorts only · mean column = positions 51-311',
  'figC2_profile_contamination')
print("FIG C done")
