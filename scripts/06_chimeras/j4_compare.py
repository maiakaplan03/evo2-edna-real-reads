#!/usr/bin/env python3
"""j4 - score Evo2's logodds channel against uchime on the identical reads."""
import numpy as np, pandas as pd, os

def rank(a):
    n=a.size;o=np.argsort(a,kind='mergesort');r=np.empty(n);r[o]=np.arange(1,n+1)
    i=0
    while i<n:
        j=i
        while j+1<n and a[o[j+1]]==a[o[i]]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2
        i=j+1
    return r
def auroc(p,n):
    p=np.asarray(p,float);n=np.asarray(n,float)
    if p.size==0 or n.size==0: return float('nan')
    return (rank(np.concatenate([p,n]))[:p.size].sum()-p.size*(p.size+1)/2)/(p.size*n.size)

lab = pd.read_csv('j3_labels.csv')
UC = {'uchime_ref':'j3_uchime_ref.tsv', 'uchime_denovo':'j3_uchime_denovo.tsv'}
COLS = ['score','Q','A','B','T','idQM','idQA','idQB','idAB','idQT',
        'LY','LN','LA','RY','RN','RA','div','YN']
for name, path in UC.items():
    if not os.path.exists(path):
        lab[name] = np.nan; lab[name+'_call'] = np.nan; continue
    u = pd.read_csv(path, sep='\t', header=None, names=COLS, usecols=range(18))
    u['id'] = u.Q.astype(str).str.split(';').str[0]
    u = u.drop_duplicates('id')
    m = dict(zip(u.id, u.score)); c = dict(zip(u.id, u.YN))
    lab[name] = lab.id.map(m).fillna(0.0)          # no hit reported = no evidence = 0
    lab[name+'_call'] = lab.id.map(c).fillna('N').eq('Y').astype(int)

neg = lab[lab.is_chimera == 0]
print('='*86)
print('DETECTION - identical reads, identical negatives (224 composition-matched C0_REF)')
print('='*86)
print(f'{"cohort":<16}{"n":>6}{"Evo2 logodds":>15}{"uchime_ref":>13}{"uchime_denovo":>15}')
print('-'*86)
for c in ['C2_chimera_T6','C2_chimera_T5']:
    g = lab[lab.cohort == c]
    print(f'{c:<16}{len(g):>6}'
          f'{auroc(g.evo2_score, neg.evo2_score):>15.3f}'
          f'{auroc(g.uchime_ref, neg.uchime_ref):>13.3f}'
          f'{auroc(g.uchime_denovo, neg.uchime_denovo):>15.3f}')
print('-'*86)
print('AUROC. 0.500 = cannot tell a chimera from a clean reference read.')
print('⛔ uchime_ref was given references_N1565.fasta, i.e. the actual parent database.')
print('   Evo2 saw nothing but the read.\n')

print('FLAG RATE at each tool\'s own default threshold (what a user would actually get):')
print(f'{"cohort":<16}{"uchime_ref":>13}{"uchime_denovo":>15}')
for c in ['C2_chimera_T6','C2_chimera_T5','C0_REF']:
    g = lab[lab.cohort == c]
    print(f'{c:<16}{100*g.uchime_ref_call.mean():>12.1f}%{100*g.uchime_denovo_call.mean():>14.1f}%')
print('  (the C0_REF row is the false-positive rate - lower is better there)\n')

print('='*86)
print('LOCALISATION - Evo2 only; uchime reports no breakpoint coordinate')
print('='*86)
for c in ['C2_chimera_T6','C2_chimera_T5']:
    g = lab[(lab.cohort == c) & lab.evo2_err_bp.notna()]
    print(f'{c:<16} median |error| {g.evo2_err_bp.median():>5.0f} bp   '
          f'within 20 bp {100*(g.evo2_err_bp<=20).mean():>5.1f} %   '
          f'within 37 bp {100*(g.evo2_err_bp<=37).mean():>5.1f} %')
print('\nuchime\'s output format carries no crossover coordinate, so there is no')
print('head-to-head on localisation. The scale references from the MBE 2024')
print('benchmark - 3SEQ 37 bp, MaxChi 74 bp, GARD 143 bp MEAN error, methods GIVEN')
print('the parents - are on different data and different read lengths. Quote them')
print('as an order-of-magnitude reference, never as a head-to-head.')
lab.to_csv('j4_benchmark_scored.csv', index=False)
print('\nwrote j4_benchmark_scored.csv')
