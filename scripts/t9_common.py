"""T9-3 程序图谱：基因集、路径、通用 K 程序 UCell。背景、maxRank、映射与 T3b/T4/T5 相同。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',MPLBACKEND='Agg')
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1];O=R/'results/t9/atlas';O.mkdir(parents=True,exist_ok=True)
E=Path('/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts/t9');E.mkdir(parents=True,exist_ok=True)
MAX_RANK=1500
BG=(R/'results/t3b/protein_coding_background.txt').read_text().split();BGSET=set(BG)
ORIGINAL=['E2F_TARGETS','INTERFERON_GAMMA_RESPONSE','HYPOXIA','EPITHELIAL_MESENCHYMAL_TRANSITION']
CELLTYPES={'CELL_T':'CD3D CD3E CD3G CD2 CD247 TRAC','CELL_CD8':'CD8A CD8B GZMK GZMA','CELL_B':'MS4A1 CD79A CD79B CD19 BANK1','CELL_PLASMA':'MZB1 JCHAIN TNFRSF17 DERL3',
 'CELL_MYELOID':'CD68 CD163 C1QA C1QB C1QC CSF1R','CELL_FIBROBLAST':'COL1A1 COL1A2 COL3A1 DCN LUM PDGFRB','CELL_ENDOTHELIAL':'PECAM1 VWF CDH5 CLDN5 EMCN','CELL_EPITHELIAL':'EPCAM KRT8 KRT18 KRT19 CDH1'}
ALIAS={'FASTKD4':'TBRG4','RIGI':'DDX58','TMT1B':'METTL7B','HACL2':'ILVBL','NUCLEOLIN':'NCL','PRP4K':'PRPF4B'}  # 与 t3b_background.py 相同
def gene_sets():
 """原 4 个程序直接取 T3b 冻结成员；其余 Hallmark 与细胞类型按 T3b 同一规则（别名 -> STPath 映射），映射不到时退回 GENCODE v36 背景符号。"""
 mapping=json.loads((R/'references/2026-10-23_STPath_PathPrism/STPath/code/utils_data/symbol2ensembl.json').read_text())
 bgsym={}
 for r in pd.read_csv(R/'results/t3b/protein_coding_background.csv').itertuples():
  for s in str(r.gencode_symbols).split(';'):bgsym.setdefault(s,r.ensembl_id)
 frozen=json.loads((R/'results/t3b/program_members.json').read_text())
 raw={}
 for line in (R/'references/gene_sets/msigdb_2026.2.Hs/h.all.v2026.2.Hs.symbols.gmt').read_text().splitlines():
  parts=line.split('\t');raw[parts[0].replace('HALLMARK_','')]=parts[2:]
 for k,v in CELLTYPES.items():raw[k]=v.split()
 members={};rows=[]
 for p,syms in raw.items():
  if p in ORIGINAL:members[p]=frozen[p];rows.append(dict(program=p,listed=len(syms),members=len(frozen[p]),source='T3b_frozen'));continue
  keep=[]
  for s in syms:
   a=ALIAS.get(s,s);g=mapping.get(a)
   if g not in BGSET:g=bgsym.get(a)
   if g in BGSET:keep.append(g)
  members[p]=sorted(set(keep));rows.append(dict(program=p,listed=len(syms),members=len(members[p]),source='celltype_markers' if p.startswith('CELL_') else 'Hallmark'))
 return members,pd.DataFrame(rows)
def ucell_from_ranks(rank,bgindex,members,programs):
 """rank: 降序平均并列秩并已截断到 MAX_RANK（行=样本，列=背景基因）。公式同 T4/T5。"""
 out=np.empty((rank.shape[0],len(programs)))
 for j,p in enumerate(programs):
  ix=[bgindex[g] for g in members[p]];n=len(ix);out[:,j]=1-(rank[:,ix].sum(1,dtype='float64')-n*(n+1)/2)/(MAX_RANK*n-n*(n+1)/2)
 return out
def capped_rank(values):return np.minimum(len(BG)+1-rankdata(values,axis=1,method='average'),MAX_RANK)
