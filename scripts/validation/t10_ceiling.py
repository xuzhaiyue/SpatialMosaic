"""T10 补充分析：实测程序分数的可达上限，以及 Visium 测量单元与图像视野不匹配的影响。
- 可靠性：每程序成员基因随机对半（20 次），两半 UCell 在 spot 间 Spearman，Spearman–Brown 校正为全集可靠性 R；
  模型与实测之间可达的最大相关 ≈ sqrt(R)（预测一侧视为无测量噪声）。报告 ρ / sqrt(R)。
- UCell 与 T4 / T8 完全相同（固定 9,036 背景、平均秩降序、maxRank 1500；HEST 缺失基因不补零，只用测到的背景）。
- 测量单元：Visium spot 直径 55 µm，训练用旧式 ST spot 100 µm、图像视野 128 µm。Visium 样本另算"spot + 6 个相邻 spot 计数合并"
  （中心距 100 µm 的一圈），预测一侧取同一组 spot 的均值。Visium HD 在 HEST 中已是 128 µm 方格，不再合并。
- 留出集：He2020 测试、HER2ST A–D（100 µm spot，不合并）、Meylan 测试（Visium，原始与合并两版）；HEST 6 张。
只读原始数据；不训练、不下载。"""
from t10_common import *
import h5py,collections
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import rankdata
import anndata as ad
bg=(R/'results/t3b/protein_coding_background.txt').read_text().split();members=json.loads((R/'results/t3b/program_members.json').read_text());mapping=json.loads((R/'references/2026-10-23_STPath_PathPrism/STPath/code/utils_data/symbol2ensembl.json').read_text())
MAXR=1500;REPS=20;rng0=20261031
def cap_ranks(X,present):
 """X: spots × present-background 计数（稀疏或稠密）。返回 union 成员基因的截断秩（降序平均秩，按测到的背景排序）。"""
 pidx={g:i for i,g in enumerate(present)};union=sorted({g for p in P for g in members[p] if g in pidx});out=np.empty((X.shape[0],len(union)),dtype='float32');cols=[pidx[g] for g in union]
 for s in range(0,X.shape[0],256):
  b=X[s:s+256];b=b.toarray() if sparse.issparse(b) else np.asarray(b);r=len(present)+1-rankdata(b,axis=1,method='average');out[s:s+len(b)]=np.minimum(r[:,cols],MAXR)
 return out,{g:i for i,g in enumerate(union)}
def ucell(cap,ui,genes):
 ix=[ui[g] for g in genes];k=len(ix);return 1-(cap[:,ix].sum(1,dtype='float64')-k*(k+1)/2)/(MAXR*k-k*(k+1)/2)
def reliability(cap,ui,p,seed):
 g=[x for x in members[p] if x in ui];rng=np.random.default_rng(seed);rs=[]
 for _ in range(REPS):
  perm=rng.permutation(g);h=len(perm)//2;a,b=ucell(cap,ui,perm[:h]),ucell(cap,ui,perm[h:]);r=corr(a,b);rs.append(2*r/(1+r) if np.isfinite(r) and r>-1 else np.nan)
 return float(np.nanmedian(rs)),len(g)
def neighbor_groups(xy_um,radius):
 t=cKDTree(xy_um);return t.query_ball_point(xy_um,radius)
def aggregate(X,groups):
 n=X.shape[0];rows=np.repeat(np.arange(n),[len(g) for g in groups]);cols=np.concatenate(groups);A=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n));return A@X
def evaluate(set_name,sample,patient,cap,ui,pred,support,seed):
 rows=[]
 for j,p in enumerate(P):
  rel,ng=reliability(cap,ui,p,seed+j);obs=ucell(cap,ui,[x for x in members[p] if x in ui]);r=corr(obs,pred[:,j]);ceil=np.sqrt(rel) if rel>0 else np.nan
  rows.append(dict(set=set_name,sample=sample,patient=patient,support=support,program=p,spots=len(obs),members_present=ng,split_half_reliability=rel,ceiling=ceil,rho=r,rho_over_ceiling=r/ceil if np.isfinite(ceil) else np.nan))
 return rows,obs
if __name__=='__main__':
 out=[];sp=pd.read_parquet(D/'st_spot_predictions.parquet');meta=clean(manifest());meta=meta[meta.split.isin(['test','external'])]
 # ---------- 训练域留出 ST ----------
 for k,row in enumerate(meta.itertuples()):
  a=ad.read_h5ad(row.rna);names=a.var_names.astype(str).tolist();raw=a.var.gene_ids.astype(str).tolist() if 'gene_ids' in a.var else names
  ids=[v.split('.')[0] if v.startswith('ENSG') else mapping.get(n) for n,v in zip(names,raw)];freq=collections.Counter(ids);index={g:i for i,g in enumerate(ids) if g in set(bg) and freq[g]==1};assert set(index)==set(bg)
  X=sparse.csr_matrix(a.X)[:,[index[g] for g in bg]];g=sp[sp['sample'].eq(row.sample)];assert np.array_equal(g.spot_id.to_numpy(),a.obs_names.astype(str).to_numpy())
  pred=g[[f'main_{s}' for s in SK]].to_numpy();cap,ui=cap_ranks(X,bg)
  full=np.column_stack([ucell(cap,ui,members[p]) for p in P]);assert np.allclose(full,g[[f'obs_{s}' for s in SK]].to_numpy(),atol=1e-6)  # 与 T4 目标一致
  r,_=evaluate(row.cohort,row.sample,row.patient,cap,ui,pred,'native',rng0+10*k);out+=r
  if row.cohort=='Meylan_ccRCC':
   rc=a.obs[['array_row','array_col']].to_numpy();lk={tuple(v):i for i,v in enumerate(rc)};grp=[[i]+[lk[t] for t in [tuple(v+o) for o in [(0,2),(0,-2),(1,1),(1,-1),(-1,1),(-1,-1)]] if t in lk] for i,v in enumerate(rc)]
   capA,uiA=cap_ranks(aggregate(X,grp),bg);predA=np.vstack([pred[x].mean(0) for x in grp]);r,_=evaluate(row.cohort,row.sample,row.patient,capA,uiA,predA,'spot+6 neighbours',rng0+10*k);out+=r
  print('ST',row.sample,flush=True)
 # ---------- HEST 6 张 ----------
 ps=pd.read_csv(R/'results/t8/paired_spots.csv');H=Path('/Volumes/INTENSO_BIG/HEST1k')
 from t8_measure import gene_columns,read_counts
 for k,s in enumerate(['TENX39','TENX53','TENX68','TENX161','TENX162','TENX180']):
  md=json.loads((H/'metadata'/f'{s}.json').read_text())
  with h5py.File(H/'st'/f'{s}.h5ad','r') as f:bar=[x.decode() if isinstance(x,bytes) else str(x) for x in f['obs']['_index'][:]];cols=gene_columns(f,mapping,bg)
  present=[x for x in bg if x in cols];X=read_counts(H/'st'/f'{s}.h5ad',np.array([cols[x] for x in present]))
  q=ps[ps['sample'].eq(s)].set_index('barcode');keep=np.array([b in q.index for b in bar]);Xk=X[keep];bk=[b for b in bar if b in q.index];qq=q.loc[bk]
  pred=qq[['E2F_predicted','IFNg_predicted','Hypoxia_predicted','EMT_predicted']].to_numpy();xy=qq[['x_um','y_um']].to_numpy()
  cap,ui=cap_ranks(Xk,present);r,obs=evaluate('HEST',s,md.get('st_technology'),cap,ui,pred,'native',rng0+1000+10*k);out+=r
  assert abs(corr(obs,pred[:,3])-r[3]['rho'])<1e-12
  if md.get('st_technology')=='Visium':
   grp=neighbor_groups(xy,105.0);capA,uiA=cap_ranks(aggregate(sparse.csr_matrix(Xk),grp),present);predA=np.vstack([pred[x].mean(0) for x in grp])
   r,_=evaluate('HEST',s,md.get('st_technology'),capA,uiA,predA,'spot+6 neighbours',rng0+1000+10*k);out+=r
  print('HEST',s,flush=True)
 df=pd.DataFrame(out);df.to_csv(D/'ceiling_by_section.csv',index=False)
 # 与 T8 报告核对（HEST native E2F / IFNγ ρ）
 chk=df[(df.set=='HEST')&(df.support=='native')].pivot(index='sample',columns='program',values='rho')[[P[0],P[1]]].round(3);print(chk,flush=True)
 st=df[df.set!='HEST'].groupby(['set','support','patient','program'],as_index=False)[['rho','ceiling','rho_over_ceiling','split_half_reliability']].mean()
 summ=st.groupby(['set','support','program'],as_index=False).agg(patients=('patient','nunique'),rho=('rho','median'),ceiling=('ceiling','median'),rho_over_ceiling=('rho_over_ceiling','median'),reliability=('split_half_reliability','median'))
 summ.to_csv(D/'ceiling_summary_ST.csv',index=False);print(summ.round(3).to_string(),flush=True)
 hs=df[df.set=='HEST'].pivot_table(index=['sample','support'],columns='program',values=['rho','ceiling']).round(3);hs.to_csv(D/'ceiling_HEST.csv');print(hs.to_string(),flush=True)
