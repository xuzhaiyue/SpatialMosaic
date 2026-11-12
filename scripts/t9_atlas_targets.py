"""T9-3 第一步：58 个程序的 ST spot 与 bulk 患者 UCell。读取与基因匹配逐行照搬 t4_targets_discovery.py / t5_st_bulk.py。
回归自检：原 4 个程序必须与 T4 spot 目标、T5 bulk 患者目标一致（max |diff| < 1e-9），否则中止。"""
from t9_common import *
import collections,time
import anndata as ad
from scipy import sparse
members,table=gene_sets();PROGRAMS=list(members);table.to_csv(O/'program_members_summary.csv',index=False)
(O/'program_members.json').write_text(json.dumps(members,indent=1))
bgindex={g:i for i,g in enumerate(BG)};mapping=json.loads((R/'references/2026-10-23_STPath_PathPrism/STPath/code/utils_data/symbol2ensembl.json').read_text())
orig_ix=[PROGRAMS.index(p) for p in ORIGINAL];checks=[]
# ---- ST ----
manifest=pd.read_csv(R/'results/t4/input_manifest.csv');sd=E/'st_targets';sd.mkdir(exist_ok=True)
for row in manifest.itertuples():
 dest=sd/f'{row.sample}.npz'
 if not dest.exists():
  a=ad.read_h5ad(row.rna);names=a.var_names.astype(str).tolist();raw=a.var.gene_ids.astype(str).tolist() if 'gene_ids' in a.var else names
  ids=[v.split('.')[0] if v.startswith('ENSG') else mapping.get(n) for n,v in zip(names,raw)];freq=collections.Counter(ids)
  index={g:i for i,g in enumerate(ids) if g in bgindex and freq[g]==1};assert set(index)==BGSET
  X=sparse.csr_matrix(a.X);ix=[index[g] for g in BG];scores=np.empty((a.n_obs,len(PROGRAMS)))
  for start in range(0,a.n_obs,32):
   block=X[start:start+32,ix].toarray();scores[start:start+len(block)]=ucell_from_ranks(capped_rank(block),bgindex,members,PROGRAMS)
  assert np.isfinite(scores).all()
  np.savez_compressed(dest,scores=scores,spot_ids=a.obs_names.astype(str).to_numpy(dtype=str),programs=np.asarray(PROGRAMS))
  time.sleep(0.5)
 with np.load(dest) as d,np.load(row.targets,allow_pickle=True) as t4:
  assert np.array_equal(d['spot_ids'],t4['spot_ids'].astype(str))
  diff=float(np.abs(d['scores'][:,orig_ix]-t4['scores']).max());checks.append(dict(kind='ST',sample=row.sample,spots=len(d['spot_ids']),max_abs_diff_vs_T4=diff))
 assert diff<1e-9,(row.sample,diff)
 print('ST',row.sample,diff,flush=True)
# ---- bulk（同 t5_st_bulk.py） ----
C5=json.loads((R/'config/t5.json').read_text());scores=[]
for cohort,name in [('TCGA-BRCA','BRCA_HiSeqV2.gz'),('TCGA-KIRC','KIRC_HiSeqV2.gz')]:
 t=pd.read_csv(Path(C5['reference_root'])/name,sep='\t',index_col=0);ids=[mapping.get(str(g)) for g in t.index];freq=pd.Series(ids).value_counts()
 index={g:i for i,g in enumerate(ids) if g in BGSET and freq[g]==1};assert set(index)==BGSET
 cols=[c for c in t.columns if c.startswith('TCGA-') and c.split('-')[3][:2]=='01'];v=t.iloc[[index[g] for g in BG]][cols].to_numpy(dtype='float64').T
 sc=ucell_from_ranks(capped_rank(v),bgindex,members,PROGRAMS)
 for n,s in zip(cols,sc):scores.append({'cohort':cohort,'patient_id':n[:12],'RNA_sample':n,**dict(zip(PROGRAMS,s))})
 print('BULK',cohort,len(cols),flush=True)
files=pd.read_csv(R/'docs/worklog/2026-10-26_T3/download_manifest.csv');sample=pd.read_csv(R/'results/t3/CPTAC_RNA_sample_manifest.csv')
sample=sample.groupby(['file_id','file','cohort','patient'],as_index=False).agg(sample=('sample',lambda x:';'.join(sorted(set(x)))));sample=sample.merge(files[['version','destination_file']],left_on='file_id',right_on='version',validate='one_to_one')
for row in sample.itertuples():
 t=pd.read_csv(row.destination_file,sep='\t',comment='#');t['id']=t.gene_id.str.split('.').str[0];freq=t.id.value_counts()
 lookup={g:i for i,g in enumerate(t.id) if g in BGSET and freq[g]==1};assert set(lookup)==BGSET
 v=t.iloc[[lookup[g] for g in BG]][C5['CPTAC_bulk_column']].to_numpy(dtype=float);sc=ucell_from_ranks(capped_rank(v[None,:]),bgindex,members,PROGRAMS)[0]
 scores.append({'cohort':row.cohort,'patient_id':row.patient,'RNA_sample':row.sample,**dict(zip(PROGRAMS,sc))})
bulk=pd.DataFrame(scores).groupby(['cohort','patient_id'],as_index=False)[PROGRAMS].mean();bulk.to_csv(O/'bulk_UCell_by_patient_58.csv',index=False)
t5=pd.read_csv(R/'results/t5/bulk_UCell_by_patient.csv').merge(bulk,on=['cohort','patient_id'],validate='one_to_one')
assert len(t5)==len(bulk)
for s,p in zip(['E2F','IFNg','Hypoxia','EMT'],ORIGINAL):
 d=float(np.abs(t5[s+'__mean']-t5[p]).max());checks.append(dict(kind='bulk',sample=p,spots=len(t5),max_abs_diff_vs_T4=d));assert d<1e-9,(p,d)
pd.DataFrame(checks).to_csv(O/'selfcheck_targets_vs_T4_T5.csv',index=False);print('TARGETS COMPLETE',len(bulk),flush=True)
