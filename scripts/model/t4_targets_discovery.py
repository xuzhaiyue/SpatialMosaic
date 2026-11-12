"""Fixed-background UCell targets for discovery maps; no sensitivity analyses."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,collections,hashlib
import pandas as pd,numpy as np,anndata as ad
from scipy import sparse
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[2]
def main():
 c=json.loads((R/'config/t4_frozen.json').read_text());assert (R/'PLAN.md').read_text().startswith('status: frozen_2026-10-26');bg=(R/c['background']).read_text().splitlines();bsha=hashlib.sha256((R/c['background']).read_bytes()).hexdigest();members=json.loads((R/c['members']).read_text());mapping=json.loads((R/'references/2026-10-23_STPath_PathPrism/STPath/code/utils_data/symbol2ensembl.json').read_text());manifest=pd.read_csv(R/c['input_manifest']);bgindex={g:i for i,g in enumerate(bg)};union=sorted({g for p in c['programs'] for g in members[p]});uix={g:i for i,g in enumerate(union)};groups={p:[uix[g] for g in members[p]] for p in c['programs']};completed=0
 for row in manifest.itertuples():
  dest=Path(row.targets)
  if dest.exists():
   with np.load(dest,allow_pickle=False) as d:assert d['background_sha256'].item()==bsha and d['scores'].shape==(row.spots,4)
   completed+=row.spots;continue
  a=ad.read_h5ad(row.rna);names=a.var_names.astype(str).tolist();raw=a.var.gene_ids.astype(str).tolist() if 'gene_ids' in a.var else names;ids=[v.split('.')[0] if v.startswith('ENSG') else mapping.get(n) for n,v in zip(names,raw)];freq=collections.Counter(ids);index={g:i for i,g in enumerate(ids) if g in bgindex and freq[g]==1};assert set(index)==set(bg);X=sparse.csr_matrix(a.X);ix=[index[g] for g in bg];scores=np.empty((a.n_obs,4),dtype='float64')
  for start in range(0,a.n_obs,32):
   block=X[start:start+32,ix].toarray();rank=len(bg)+1-rankdata(block,axis=1,method='average');cap=np.minimum(rank[:,[bgindex[g] for g in union]],c['max_rank'])
   for j,p in enumerate(c['programs']):
    k=len(groups[p]);scores[start:start+len(block),j]=1-(cap[:,groups[p]].sum(1,dtype='float64')-k*(k+1)/2)/(c['max_rank']*k-k*(k+1)/2)
  assert np.isfinite(scores).all();dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial.npz');np.savez_compressed(tmp,scores=scores,spot_ids=a.obs_names.astype(str).to_numpy(dtype=str),coordinates=np.asarray(a.obsm['spatial']),programs=np.asarray(c['programs']),background_sha256=np.asarray(bsha));tmp.replace(dest);completed+=a.n_obs;print('TARGET',row.sample,completed,flush=True)
 (R/'results/t4/targets_complete.json').write_text(json.dumps({'complete':True,'spots':completed,'sections':len(manifest),'mode':c['mode'],'background_sha256':bsha}));print('TARGETS COMPLETE',flush=True)
if __name__=='__main__':main()
