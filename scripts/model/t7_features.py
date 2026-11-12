"""Unchanged spatial descriptors for OOF maps and fold-consistent clinical maps."""
from t7_common import *
import h5py,time

def thresholds(rows,dataset='predicted',fold=None,exclude_fold=None):
 chosen=rows[rows.own_fold.ne(exclude_fold)] if exclude_fold is not None else rows
 path=E/'quantile_work.npy';pool=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=(int(chosen.patches.sum()),4));offset=0
 for row in chosen.itertuples():
  with h5py.File(row.path) as f:v=f[dataset][fold-1] if dataset=='fold_pred' else f[dataset][:]
  pool[offset:offset+len(v)]=v;offset+=len(v)
 pool.flush();q=np.quantile(pool,np.linspace(0,1,11),axis=0);hi=np.quantile(pool,.75,axis=0);del pool;path.unlink();return dict(deciles=q.tolist(),q75=hi.tolist(),training_patches=offset)

def descriptors(rows,t,dataset='predicted',fold=None):
 out=[];bins=np.asarray(t['deciles']);hi=np.asarray(t['q75'])
 for row in rows.itertuples():
  with h5py.File(row.path) as f:v=f[dataset][fold-1] if dataset=='fold_pred' else f[dataset][:];xy=f['coords_top_left'][:];step=int(f.attrs['grid_step'])
  edges=graph(xy,step);A=adjacency(len(v),edges);means=v.mean(0);I=moran(v,edges);high=(v>hi).mean(0);co=coloc(v);r=dict(cohort=row.cohort,patient_id=row.patient_id,slide=row.slide,n_patches=len(v),prediction_path=row.path,grid_step=step,edges=len(edges))
  for j,s in enumerate(S):r.update({s+'__mean':means[j],s+'__moran':I[j],s+'__high_fraction':high[j],s+'__entropy':local_entropy(np.searchsorted(bins[1:-1,j],v[:,j],side='right'),A)})
  for (a,b),val in zip(PAIRS,co):r[S[a]+'__'+S[b]+'__coloc']=val
  out.append(r);time.sleep(.005)
 return pd.DataFrame(out)

def patients(slides):
 d=slides.groupby(['cohort','patient_id'],as_index=False)[FEATURES].mean();counts=slides.groupby(['cohort','patient_id'],as_index=False).agg(n_slides=('slide','size'),n_patches=('n_patches','sum'));return d.merge(counts,on=['cohort','patient_id'],validate='one_to_one')
if __name__=='__main__':
 manifest=pd.read_csv(O/'wsi_prediction_manifest.csv');saved={};allslides=[]
 for cohort,rows in manifest.groupby('cohort'):
  dest=O/f'main_spatial_{cohort}.csv'
  if not dest.exists():
   t=thresholds(rows);saved[cohort]=t;df=descriptors(rows,t);df.to_csv(dest,index=False);(O/f'thresholds_{cohort}.json').write_text(json.dumps(t,indent=2))
  allslides.append(pd.read_csv(dest));print('MAIN SPATIAL',cohort,flush=True)
 slides=pd.concat(allslides,ignore_index=True);slides.to_csv(O/'wsi_spatial_by_slide.csv',index=False);patient=patients(slides);meta=pd.read_csv(R/C['T5_patient_features']);meta=meta.drop(columns=[c for c in FEATURES+['n_slides','n_patches'] if c in meta]);patient=patient.merge(meta,on=['cohort','patient_id'],validate='one_to_one',how='left');patient.to_csv(O/'patient_spatial_clinical.csv',index=False)

 status(phase="features_complete",patients=len(patient));(W/"features_complete.json").write_text(json.dumps({"complete":True}));print("FEATURES COMPLETE",flush=True)
