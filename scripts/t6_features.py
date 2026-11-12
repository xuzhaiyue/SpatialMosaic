"""Unchanged spatial descriptors for OOF maps and fold-consistent clinical maps."""
from t6_common import *
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
 manifest=pd.read_csv(O/'wsi_prediction_manifest.csv');foldtable=pd.read_csv(O/'TCGA_patient_folds.csv');baseline=pd.read_csv(R/C['T5_prediction_manifest']).merge(foldtable,on=['cohort','patient_id'],how='left');baseline['own_fold']=baseline.fold.fillna(0).astype(int);saved={};allslides=[]
 for cohort,rows in manifest.groupby('cohort'):
  dest=O/f'main_spatial_{cohort}.csv'
  if not dest.exists():
   t=thresholds(rows);saved[cohort]=t;df=descriptors(rows,t);df.to_csv(dest,index=False);(O/f'thresholds_{cohort}.json').write_text(json.dumps(t,indent=2))
  allslides.append(pd.read_csv(dest));print('MAIN SPATIAL',cohort,flush=True)
 slides=pd.concat(allslides,ignore_index=True);slides.to_csv(O/'wsi_spatial_by_slide.csv',index=False);patient=patients(slides);meta=pd.read_csv(R/C['T5_patient_features']);meta=meta.drop(columns=[c for c in FEATURES+['n_slides','n_patches'] if c in meta]);patient=patient.merge(meta,on=['cohort','patient_id'],validate='one_to_one',how='left');patient.to_csv(O/'patient_spatial_clinical.csv',index=False)
 # Fold-specific maps share the same graph; compute it once per slide.
 threshold_sets={};pending=[]
 for source in ['T6_dual','T5_STonly']:
  use=manifest if source=='T6_dual' else baseline
  for cohort in ['TCGA-BRCA','TCGA-KIRC']:
   rows=use[use.cohort.eq(cohort)]
   for fold in range(1,6):
    dest=O/f'clinical_features_{source}_{cohort}_fold{fold}.csv'
    if dest.exists():continue
    tp=O/f'clinical_thresholds_{source}_{cohort}_fold{fold}.json'
    if tp.exists():t=json.loads(tp.read_text())
    else:
     ds='fold_pred' if source=='T6_dual' else 'predicted';t=thresholds(rows,ds,fold,fold);tp.write_text(json.dumps(t,indent=2))
    threshold_sets[source,cohort,fold]=t;pending.append((source,cohort,fold));print('CLINICAL THRESHOLDS',source,cohort,fold,flush=True)
 accum={key:[] for key in pending};baseline_paths={(r.cohort,r.slide):r.path for r in baseline.itertuples()}
 for i,row in enumerate(manifest[manifest.cohort.str.startswith('TCGA')].itertuples(),1):
  relevant=[key for key in pending if key[1]==row.cohort]
  if not relevant:continue
  with h5py.File(row.path) as f:values=f['fold_pred'][:];xy=f['coords_top_left'][:];step=int(f.attrs['grid_step'])
  with h5py.File(baseline_paths[row.cohort,row.slide]) as f:old=f['predicted'][:];assert np.array_equal(xy,f['coords_top_left'][:])
  edges=graph(xy,step);A=adjacency(len(xy),edges)
  for key in relevant:
   source,cohort,fold=key;v=values[fold-1] if source=='T6_dual' else old;t=threshold_sets[key];bins=np.asarray(t['deciles']);means=v.mean(0);I=moran(v,edges);high=(v>np.asarray(t['q75'])).mean(0);co=coloc(v);r=dict(cohort=cohort,patient_id=row.patient_id,slide=row.slide,n_patches=len(v))
   for j,short in enumerate(S):r.update({short+'__mean':means[j],short+'__moran':I[j],short+'__high_fraction':high[j],short+'__entropy':local_entropy(np.searchsorted(bins[1:-1,j],v[:,j],side='right'),A)})
   for (a,b),val in zip(PAIRS,co):r[S[a]+'__'+S[b]+'__coloc']=val
   accum[key].append(r)
  if i%50==0:status('spatial_features_status.json',phase='clinical_fold_features',slides_done=i,slides_total=int(manifest.cohort.str.startswith('TCGA').sum()));print('CLINICAL SPATIAL',i,flush=True)
  time.sleep(.01)
 for (source,cohort,fold),records in accum.items():patients(pd.DataFrame(records)).to_csv(O/f'clinical_features_{source}_{cohort}_fold{fold}.csv',index=False)
 status('spatial_features_status.json',complete=True,slides=len(slides),patients=len(patient));print('FEATURES COMPLETE',flush=True)
