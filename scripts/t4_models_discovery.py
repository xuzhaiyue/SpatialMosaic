"""UNI2-h Ridge only: validation selection, then one prediction pass for discovery."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
from pathlib import Path
import json,hashlib,datetime
import numpy as np,pandas as pd,h5py,joblib
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from t4_display_provenance import allowed_hashes
R=Path(__file__).resolve().parents[1];W=R/'docs/worklog/2026-10-26_T4';O=R/'results/t4'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corr(a,b):return float(spearmanr(a,b).statistic) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan

def load(rows):
 xs=[];ys=[];frames=[]
 for row in rows.itertuples():
  # These target archives were generated locally by this task; legacy spot IDs use object dtype.
  with np.load(row.targets,allow_pickle=True) as target,h5py.File(row.features) as f:
   assert f['done'][:].all() and np.array_equal(f['spot_ids'].asstr()[:],target['spot_ids']) and np.array_equal(f['coordinates'][:],target['coordinates']);x=f['features'][:];y=target['scores'];assert np.isfinite(x).all() and np.isfinite(y).all();xs.append(x);ys.append(y);xy=target['coordinates'];frames.append(pd.DataFrame({'cohort':row.cohort,'patient_key':row.patient_key,'patient':row.patient,'sample':row.sample,'spot_id':target['spot_ids'],'split':row.split,'x':xy[:,0],'y':xy[:,1]}))
 return np.vstack(xs),np.vstack(ys),pd.concat(frames,ignore_index=True)

def main():
 cfg=json.loads((R/'config/t4_frozen.json').read_text());assert cfg['model_inputs']==['UNI2h'];allowed_hashes();assert not (W/'evaluation_started.json').exists()
 manifest=pd.read_csv(R/cfg['input_manifest']);modeldir=Path(cfg['large_output'])/'models';modeldir.mkdir(parents=True,exist_ok=True);models={};grid=[];selected=[];limits={}
 for cohort in ['He2020','Meylan_ccRCC']:
  x,y,meta=load(manifest[manifest.cohort.eq(cohort)&manifest.split.eq('train')]);xv,yv,mv=load(manifest[manifest.cohort.eq(cohort)&manifest.split.eq('val')]);sc=StandardScaler().fit(x);z=sc.transform(x).astype('float64');zv=sc.transform(xv).astype('float64');scores=[];limits[cohort]={p:[float(y[:,j].min()),float(y[:,j].max())] for j,p in enumerate(cfg['programs'])}
  for alpha in cfg['alpha']:
   model=Ridge(alpha=alpha,solver='cholesky').fit(z,y);pred=model.predict(zv);sections=[]
   for (key,sample),ix in mv.groupby(['patient_key','sample']).groups.items():
    for j,p in enumerate(cfg['programs']):sections.append({'patient_key':key,'sample':sample,'program':p,'spearman':corr(yv[ix,j],pred[ix,j])})
   section=pd.DataFrame(sections);patient=section.groupby(['patient_key','program']).spearman.mean().reset_index();means=patient.groupby('program').spearman.mean();scores.append([means[p] for p in cfg['programs']])
   for row in patient.itertuples():grid.append({'cohort':cohort,'program':row.program,'patient_key':row.patient_key,'alpha':alpha,'spearman':row.spearman})
   print('VALIDATION',cohort,alpha,flush=True)
  score=np.asarray(scores);chosen=[]
  for j,p in enumerate(cfg['programs']):
   v=score[:,j];best=np.nanmax(v) if np.isfinite(v).any() else np.nan;ix=np.flatnonzero(np.isclose(v,best,atol=1e-12,rtol=0)) if np.isfinite(best) else np.array([len(cfg['alpha'])-1]);a=cfg['alpha'][ix[-1]];chosen.append(a);selected.append({'cohort':cohort,'program':p,'alpha':a,'validation_patient_mean_spearman':v[ix[-1]],'reason':'validation_patient_mean' if np.isfinite(best) else 'all_NA_largest_alpha'})
  xall=np.vstack([x,xv]);yall=np.vstack([y,yv]);sc=StandardScaler().fit(xall);model=Ridge(alpha=np.asarray(chosen),solver='cholesky').fit(sc.transform(xall).astype('float64'),yall);path=modeldir/(cohort+'_UNI2h_discovery.joblib');joblib.dump({'scaler':sc,'model':model,'programs':cfg['programs'],'alpha':chosen},path);models[cohort]=path
 pd.DataFrame(grid).to_csv(O/'validation_patient_grid.csv',index=False);pd.DataFrame(selected).to_csv(O/'selected_alpha.csv',index=False);(O/'training_map_color_limits.json').write_text(json.dumps(limits,indent=2))
 with (W/'evaluation_started.json').open('x') as f:json.dump({'date':datetime.datetime.now().astimezone().isoformat(),'mode':cfg['mode'],'plan_sha256':sha(R/'PLAN.md'),'config_sha256':sha(R/'config/t4_frozen.json'),'models':{k:{'path':str(p),'sha256':sha(p)} for k,p in models.items()},'evaluation_count':1},f,indent=2)
 bundles={k:joblib.load(p) for k,p in models.items()};predictions=[];section_metrics=[];predictiondir=Path(cfg['large_output'])/'predictions';predictiondir.mkdir(parents=True,exist_ok=True)
 for row in manifest.itertuples():
  x,y,metadata=load(manifest[manifest['sample'].eq(row.sample)]);cohort='He2020' if row.cohort in ['He2020','HER2ST'] else 'Meylan_ccRCC';bundle=bundles[cohort];pred=bundle['model'].predict(bundle['scaler'].transform(x).astype('float64'));assert np.isfinite(pred).all();frame=metadata.copy()
  for j,p in enumerate(cfg['programs']):frame[p+'__observed']=y[:,j];frame[p+'__predicted']=pred[:,j]
  frame['prediction_role']='heldout' if row.split in ['test','external'] else 'fitted_display';predictions.append(frame);np.savez_compressed(predictiondir/(row.sample+'.npz'),observed=y,predicted=pred,coordinates=metadata[['x','y']].to_numpy(),spot_ids=metadata.spot_id.to_numpy(dtype=str),programs=np.asarray(cfg['programs']),prediction_role=np.asarray(frame.prediction_role.iloc[0]))
  if row.split in ['test','external']:
   evaluation='HER2ST_external' if row.cohort=='HER2ST' else row.cohort+'_test'
   for j,p in enumerate(cfg['programs']):section_metrics.append({'evaluation':evaluation,'cohort':row.cohort,'patient_key':row.patient_key,'patient':row.patient,'sample':row.sample,'program':p,'spots':row.spots,'spearman':corr(y[:,j],pred[:,j])})
  print('PREDICTED ONCE',row.sample,row.split,flush=True)
 pd.concat(predictions,ignore_index=True).to_csv(O/'all_spot_predictions.csv.gz',index=False,compression='gzip');sections=pd.DataFrame(section_metrics);sections.to_csv(O/'heldout_section_spearman.csv',index=False);patients=sections.groupby(['evaluation','cohort','patient_key','patient','program'],as_index=False).agg(spearman=('spearman','mean'),sections=('sample','size'),spots=('spots','sum'));patients.to_csv(O/'heldout_patient_spearman.csv',index=False);summary=patients.groupby(['evaluation','cohort','program'],as_index=False).agg(patients=('patient_key','size'),patients_finite=('spearman','count'),median_spearman=('spearman','median'),mean_spearman=('spearman','mean'));summary.to_csv(O/'heldout_spearman_summary.csv',index=False)
 (W/'evaluation_complete.json').write_text(json.dumps({'complete':True,'mode':cfg['mode'],'evaluation_count':1,'prediction_sections':len(manifest),'prediction_spots':int(manifest.spots.sum()),'heldout_patient_program_rows':len(patients)},indent=2));print('DISCOVERY PREDICTION COMPLETE',flush=True)
if __name__=='__main__':main()
