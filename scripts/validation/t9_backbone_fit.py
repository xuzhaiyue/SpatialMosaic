"""T9-2：同一 Ridge 流程比较骨干（只换特征）。逻辑逐行移植 t4_models_discovery.py：
train 上 StandardScaler + 逐程序独立 Ridge，验证集按 切片内Spearman→患者内切片等权→验证患者等权 选 alpha（并列取大），
train+val 重拟合；He2020 模型同时评估 He2020 测试与 HER2ST，Meylan 模型评估 Meylan 测试。
--backbone UNI2h 读 T4 原特征，用于自检（须与 results/t4/heldout_spearman_summary.csv 一致）。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import json,argparse
from pathlib import Path
import numpy as np,pandas as pd,h5py
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
R=Path(__file__).resolve().parents[2];O=R/'results/t9/backbones';E=Path('/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts/t9/backbones')
def corr(a,b):return float(spearmanr(a,b).statistic) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan

def feature_path(row,backbone):return Path(row.features) if backbone=='UNI2h' else E/backbone/'features'/(row.sample+'.h5')
def load(rows,backbone):
 xs=[];ys=[];frames=[]
 for row in rows.itertuples():
  with np.load(row.targets,allow_pickle=True) as target,h5py.File(feature_path(row,backbone)) as f:
   assert f['done'][:].all() and np.array_equal(f['spot_ids'].asstr()[:],target['spot_ids']) and np.array_equal(f['coordinates'][:],target['coordinates']);x=f['features'][:];y=target['scores'];assert np.isfinite(x).all() and np.isfinite(y).all();xs.append(x);ys.append(y);frames.append(pd.DataFrame({'cohort':row.cohort,'patient_key':row.patient_key,'patient':row.patient,'sample':row.sample,'spot_id':target['spot_ids'],'split':row.split}))
 return np.vstack(xs),np.vstack(ys),pd.concat(frames,ignore_index=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True,choices=['UNI2h','densenet121','conch']);ap.add_argument('--cohorts',nargs='*',default=['He2020','Meylan_ccRCC']);a=ap.parse_args()
 cfg=json.loads((R/'config/t4_frozen.json').read_text());programs=cfg['programs'];manifest=pd.read_csv(R/cfg['input_manifest']);out=O/a.backbone;out.mkdir(parents=True,exist_ok=True);selected=[];sections=[]
 for cohort in a.cohorts:
  x,y,meta=load(manifest[manifest.cohort.eq(cohort)&manifest.split.eq('train')],a.backbone);xv,yv,mv=load(manifest[manifest.cohort.eq(cohort)&manifest.split.eq('val')],a.backbone);sc=StandardScaler().fit(x);z=sc.transform(x).astype('float64');zv=sc.transform(xv).astype('float64');scores=[]
  for alpha in cfg['alpha']:
   pred=Ridge(alpha=alpha,solver='cholesky').fit(z,y).predict(zv);rows=[]
   for (key,sample),ix in mv.groupby(['patient_key','sample']).groups.items():
    for j,p in enumerate(programs):rows.append({'patient_key':key,'program':p,'spearman':corr(yv[ix,j],pred[ix,j])})
   means=pd.DataFrame(rows).groupby(['patient_key','program']).spearman.mean().reset_index().groupby('program').spearman.mean();scores.append([means[p] for p in programs])
  score=np.asarray(scores);chosen=[]
  for j,p in enumerate(programs):
   v=score[:,j];best=np.nanmax(v) if np.isfinite(v).any() else np.nan;ix=np.flatnonzero(np.isclose(v,best,atol=1e-12,rtol=0)) if np.isfinite(best) else np.array([len(cfg['alpha'])-1]);chosen.append(cfg['alpha'][ix[-1]]);selected.append({'backbone':a.backbone,'cohort':cohort,'program':p,'alpha':cfg['alpha'][ix[-1]],'validation_patient_mean_spearman':v[ix[-1]]})
  xall=np.vstack([x,xv]);yall=np.vstack([y,yv]);sc=StandardScaler().fit(xall);model=Ridge(alpha=np.asarray(chosen),solver='cholesky').fit(sc.transform(xall).astype('float64'),yall)
  # 评估：只用测试 / 外部切片，各预测一次
  evals=manifest[manifest.split.isin(['test','external'])&(manifest.cohort.isin(['He2020','HER2ST']) if cohort=='He2020' else manifest.cohort.eq(cohort))]
  for row in evals.itertuples():
   xt,yt,_=load(manifest[manifest['sample'].eq(row.sample)],a.backbone);pred=model.predict(sc.transform(xt).astype('float64'));assert np.isfinite(pred).all()
   evaluation='HER2ST_external' if row.cohort=='HER2ST' else row.cohort+'_test'
   for j,p in enumerate(programs):sections.append({'backbone':a.backbone,'evaluation':evaluation,'cohort':row.cohort,'patient_key':row.patient_key,'patient':row.patient,'sample':row.sample,'program':p,'spots':row.spots,'spearman':corr(yt[:,j],pred[:,j])})
  print('FIT+EVAL',a.backbone,cohort,dict(zip(programs,chosen)),flush=True)
 sections=pd.DataFrame(sections);patients=sections.groupby(['backbone','evaluation','cohort','patient_key','patient','program'],as_index=False).agg(spearman=('spearman','mean'),sections=('sample','size'),spots=('spots','sum'))
 summary=patients.groupby(['backbone','evaluation','cohort','program'],as_index=False).agg(patients=('patient_key','size'),patients_finite=('spearman','count'),median_spearman=('spearman','median'),mean_spearman=('spearman','mean'))
 pd.DataFrame(selected).to_csv(out/'selected_alpha.csv',index=False);sections.to_csv(out/'heldout_section_spearman.csv',index=False);patients.to_csv(out/'heldout_patient_spearman.csv',index=False);summary.to_csv(out/'heldout_spearman_summary.csv',index=False);print(summary.to_string(),flush=True)
if __name__=='__main__':main()
