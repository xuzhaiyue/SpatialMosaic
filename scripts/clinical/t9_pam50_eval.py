"""T9-1 第三步：H&E 四程序全片均值 -> PAM50。TCGA 全部训练，CPTAC 一次性外部评估。
特征：T6/T7 全量主模型（TCGA-BRCA_fold0，λ=0.5）× 患者等切片平均特征，bulk 标度（线性恒等，等于全片 patch 均值的切片等权平均）。
分类器与 T6 'mean' 模型相同：中位数填补 + StandardScaler + LogisticRegression(C=1, lbfgs)，只在 TCGA 拟合。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,joblib,h5py
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,cohen_kappa_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parents[2];O=R/'results/t9/pam50';C=json.loads((R/'config/t6.json').read_text())
E=Path(C['output_root']);CLASSES=C['main_PAM50'];MEANS=['E2F__mean','IFNg__mean','Hypoxia__mean','EMT__mean']
model=joblib.load(E/'models/TCGA-BRCA_fold0.joblib');assert model['outer_fold']==0 and model['cohort']=='TCGA-BRCA'
index=pd.read_csv(R/'results/t6/patient_mean_feature_index.csv');xall=np.load(E/'patient_mean_features.npy')
pred=(xall@model['coef'].T+model['intercept'])*model['bulk_scale']+model['bulk_mean']
feat=pd.concat([index[['cohort','patient_id','n_slides']],pd.DataFrame(pred,columns=MEANS)],axis=1)
feat=feat[feat.cohort.isin(['TCGA-BRCA','CPTAC-BRCA'])]
# 自检 1：CPTAC 20 例，线性恒等 vs T7 全片 patch 预测（切片 patch 均值再切片等权平均）
check=[];rng=np.random.default_rng(20261028);T7=Path('/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts/t7/patch_predictions/CPTAC-BRCA')
slides=pd.read_csv(R/'results/t6/slide_mean_feature_index.csv');slides=slides[slides.cohort.eq('CPTAC-BRCA')]
for pid in rng.choice(sorted(slides.patient_id.unique()),20,replace=False):
 means=[]
 for s in slides[slides.patient_id.eq(pid)].slide:
  with h5py.File(T7/f'{s}.h5','r') as f:
   key=[k for k in f.keys() if f[k].ndim==2 and f[k].shape[1]==4]
   means.append(np.asarray(f[key[0]][:],dtype='float64').mean(0))
 check.append(dict(patient_id=pid,keys=';'.join(key),max_abs_diff=float(np.abs(np.mean(means,0)-feat.set_index('patient_id').loc[pid,MEANS].to_numpy(float)).max())))
check=pd.DataFrame(check);check.to_csv(O/'selfcheck_linear_identity_CPTAC20.csv',index=False);print('identity max diff',check.max_abs_diff.max())
# 标签
gen={c:pd.read_csv(O/f'{c}_PAM50_genefu_calls.csv')[['patient_id','PAM50_genefu']] for c in ['TCGA-BRCA','CPTAC-BRCA']}
off=pd.read_csv(R/'results/t2/patient_intersection.csv');off=off[off.cohort.eq('TCGA-BRCA')][['patient_id','PAM50Call_RNAseq']].drop_duplicates()
tc=gen['TCGA-BRCA'].merge(off,on='patient_id',how='inner');both=tc[tc.PAM50Call_RNAseq.isin(['Basal','Her2','LumA','LumB','Normal'])]
labs=['Basal','Her2','LumA','LumB','Normal'];cm=pd.DataFrame(confusion_matrix(both.PAM50Call_RNAseq,both.PAM50_genefu,labels=labs),index=['official_'+l for l in labs],columns=['genefu_'+l for l in labs])
cm.to_csv(O/'selfcheck_TCGA_genefu_vs_official_confusion.csv')
agree=dict(n=len(both),agreement=float((both.PAM50Call_RNAseq==both.PAM50_genefu).mean()),kappa=float(cohen_kappa_score(both.PAM50Call_RNAseq,both.PAM50_genefu)))
print('TCGA genefu vs official',agree);print(cm.to_string())
def design(d):return d[MEANS].to_numpy(float)
def fit(d,label):
 imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler();x=sc.fit_transform(imp.fit_transform(design(d)))
 clf=LogisticRegression(C=C['logistic_C'],solver='lbfgs',max_iter=5000,random_state=C['seed']).fit(x,d[label]);return imp,sc,clf
def evaluate(m,d,label):
 imp,sc,clf=m;p=clf.predict_proba(sc.transform(imp.transform(design(d))))
 keep=[k for k in clf.classes_ if (d[label]==k).sum()>=5];rows={}
 for k in clf.classes_:
  y=(d[label]==k).astype(int)
  rows['AUROC_'+k]=float(roc_auc_score(y,p[:,list(clf.classes_).index(k)])) if 0<y.sum()<len(y) else np.nan
 rows['macro_AUROC']=float(np.mean([rows['AUROC_'+k] for k in keep]));rows['classes_in_macro']=';'.join(keep);rows['n']=len(d)
 rows.update({'n_'+k:int((d[label]==k).sum()) for k in clf.classes_});return rows,p
results=[];preds=[]
for name,train_label_source in [('primary_genefu_labels','genefu'),('secondary_official_labels','official')]:
 tr=feat[feat.cohort.eq('TCGA-BRCA')].merge(gen['TCGA-BRCA'],on='patient_id').merge(off,on='patient_id',how='left')
 label='PAM50_genefu' if train_label_source=='genefu' else 'PAM50Call_RNAseq';tr=tr[tr[label].isin(CLASSES)].reset_index(drop=True)
 # TCGA 内部 5 折（同一特征、同一分类器）作参照
 cv=[]
 for a,b in StratifiedKFold(5,shuffle=True,random_state=C['seed']).split(tr,tr[label]):
  m=fit(tr.iloc[a],label);cv.append(evaluate(m,tr.iloc[b],label)[0]['macro_AUROC'])
 m=fit(tr,label);te=feat[feat.cohort.eq('CPTAC-BRCA')].merge(gen['CPTAC-BRCA'],on='patient_id',how='inner');te=te[te.PAM50_genefu.isin(CLASSES)].reset_index(drop=True)
 ext,p=evaluate(m,te,'PAM50_genefu')
 results.append(dict(analysis=name,train_n=len(tr),TCGA_CV_macro_AUROC_mean=float(np.mean(cv)),TCGA_CV_SD=float(np.std(cv,ddof=1)),**{'CPTAC_'+k:v for k,v in ext.items()}))
 for i,r in te.iterrows():preds.append(dict(analysis=name,patient_id=r.patient_id,label=r.PAM50_genefu,**{'prob_'+k:p[i,j] for j,k in enumerate(m[2].classes_)}))
 joblib.dump(dict(imputer=m[0],scaler=m[1],model=m[2],features=MEANS,expression_model='t6/models/TCGA-BRCA_fold0.joblib'),O/f'PAM50_{name}.joblib')
res=pd.DataFrame(results);res.to_csv(O/'PAM50_external_CPTAC.csv',index=False);pd.DataFrame(preds).to_csv(O/'PAM50_CPTAC_predictions.csv',index=False)
(O/'TCGA_label_agreement.json').write_text(json.dumps(agree,indent=2));print(res.T.to_string())
