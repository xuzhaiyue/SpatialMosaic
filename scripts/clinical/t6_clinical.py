"""Patient-fold clinical comparison, keeping expression test bulk outside each fitted map."""
from t6_common import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.duration.hazard_regression import PHReg
import joblib

def cindex(time,event,risk):
 score=0.;pairs=0
 for i in np.flatnonzero(event):
  mask=(time>time[i])|((time==time[i])&(event==0));d=risk[i]-risk[mask];score+=np.sum(d>0)+.5*np.sum(d==0);pairs+=len(d)
 return float(score/pairs) if pairs else np.nan
if __name__=='__main__':
 meta=pd.read_csv(O/'patient_spatial_clinical.csv').drop(columns=FEATURES+['n_slides','n_patches']);folds=pd.read_csv(O/'TCGA_patient_folds.csv');metrics=[];predictions=[];counts=[];models=E/'clinical_models';models.mkdir(exist_ok=True)
 for source in ['T5_STonly','T6_dual']:
  for cohort in ['TCGA-BRCA','TCGA-KIRC']:
   for fold in range(1,6):
    data=pd.read_csv(O/f'clinical_features_{source}_{cohort}_fold{fold}.csv').merge(meta,on=['cohort','patient_id'],validate='one_to_one').merge(folds,on=['cohort','patient_id'],validate='one_to_one');data['stage_high']=data.stage_I_IV.isin(['III','IV']).astype(int)
    for task in (['PFI','PAM50'] if cohort=='TCGA-BRCA' else ['PFI']):
     if task=='PFI':d=data[data.PFI.isin([0,1])&(data.PFI_time>0)&data.age.notna()&data.stage_I_IV.isin(['I','II','III','IV'])].reset_index(drop=True)
     else:d=data[data.PAM50Call_RNAseq.isin(C['main_PAM50'])].reset_index(drop=True)
     tr=np.flatnonzero(d.fold.to_numpy()!=fold);te=np.flatnonzero(d.fold.to_numpy()==fold);counts.append(dict(source=source,task=task,cohort=cohort,fold=fold,train_n=len(tr),test_n=len(te),train_events=d.iloc[tr].PFI.sum() if task=='PFI' else np.nan,test_events=d.iloc[te].PFI.sum() if task=='PFI' else np.nan))
     for name,features in [('mean',MEANS),('mean_plus_spatial',FEATURES)]:
      if task=='PFI':features=['age','stage_high']+features
      imp=SimpleImputer(strategy='median',keep_empty_features=True);sc=StandardScaler();x=sc.fit_transform(imp.fit_transform(d.loc[tr,features]));xt=sc.transform(imp.transform(d.loc[te,features]));modeldict=dict(imputer=imp,scaler=sc,features=features,source=source,fold=fold,expression_model_fold=fold)
      if task=='PFI':
       fit=PHReg(d.loc[tr,'PFI_time'].to_numpy(),x,status=d.loc[tr,'PFI'].to_numpy().astype(int),ties='breslow').fit_regularized(alpha=C['cox_ridge_alpha'],L1_wt=0,maxiter=1000,cnvrg_tol=1e-7);risk=xt@fit.params;val=cindex(d.loc[te,'PFI_time'].to_numpy(),d.loc[te,'PFI'].to_numpy().astype(int),risk);modeldict['coef']=fit.params
       for i,v in zip(te,risk):predictions.append(dict(source=source,task=task,cohort=cohort,fold=fold,model=name,patient_id=d.loc[i,'patient_id'],risk=v,time=d.loc[i,'PFI_time'],event=d.loc[i,'PFI']))
      else:
       fit=LogisticRegression(C=C['logistic_C'],solver='lbfgs',max_iter=5000,random_state=C['seed']).fit(x,d.loc[tr,'PAM50Call_RNAseq']);pr=fit.predict_proba(xt);val=roc_auc_score(d.loc[te,'PAM50Call_RNAseq'],pr,labels=fit.classes_,multi_class='ovr',average='macro');modeldict['model']=fit
       for j,i in enumerate(te):predictions.append(dict(source=source,task=task,cohort=cohort,fold=fold,model=name,patient_id=d.loc[i,'patient_id'],label=d.loc[i,'PAM50Call_RNAseq'],**{'prob_'+k:pr[j,kix] for kix,k in enumerate(fit.classes_)}))
      joblib.dump(modeldict,models/f'{source}_{cohort}_{task}_{name}_fold{fold}.joblib');metrics.append(dict(source=source,cohort=cohort,task=task,fold=fold,model=name,value=float(val),patients=len(te),metric='C_index' if task=='PFI' else 'macro_AUROC'));print('CLINICAL',source,cohort,fold,task,name,val,flush=True)
 df=pd.DataFrame(metrics);df.to_csv(O/'clinical_CV_fold_metrics.csv',index=False);df.groupby(['source','cohort','task','model','metric'],as_index=False).agg(mean_CV=('value','mean'),SD_folds=('value','std'),folds=('fold','size')).to_csv(O/'clinical_CV_summary.csv',index=False);pd.DataFrame(predictions).to_csv(O/'clinical_OOF_predictions.csv',index=False);pd.DataFrame(counts).to_csv(O/'clinical_patient_fold_counts.csv',index=False);status('clinical_status.json',complete=True);print('CLINICAL COMPLETE',flush=True)
