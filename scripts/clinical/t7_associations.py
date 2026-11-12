"""Discovery associations: one spatial feature at a time, with fixed covariates."""
from t7_common import *
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.discrete.discrete_model import Logit
from statsmodels.stats.multitest import multipletests
import warnings
if __name__=='__main__':
 patient=pd.read_csv(O/'patient_prototypes.csv');rows=[]
 for co in ['TCGA-BRCA','TCGA-KIRC']:
  base=patient[patient.cohort.eq(co)&patient.PFI.isin([0,1])&(patient.PFI_time>0)&patient.age.notna()&patient.stage_I_IV.isin(['I','II','III','IV'])].copy();base['stage_high']=base.stage_I_IV.isin(['III','IV']).astype(int)
  for feature in SPACE:
   d=base.dropna(subset=MEANS+[feature]);cols=['age','stage_high']+MEANS+[feature];x=d[cols].to_numpy(dtype=float);sd=x.std(0);sd[sd==0]=1;sd[1]=1;x=(x-x.mean(0))/sd;rec=dict(cohort=co,feature=feature,n=len(d),events=int(d.PFI.sum()),feature_SD=float(sd[-1]))
   try:
    with warnings.catch_warnings(record=True) as warning:
     fit=PHReg(d.PFI_time.to_numpy(),x,status=d.PFI.to_numpy(),ties='breslow').fit(disp=False,maxiter=200)
    beta=float(fit.params[-1]);se=float(fit.bse[-1]);p=float(fit.pvalues[-1]);rec.update(beta=beta,SE=se,HR=np.exp(beta),CI_low=np.exp(beta-1.95996398454*se),CI_high=np.exp(beta+1.95996398454*se),p=p,warnings=';'.join(str(v.message) for v in warning))
   except Exception as e:rec.update(beta=np.nan,SE=np.nan,HR=np.nan,CI_low=np.nan,CI_high=np.nan,p=np.nan,error=str(e))
   rows.append(rec)
 df=pd.DataFrame(rows);df['q']=np.nan
 for co,g in df.groupby('cohort'):
  df.loc[g.index,'q']=multipletests(g.p.fillna(1),method='fdr_bh')[1];df.loc[g.index[g.p.isna()],'q']=np.nan
 df.to_csv(O/'Cox_single_feature_associations.csv',index=False);rows=[];base=patient[patient.cohort.eq('TCGA-BRCA')&patient.PAM50Call_RNAseq.isin(C['main_PAM50'])]
 for label in C['main_PAM50']:
  for feature in SPACE:
   d=base.dropna(subset=MEANS+[feature]);x=d[MEANS+[feature]].to_numpy(dtype=float);sd=x.std(0);sd[sd==0]=1;x=np.c_[np.ones(len(d)),(x-x.mean(0))/sd];y=d.PAM50Call_RNAseq.eq(label).to_numpy().astype(int);rec=dict(subtype=label,feature=feature,n=len(d),subtype_n=int(y.sum()),feature_SD=float(sd[-1]))
   try:
    with warnings.catch_warnings(record=True) as warning:fit=Logit(y,x).fit(disp=False,maxiter=200)
    assert fit.mle_retvals['converged'],'Logit did not converge';beta=float(fit.params[-1]);se=float(fit.bse[-1]);rec.update(beta=beta,SE=se,OR=np.exp(beta),CI_low=np.exp(beta-1.95996398454*se),CI_high=np.exp(beta+1.95996398454*se),p=float(fit.pvalues[-1]),warnings=';'.join(str(v.message) for v in warning))
   except Exception as e:rec.update(beta=np.nan,SE=np.nan,OR=np.nan,CI_low=np.nan,CI_high=np.nan,p=np.nan,error=str(e))
   rows.append(rec)
 df=pd.DataFrame(rows);df['q']=np.nan;df['q']=multipletests(df.p.fillna(1),method='fdr_bh')[1];df.loc[df.p.isna(),'q']=np.nan;df.to_csv(O/'PAM50_single_feature_associations.csv',index=False);(W/'associations_complete.json').write_text(json.dumps({'complete':True,'time':stamp()}));print('ASSOCIATIONS COMPLETE',flush=True)
