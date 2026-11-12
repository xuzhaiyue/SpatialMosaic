"""Nested validation chooses weight/model; all held-out curves remain descriptive."""
from t7_common import *
from t6_fit import moments,stload,stscore,predict
from scipy.linalg import eigh
from sklearn.model_selection import KFold
import joblib,time,h5py
MD=E/'models';MD.mkdir(exist_ok=True)
def fit_weight(a,b,lam,alpha=None,system=None):
 n=a['n']+b['n']
 if system is None:
  mx=lam*a['mx']+(1-lam)*b['mx'];delta=a['mx']-b['mx'];g=lam*a['xx']+(1-lam)*b['xx']+lam*(1-lam)*np.outer(delta,delta);sx=np.sqrt(np.maximum(np.diag(g),0));sx[sx==0]=1;g/=sx[:,None]*sx[None,:];h=(lam*a['xy']/a['sy']+(1-lam)*b['xy']/b['sy'])/sx[:,None];ev,v=eigh(g,check_finite=False);system=(mx,sx,np.maximum(ev,0),v,v.T@h)
 if alpha is None:return system
 mx,sx,ev,v,h=system;coef=(v@(h/(ev[:,None]+np.asarray(alpha)/n))).T/sx
 return dict(coef=coef,intercept=-coef@mx,ST_mean=a['my'],ST_scale=a['sy'],bulk_mean=b['my'],bulk_scale=b['sy'],alpha=np.broadcast_to(alpha,(4,)).tolist(),lambda_ST=lam,n_ST=a['n'],n_bulk=b['n'])
def raw_model(model,source):return model['coef']*model[source+'_scale'][:,None],model['intercept']*model[source+'_scale']+model[source+'_mean']
def baseline(key):
 d=joblib.load(C['baseline_T4_models'][key]['path']);coef=d['model'].coef_/d['scaler'].scale_;inter=d['model'].intercept_-coef@d['scaler'].mean_;return dict(raw_coef=coef,raw_intercept=inter,alpha=d['alpha'])
def train_only_ST(a,alpha):
 b={**a};m=fit_weight(a,b,1,alpha=np.asarray(alpha)*2);co,inter=raw_model(m,'ST');return dict(raw_coef=co,raw_intercept=inter)
def candidate_predict(m,x,source,slices=None):
 kind=m['candidate']
 if kind=='lambda_1':return x@m['raw_coef'].T+m['raw_intercept']
 if kind=='decomposed':
  if source=='bulk':return predict(m,x,'bulk')
  out=np.empty((len(x),4))
  for _,_,l,h in slices:
   mx=x[l:h].mean(0);out[l:h]=(x[l:h]-mx)@m['raw_coef'].T+predict(m,mx[None,:],'bulk')
  return out
 return predict(m,x,source)
if __name__=='__main__':
 if (O/'model_registry.json').exists():raise SystemExit('T7 fit already complete; use saved models and predictions')
 index=pd.read_csv(R/'results/t6/patient_mean_feature_index.csv');xa=np.load(E6/'patient_mean_features.npy');foldtable=pd.read_csv(R/'results/t6/TCGA_patient_folds.csv');bulk=pd.read_csv(R/C['bulk_targets']);manifest=pd.read_csv(R/C['T4_manifest']);registry=[];selectionrows=[];bulkrows=[];strows=[]
 for co,stco in [('TCGA-BRCA','He2020'),('TCGA-KIRC','Meylan_ccRCC')]:
  pts=index[index.cohort.eq(co)].merge(foldtable,on=['cohort','patient_id'],validate='one_to_one');paired=pts.merge(bulk,on=['cohort','patient_id'],validate='one_to_one');xb=xa[paired['row'].to_numpy()];yb=paired[MEANS].to_numpy();xt,yt,_=stload(manifest[manifest.cohort.eq(stco)&manifest.split.eq('train')]);xv,yv,sv=stload(manifest[manifest.cohort.eq(stco)&manifest.split.eq('val')]);a=moments(xt,yt);af=moments(np.r_[xt,xv],np.r_[yt,yv]);base=baseline(stco);baseval=train_only_ST(a,base['alpha']);STrho=stscore(yv,xv@baseval['raw_coef'].T+baseval['raw_intercept'],sv);del xt,yt
  for outer in list(range(1,6))+[0]:
   train=np.flatnonzero(paired.fold.to_numpy()!=outer) if outer else np.arange(len(paired));metrics=[];candidate_models={}
   # Every training fold fixes all candidate scores before held-out prediction.
   for inner,(ii,jj) in enumerate(KFold(3,shuffle=True,random_state=C['seed']).split(train),1):
    tr,va=train[ii],train[jj];b=moments(xb[tr],yb[tr]);oldbulk=xb[va]@baseval['raw_coef'].T+baseval['raw_intercept'];br=np.array([corr(yb[va,j],oldbulk[:,j]) for j in range(4)])
    for j,p in enumerate(P):metrics.append(dict(candidate='lambda_1',program=p,alpha=base['alpha'][j],inner_fold=inner,ST=STrho[j],bulk=br[j],score=(STrho[j]+br[j])/2))
    for lam in [0,.1,.25,.75,.9]:
     key='lambda_'+str(lam);sys=fit_weight(a,b,lam)
     for alpha in C['alpha']:
      m=fit_weight(a,b,lam,alpha,sys);sr=stscore(yv,predict(m,xv,'ST'),sv);bp=predict(m,xb[va],'bulk');br=np.array([corr(yb[va,j],bp[:,j]) for j in range(4)])
      for j,p in enumerate(P):metrics.append(dict(candidate=key,program=p,alpha=alpha,inner_fold=inner,ST=sr[j],bulk=br[j],score=(sr[j]+br[j])/2))
      if lam==0:
       for j,p in enumerate(P):metrics.append(dict(candidate='decomposed',program=p,alpha=alpha,inner_fold=inner,ST=STrho[j],bulk=br[j],score=(STrho[j]+br[j])/2))
    print('INNER',co,outer,inner,flush=True);time.sleep(.1)
   t6=pd.read_csv(R/f'results/t6/alpha_scores_{co}_fold{outer}.csv').rename(columns={'ST_validation_spearman':'ST','bulk_validation_spearman':'bulk','selection_score':'score'});t6['candidate']='lambda_0.5';df=pd.concat([pd.DataFrame(metrics),t6[['candidate','program','alpha','inner_fold','ST','bulk','score']]],ignore_index=True);df.to_csv(O/f'inner_validation_{co}_fold{outer}.csv',index=False);av=df.groupby(['candidate','program','alpha'],as_index=False)[['ST','bulk','score']].mean();summary=[];b=moments(xb[train],yb[train]);systems={}
   for key in C['candidates']:
    alphas=[];chosen=[]
    for p in P:
     z=av[av.candidate.eq(key)&av.program.eq(p)].sort_values(['score','alpha'],ascending=[False,False]).iloc[0];alphas.append(float(z.alpha));chosen.append(z)
    score=float(np.mean([z.score for z in chosen]));summary.append(dict(cohort=co,outer_fold=outer,candidate=key,score=score,ST_validation=float(np.mean([z.ST for z in chosen])),bulk_validation=float(np.mean([z.bulk for z in chosen]))))
    dest=MD/f'{co}_{key}_fold{outer}.joblib'
    if key=='lambda_1':m={**base,'candidate':key}
    elif key=='lambda_0.5':m=joblib.load(E6/'models'/f'{co}_fold{outer}.joblib');m['candidate']=key
    else:
     lam=0 if key=='decomposed' else float(key.split('_')[1]);sys=systems.setdefault(lam,fit_weight(af,b,lam));m=fit_weight(af,b,lam,alphas,sys);m['candidate']=key
     if key=='decomposed':m.update(raw_coef=base['raw_coef'],raw_intercept=base['raw_intercept'])
    m.update(cohort=co,ST_cohort=stco,outer_fold=outer,training_bulk_patients=paired.iloc[train].patient_id.tolist());joblib.dump(m,dest);candidate_models[key]=m;registry.append(dict(cohort=co,fold=outer,candidate=key,path=str(dest),sha256=sha(dest),alpha=m['alpha']))
   best=sorted(summary,key=lambda x:(-x['score'],C['selection_tie_preference'].index(x['candidate'])))[0];selectionrows.extend([{**v,'selected':v['candidate']==best['candidate']} for v in summary]);print('SELECTED',co,outer,best,flush=True)
   # Selection is fixed before accessing held-out targets or external cohorts.
   if outer:groups=[(co,pts[pts.fold.eq(outer)])]
   else:groups=[('CPTAC-BRCA' if co=='TCGA-BRCA' else 'CPTAC-CCRCC',index[index.cohort.eq('CPTAC-BRCA' if co=='TCGA-BRCA' else 'CPTAC-CCRCC')])]
   for evalco,te in groups:
    for key,m in candidate_models.items():
     pred=candidate_predict(m,xa[te['row'].to_numpy()],'bulk')
     for j,r in enumerate(te.itertuples()):bulkrows.append(dict(cohort=evalco,patient_id=r.patient_id,fold=outer,candidate=key,selected=key==best['candidate'],**dict(zip(MEANS,pred[j]))))
   if outer==0:
    ev=manifest[(manifest.cohort.eq(stco)&manifest.split.eq('test'))|(manifest.cohort.eq('HER2ST') if co=='TCGA-BRCA' else False)];xx,yy,ss=stload(ev)
    for key,m in candidate_models.items():
     pred=candidate_predict(m,xx,'ST',ss)
     for sample,pid,l,h in ss:
      ec=ev[ev['sample'].eq(sample)].iloc[0].cohort
      for j,p in enumerate(P):strows.append(dict(cohort=ec,patient_key=pid,sample=sample,program=p,candidate=key,selected=key==best['candidate'],rho=corr(yy[l:h,j],pred[l:h,j]),spots=h-l))
    del xx,yy
 (O/'model_registry.json').write_text(json.dumps(registry,indent=2));pd.DataFrame(selectionrows).to_csv(O/'candidate_validation_selection.csv',index=False);pd.DataFrame(bulkrows).to_csv(O/'bulk_predictions_all_candidates.csv',index=False);df=pd.DataFrame(strows);df.to_csv(O/'ST_section_all_candidates.csv',index=False);df.groupby(['cohort','patient_key','program','candidate','selected'],as_index=False).agg(rho=('rho','mean'),sections=('sample','size'),spots=('spots','sum')).to_csv(O/'ST_patient_all_candidates.csv',index=False);status(phase='fit_complete',models=len(registry));print('FIT COMPLETE',flush=True)
