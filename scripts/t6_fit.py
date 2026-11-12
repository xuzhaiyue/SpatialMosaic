"""Patient cross-fitted dual-source weighted Ridge, using exact sufficient statistics."""
from t6_common import *
import h5py,joblib,time
from sklearn.model_selection import KFold
from scipy.linalg import eigh
MD=E/'models';MD.mkdir(exist_ok=True)
def moments(x,y):
 mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;sy=y.std(0);sy[sy==0]=1
 return dict(n=len(x),mx=mx,my=my,sy=sy,xx=xc.T@xc/len(x),xy=xc.T@yc/len(x))
def system(a,b):
 n=a['n']+b['n'];mx=(a['mx']+b['mx'])/2;delta=a['mx']-b['mx'];g=(a['xx']+b['xx'])/2+np.outer(delta,delta)/4;sx=np.sqrt(np.maximum(np.diag(g),0));sx[sx==0]=1;g/=sx[:,None]*sx[None,:];h=(a['xy']/a['sy']+b['xy']/b['sy'])/2/sx[:,None];lam,v=eigh(g,check_finite=False);return n,mx,sx,np.maximum(lam,0),v,v.T@h

def fit(sys,a,b,alpha):
 n,mx,sx,lam,v,h=sys;coef=((v@(h/(lam[:,None]+np.asarray(alpha)/n))).T/sx);return dict(coef=coef,intercept=-coef@mx,ST_mean=a['my'],ST_scale=a['sy'],bulk_mean=b['my'],bulk_scale=b['sy'],alpha=np.broadcast_to(alpha,(4,)).tolist(),n_ST=a['n'],n_bulk=b['n'],source_total_weight=n/2)
def predict(m,x,source):return (x@m['coef'].T+m['intercept'])*m[source+'_scale']+m[source+'_mean']
def stload(rows):
 xs=[];ys=[];slices=[];start=0
 for r in rows.itertuples():
  with h5py.File(r.features) as f:x=f['features'][:].astype('float64');sid=f['spot_ids'].asstr()[:]
  with np.load(r.targets,allow_pickle=True) as f:y=f['scores'];assert np.array_equal(sid,f['spot_ids'])
  xs.append(x);ys.append(y);slices.append((r.sample,r.patient_key,start,start+len(x)));start+=len(x)
 return np.concatenate(xs),np.concatenate(ys),slices

def stscore(y,pred,slices):
 rows=[]
 for sample,pid,a,b in slices:
  for j in range(4):rows.append(dict(patient=pid,program=j,value=corr(y[a:b,j],pred[a:b,j])))
 return pd.DataFrame(rows).groupby(['patient','program']).value.mean().groupby('program').mean().reindex(range(4)).to_numpy()
if __name__=='__main__':
 index=pd.read_csv(O/'patient_mean_feature_index.csv');xall=np.load(E/'patient_mean_features.npy');bulk=pd.read_csv(R/C['bulk_targets']);manifest=pd.read_csv(R/C['T4_manifest']);foldrows=[];registry=[];selections=[];scores=[];bulkpred=[]
 for cohort,stcohort in [('TCGA-BRCA','He2020'),('TCGA-KIRC','Meylan_ccRCC')]:
  pts=index[index.cohort.eq(cohort)].sort_values('patient_id').copy();pts['fold']=0
  for fold,(_,te) in enumerate(KFold(5,shuffle=True,random_state=C['seed']).split(pts),1):pts.iloc[te,pts.columns.get_loc('fold')]=fold
  foldrows.extend(pts[['cohort','patient_id','fold']].to_dict('records'));paired=pts.merge(bulk,on=['cohort','patient_id'],validate='one_to_one');xb=xall[paired['row'].to_numpy()];yb=paired[MEANS].to_numpy();xt,yt,_=stload(manifest[manifest.cohort.eq(stcohort)&manifest.split.eq('train')]);xv,yv,sv=stload(manifest[manifest.cohort.eq(stcohort)&manifest.split.eq('val')]);a=moments(xt,yt);af=moments(np.concatenate([xt,xv]),np.concatenate([yt,yv]));del xt,yt
  for outer in list(range(1,6))+[0]:
   dest=MD/f'{cohort}_fold{outer}.joblib';train=np.flatnonzero(paired.fold.to_numpy()!=outer) if outer else np.arange(len(paired))
   if dest.exists():m=joblib.load(dest)
   else:
    metrics=[]
    for inner,(ii,jj) in enumerate(KFold(3,shuffle=True,random_state=C['seed']).split(train),1):
     tr,va=train[ii],train[jj];b=moments(xb[tr],yb[tr]);sys=system(a,b)
     for alpha in C['alpha']:
      m=fit(sys,a,b,alpha);st=stscore(yv,predict(m,xv,'ST'),sv);bp=predict(m,xb[va],'bulk');bs=np.array([corr(yb[va,j],bp[:,j]) for j in range(4)]);combined=(st+bs)/2
      for j in range(4):metrics.append(dict(cohort=cohort,outer_fold=outer,inner_fold=inner,alpha=alpha,program=P[j],ST_validation_spearman=st[j],bulk_validation_spearman=bs[j],selection_score=combined[j]))
     print('INNER',cohort,outer,inner,flush=True);time.sleep(.2)
    df=pd.DataFrame(metrics);df.to_csv(O/f'alpha_scores_{cohort}_fold{outer}.csv',index=False);selection=df.groupby(['program','alpha'],as_index=False).selection_score.mean();alphas=[]
    for p in P:alphas.append(float(selection[selection.program.eq(p)].sort_values(['selection_score','alpha'],ascending=[False,False]).iloc[0].alpha))
    b=moments(xb[train],yb[train]);m=fit(system(af,b),af,b,alphas);m.update(cohort=cohort,ST_cohort=stcohort,outer_fold=outer,training_bulk_patients=paired.iloc[train].patient_id.tolist(),ST_refit='train+val',seed=C['seed']);joblib.dump(m,dest)
   registry.append(dict(cohort=cohort,fold=outer,path=str(dest),sha256=sha(dest),alphas=m['alpha'],bulk_train_n=m['n_bulk']));print('FIT',cohort,outer,m['alpha'],flush=True)
   if outer:
    te=pts[pts.fold.eq(outer)];out=predict(m,xall[te['row'].to_numpy()],'bulk')
    for i,r in enumerate(te.itertuples()):bulkpred.append(dict(cohort=cohort,patient_id=r.patient_id,fold=outer,**dict(zip(MEANS,out[i]))))
   else:
    for ext in (['CPTAC-BRCA'] if cohort=='TCGA-BRCA' else ['CPTAC-CCRCC']):
     te=index[index.cohort.eq(ext)];out=predict(m,xall[te['row'].to_numpy()],'bulk')
     for i,r in enumerate(te.itertuples()):bulkpred.append(dict(cohort=ext,patient_id=r.patient_id,fold=0,**dict(zip(MEANS,out[i]))))
    evalrows=manifest[(manifest.cohort.eq(stcohort)&manifest.split.eq('test'))|(manifest.cohort.eq('HER2ST') if cohort=='TCGA-BRCA' else False)];xx,yy,ss=stload(evalrows);pr=predict(m,xx,'ST');records=[]
    for sample,pid,l,h in ss:
     row=evalrows[evalrows['sample'].eq(sample)].iloc[0];outdir=E/'ST_predictions';outdir.mkdir(exist_ok=True)
     with np.load(row.targets,allow_pickle=True) as source:np.savez_compressed(outdir/(sample+'.npz'),observed=yy[l:h],predicted=pr[l:h],coordinates=source['coordinates'],spot_ids=source['spot_ids'].astype(str),programs=np.asarray(P))
     for j,p in enumerate(P):records.append(dict(cohort=row.cohort,patient_key=pid,sample=sample,program=p,spearman=corr(yy[l:h,j],pr[l:h,j]),spots=h-l))
    sd=pd.DataFrame(records);scores.extend(sd.to_dict('records'));del xx,yy,pr
 pd.DataFrame(foldrows).to_csv(O/'TCGA_patient_folds.csv',index=False);(O/'model_registry.json').write_text(json.dumps(registry,indent=2));pd.DataFrame(bulkpred).to_csv(O/'bulk_prediction_by_mean_feature.csv',index=False);pd.DataFrame(scores).to_csv(O/'ST_section_spearman.csv',index=False);pd.DataFrame(scores).groupby(['cohort','patient_key','program'],as_index=False).agg(spearman=('spearman','mean'),sections=('sample','size'),spots=('spots','sum')).to_csv(O/'ST_patient_spearman.csv',index=False);status('fit_status.json',complete=True,models=len(registry));print('FIT COMPLETE',flush=True)
