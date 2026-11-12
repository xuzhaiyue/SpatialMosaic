"""T9-3 第二步：K 个程序的双尺度 Ridge。流程逐步照搬 t6_fit.py（同折、同内层 3 折、同 alpha 网格与并列规则、train+val 重拟合），
拟合式为 t7_fit.fit_weight 的 λ 加权形式（λ=0.5 时与 T6 system 完全相同），只把写死的 4 个程序改为 K 个。
用法：python t9_atlas_fit.py regress   -> 原 4 程序、λ=0.5，与 T6 结果逐项比对；
      python t9_atlas_fit.py atlas     -> 58 程序，λ 乳腺 0.5 / 肾 0.75（T7 验证选择）。"""
import sys,time,h5py,joblib
from t9_common import *
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
C6=json.loads((R/'config/t6.json').read_text());SEED=C6['seed'];ALPHAS=C6['alpha']
MODE=sys.argv[1];assert MODE in ('regress','atlas')
LAMBDA={'TCGA-BRCA':0.5,'TCGA-KIRC':0.5 if MODE=='regress' else 0.75}
def corr(a,b):return float(spearmanr(a,b).statistic) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan  # 同 t6_common.corr
def moments(x,y):
 mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my;sy=y.std(0);sy[sy==0]=1
 return dict(n=len(x),mx=mx,my=my,sy=sy,xx=xc.T@xc/len(x),xy=xc.T@yc/len(x))
def system(a,b,lam):
 mx=lam*a['mx']+(1-lam)*b['mx'];delta=a['mx']-b['mx'];g=lam*a['xx']+(1-lam)*b['xx']+lam*(1-lam)*np.outer(delta,delta);sx=np.sqrt(np.maximum(np.diag(g),0));sx[sx==0]=1;g/=sx[:,None]*sx[None,:]
 h=(lam*a['xy']/a['sy']+(1-lam)*b['xy']/b['sy'])/sx[:,None];ev,v=eigh(g,check_finite=False);return a['n']+b['n'],mx,sx,np.maximum(ev,0),v,v.T@h
def fit(sys_,a,b,alpha,lam):
 n,mx,sx,ev,v,h=sys_;alpha=np.broadcast_to(np.asarray(alpha,dtype=float),(h.shape[1],));coef=(v@(h/(ev[:,None]+alpha/n))).T/sx
 return dict(coef=coef,intercept=-coef@mx,ST_mean=a['my'],ST_scale=a['sy'],bulk_mean=b['my'],bulk_scale=b['sy'],alpha=alpha.tolist(),lambda_ST=lam,n_ST=a['n'],n_bulk=b['n'])
def predict(m,x,source):return (x@m['coef'].T+m['intercept'])*m[source+'_scale']+m[source+'_mean']
def stload(rows,programs):
 xs=[];ys=[];slices=[];start=0
 for r in rows.itertuples():
  with h5py.File(r.features) as f:x=f['features'][:].astype('float64');sid=f['spot_ids'].asstr()[:]
  if MODE=='regress':
   with np.load(r.targets,allow_pickle=True) as f:y=f['scores'];assert np.array_equal(sid,f['spot_ids'])
  else:
   with np.load(E/'st_targets'/f'{r.sample}.npz') as f:assert np.array_equal(sid,f['spot_ids']) and list(f['programs'])==programs;y=f['scores']
  xs.append(x);ys.append(y);slices.append((r.sample,r.patient_key,start,start+len(x)));start+=len(x)
 return np.concatenate(xs),np.concatenate(ys),slices
def stscore(y,pred,slices):
 K=y.shape[1];rows=[]
 for sample,pid,a,b in slices:
  for j in range(K):rows.append(dict(patient=pid,program=j,value=corr(y[a:b,j],pred[a:b,j])))
 return pd.DataFrame(rows).groupby(['patient','program']).value.mean().groupby('program').mean().reindex(range(K)).to_numpy()
if MODE=='regress':
 PROGRAMS=ORIGINAL;bulk=pd.read_csv(R/'results/t5/bulk_UCell_by_patient.csv').rename(columns=dict(zip(['E2F__mean','IFNg__mean','Hypoxia__mean','EMT__mean'],ORIGINAL)))
else:
 PROGRAMS=json.loads((O/'program_members.json').read_text());PROGRAMS=list(PROGRAMS);bulk=pd.read_csv(O/'bulk_UCell_by_patient_58.csv')
OUT=O/MODE;OUT.mkdir(exist_ok=True);MD=E/f'models_{MODE}';MD.mkdir(exist_ok=True)
index=pd.read_csv(R/'results/t6/patient_mean_feature_index.csv');xall=np.load(Path(C6['output_root'])/'patient_mean_features.npy')
folds=pd.read_csv(R/'results/t6/TCGA_patient_folds.csv');manifest=pd.read_csv(R/'results/t4/input_manifest.csv')
bulkpred=[];stscores=[];alphas_out=[]
for cohort,stcohort,ext in [('TCGA-BRCA','He2020','CPTAC-BRCA'),('TCGA-KIRC','Meylan_ccRCC','CPTAC-CCRCC')]:
 lam=LAMBDA[cohort];pts=index[index.cohort.eq(cohort)].sort_values('patient_id').merge(folds,on=['cohort','patient_id'],validate='one_to_one')
 paired=pts.merge(bulk,on=['cohort','patient_id'],validate='one_to_one');xb=xall[paired['row'].to_numpy()];yb=paired[PROGRAMS].to_numpy(float)
 xt,yt,_=stload(manifest[manifest.cohort.eq(stcohort)&manifest.split.eq('train')],PROGRAMS);xv,yv,sv=stload(manifest[manifest.cohort.eq(stcohort)&manifest.split.eq('val')],PROGRAMS)
 a=moments(xt,yt);af=moments(np.concatenate([xt,xv]),np.concatenate([yt,yv]));del xt,yt
 for outer in list(range(1,6))+[0]:
  train=np.flatnonzero(paired.fold.to_numpy()!=outer) if outer else np.arange(len(paired));metrics=[]
  for inner,(ii,jj) in enumerate(KFold(3,shuffle=True,random_state=SEED).split(train),1):
   tr,va=train[ii],train[jj];b=moments(xb[tr],yb[tr]);sys_=system(a,b,lam)
   for alpha in ALPHAS:
    m=fit(sys_,a,b,alpha,lam);st=stscore(yv,predict(m,xv,'ST'),sv);bp=predict(m,xb[va],'bulk');bs=np.array([corr(yb[va,j],bp[:,j]) for j in range(len(PROGRAMS))])
    for j,p in enumerate(PROGRAMS):metrics.append(dict(program=p,alpha=alpha,inner=inner,score=(st[j]+bs[j])/2))
   time.sleep(0.2)
  sel=pd.DataFrame(metrics).groupby(['program','alpha'],as_index=False).score.mean();chosen=[]
  for p in PROGRAMS:chosen.append(float(sel[sel.program.eq(p)].sort_values(['score','alpha'],ascending=[False,False]).iloc[0].alpha))
  b=moments(xb[train],yb[train]);m=fit(system(af,b,lam),af,b,chosen,lam);m.update(cohort=cohort,outer_fold=outer,programs=PROGRAMS);joblib.dump(m,MD/f'{cohort}_fold{outer}.joblib')
  alphas_out.extend(dict(cohort=cohort,fold=outer,program=p,alpha=al) for p,al in zip(PROGRAMS,chosen))
  te=pts[pts.fold.eq(outer)] if outer else index[index.cohort.eq(ext)];out=predict(m,xall[te['row'].to_numpy()],'bulk')
  for i,r in enumerate(te.itertuples()):bulkpred.append(dict(cohort=r.cohort,patient_id=r.patient_id,fold=outer,**dict(zip(PROGRAMS,out[i]))))
  print('FIT',MODE,cohort,outer,flush=True)
 evalrows=manifest[(manifest.cohort.eq(stcohort)&manifest.split.eq('test'))|(manifest.cohort.eq('HER2ST') if cohort=='TCGA-BRCA' else False)];xx,yy,ss=stload(evalrows,PROGRAMS);pr=predict(m,xx,'ST')
 for sample,pid,l,h in ss:
  co=evalrows[evalrows['sample'].eq(sample)].iloc[0].cohort
  for j,p in enumerate(PROGRAMS):stscores.append(dict(cohort=co,patient_key=pid,sample=sample,program=p,spearman=corr(yy[l:h,j],pr[l:h,j]),spots=h-l))
 del xx,yy,pr
bp=pd.DataFrame(bulkpred);bp.to_csv(OUT/'bulk_prediction_by_mean_feature.csv',index=False);pd.DataFrame(alphas_out).to_csv(OUT/'selected_alpha.csv',index=False)
sec=pd.DataFrame(stscores);sec.to_csv(OUT/'ST_section_spearman.csv',index=False)
pat=sec.groupby(['cohort','patient_key','program'],as_index=False).spearman.mean();pat.to_csv(OUT/'ST_patient_spearman.csv',index=False)
rows=[]
for co in ['TCGA-BRCA','TCGA-KIRC','CPTAC-BRCA','CPTAC-CCRCC']:
 d=bp[bp.cohort.eq(co)].merge(bulk,on=['cohort','patient_id'],suffixes=('_pred',''),validate='one_to_one')
 for p in PROGRAMS:rows.append(dict(cohort=co,program=p,Spearman=corr(d[p+'_pred'].to_numpy(),d[p].to_numpy()),patients=len(d)))
pd.DataFrame(rows).to_csv(OUT/'bulk_correlations.csv',index=False);print('FIT COMPLETE',MODE,flush=True)
