"""T10 补充分析（2026-10-31 用户要求）：审查"空间特征临床增量不显著"是否源于分析设计。
原设计（t6_clinical.py）：4 均值 + 18 空间特征 + 年龄、分期一次放入，固定 ridge α=0.1（未调），单次 5 折，报告折均值 C-index（折间 SD 0.05–0.07）。
本脚本（全部预设，结果全部报告，标为探索性）：
 1 全数据嵌套似然比检验（关联而非预测）：基线(年龄+分期+4均值) vs +18 空间特征；vs +E2F–IFNγ 共定位。
 2 重复 20 次分层 5 折交叉验证，合并折外风险算 C-index：M1 基线；M2 +18 空间特征，ridge 强度在训练折内 3 折交叉验证选择；
   M3 +1 个空间特征，在训练折内按调整后 Wald p 选择（嵌套选择，避免用全数据挑特征）。
 3 乳腺：E2F–IFNγ 共定位在各 PAM50 亚型内的 HR 及交互似然比检验。
 4 乳腺：共定位与 Saltz 影像 TIL 的预后信息比较（Saltz 匹配子集，同一重复交叉验证）。
特征来自 T7 交叉拟合全片图（每位患者的图来自未见其 bulk 的表达模型；PFI 标签从未参与表达模型）。"""
from t10_common import *
from statsmodels.duration.hazard_regression import PHReg
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold
MEANS=['E2F__mean','IFNg__mean','Hypoxia__mean','EMT__mean'];SPACE=[f'{s}__{k}' for k in ['moran','high_fraction','entropy'] for s in ['E2F','IFNg','Hypoxia','EMT']]+[f'{a}__{b}__coloc' for a,b in [('E2F','IFNg'),('E2F','Hypoxia'),('E2F','EMT'),('IFNg','Hypoxia'),('IFNg','EMT'),('Hypoxia','EMT')]];F='E2F__IFNg__coloc'
REPS=20;ALPHAS=[.001,.01,.03,.1,.3];SEED=20261031
def cindex(time,event,risk):
 s=0.;n=0
 for i in np.flatnonzero(event):
  m=(time>time[i])|((time==time[i])&(event==0));d=risk[i]-risk[m];s+=np.sum(d>0)+.5*np.sum(d==0);n+=len(d)
 return s/n if n else np.nan
def std(tr,te):
 mu=tr.mean(0);sd=tr.std(0);sd[sd==0]=1;return (tr-mu)/sd,(te-mu)/sd
def cox(x,t,e,alpha=None):
 m=PHReg(t,x,status=e,ties='breslow');return m.fit(disp=False) if alpha is None else _Ridge(ridge_cox(x,t,e,alpha))
class _Ridge:
 def __init__(s,b):s.params=b
def ridge_cox(x,t,e,alpha,it=50):
 # Breslow 偏似然的 ridge Cox（牛顿法），目标与 statsmodels fit_regularized(L1_wt=0) 相同：-llf/n + alpha*|b|²/2
 o=np.argsort(-t,kind='stable');x,t,e=x[o],t[o],e[o].astype(float);n,p=x.shape;b=np.zeros(p)
 # 并列时间：风险集取到同一时间的最后一位
 last=np.r_[np.flatnonzero(t[1:]!=t[:-1]),n-1];grp=np.searchsorted(last,np.arange(n))
 for _ in range(it):
  w=np.exp(x@b-np.max(x@b));S0=np.cumsum(w)[last][grp];S1=np.cumsum(w[:,None]*x,0)[last][grp];S2=np.cumsum(w[:,None,None]*(x[:,:,None]*x[:,None,:]),0)[last][grp]
  m1=S1/S0[:,None];g=(e[:,None]*(x-m1)).sum(0)/n-alpha*b;H=-(e[:,None,None]*(S2/S0[:,None,None]-m1[:,:,None]*m1[:,None,:])).sum(0)/n-alpha*np.eye(p)
  step=np.linalg.solve(H,g);b=b-step
  if np.abs(step).max()<1e-9:break
 return b
def load(co):
 d=pd.read_csv(R/'results/t7/patient_spatial_clinical.csv');d=d[d.cohort.eq(co)&d.PFI.isin([0,1])&(d.PFI_time>0)&d.age.notna()&d.stage_I_IV.isin(['I','II','III','IV'])].copy();d['stage_high']=d.stage_I_IV.isin(['III','IV']).astype(int)
 return d.dropna(subset=MEANS+SPACE).reset_index(drop=True)
BASE=['age','stage_high']+MEANS
def lrt(d,extra):
 t,e=d.PFI_time.to_numpy(),d.PFI.to_numpy().astype(int);x0=d[BASE].to_numpy(float);x1=d[BASE+extra].to_numpy(float);x0,_=std(x0,x0);x1,_=std(x1,x1)
 l0=cox(x0,t,e).llf;l1=cox(x1,t,e).llf;return dict(added=';'.join(extra) if len(extra)<3 else f'{len(extra)} spatial features',df=len(extra),LR=2*(l1-l0),p=float(chi2.sf(2*(l1-l0),len(extra))))
def oof_risk(d,cols,tr,te,mode,inner_seed):
 t,e=d.PFI_time.to_numpy(),d.PFI.to_numpy().astype(int)
 if mode=='select':
  # 训练折内：逐个空间特征加到基线，按 Wald p 选最小者
  best=None
  for f in SPACE:
   x,_=std(d.loc[tr,BASE+[f]].to_numpy(float),d.loc[te,BASE+[f]].to_numpy(float));r=cox(x,t[tr],e[tr]);p=r.pvalues[-1]
   if best is None or p<best[0]:best=(p,f)
  cols=BASE+[best[1]];mode='plain'
 xtr,xte=std(d.loc[tr,cols].to_numpy(float),d.loc[te,cols].to_numpy(float))
 if mode=='plain':return xte@cox(xtr,t[tr],e[tr]).params,cols[-1]
 # ridge：训练折内 3 折选 α
 sk=StratifiedKFold(3,shuffle=True,random_state=inner_seed);sc={}
 for a in ALPHAS:
  rs=[];
  for i_tr,i_va in sk.split(tr,e[tr]):
   A,B=std(d.loc[tr[i_tr],cols].to_numpy(float),d.loc[tr[i_va],cols].to_numpy(float));rs.append(cindex(t[tr[i_va]],e[tr[i_va]],B@cox(A,t[tr[i_tr]],e[tr[i_tr]],a).params))
  sc[a]=np.mean(rs)
 a=max(sc,key=lambda k:(sc[k],k));return xte@cox(xtr,t[tr],e[tr],a).params,a
def repeated_cv(d,models,tag):
 t,e=d.PFI_time.to_numpy(),d.PFI.to_numpy().astype(int);rows=[];picks=[]
 for rep in range(REPS):
  sk=StratifiedKFold(5,shuffle=True,random_state=SEED+rep);risk={m:np.zeros(len(d)) for m in models}
  for k,(tr,te) in enumerate(sk.split(np.zeros(len(d)),e)):
   for m,(cols,mode) in models.items():
    r,info=oof_risk(d,cols,tr,te,mode,SEED+rep*10+k)
    # 折间风险标度不同：折内中心化后再合并（C-index 只看折内与跨折排序，中心化减少跨折偏移）
    risk[m][te]=r-np.median(r);picks.append(dict(set=tag,rep=rep,fold=k,model=m,choice=str(info)))
  rows+=[dict(set=tag,rep=rep,model=m,C=cindex(t,e,risk[m])) for m in models]
  print(tag,'rep',rep,{m:round(rows[-len(models)+i]['C'],3) for i,m in enumerate(models)},flush=True)
 return pd.DataFrame(rows),pd.DataFrame(picks)
if __name__=='__main__':
 res=[];cvs=[];pk=[];sub=[]
 for co in ['TCGA-BRCA','TCGA-KIRC']:
  d=load(co);res+=[dict(cohort=co,n=len(d),events=int(d.PFI.sum()),**lrt(d,SPACE)),dict(cohort=co,n=len(d),events=int(d.PFI.sum()),**lrt(d,[F]))]
  c,p=repeated_cv(d,{'M1 base':(BASE,'plain'),'M2 +18 spatial (tuned ridge)':(BASE+SPACE,'ridge'),'M3 +1 spatial (nested selection)':(BASE,'select')},co);cvs.append(c);pk.append(p)
 # 亚型内共定位
 d=load('TCGA-BRCA');d=d[d.PAM50Call_RNAseq.isin(['Basal','Her2','LumA','LumB'])].reset_index(drop=True);t,e=d.PFI_time.to_numpy(),d.PFI.to_numpy().astype(int)
 for s_ in ['Basal','Her2','LumA','LumB']:
  g=d[d.PAM50Call_RNAseq.eq(s_)];x,_=std(g[BASE+[F]].to_numpy(float),g[BASE+[F]].to_numpy(float));r=cox(x,g.PFI_time.to_numpy(),g.PFI.to_numpy().astype(int));b,se=r.params[-1],r.bse[-1]
  sub.append(dict(subtype=s_,n=len(g),events=int(g.PFI.sum()),HR=float(np.exp(b)),CI_low=float(np.exp(b-1.96*se)),CI_high=float(np.exp(b+1.96*se)),p=float(r.pvalues[-1])))
 dm=pd.get_dummies(d.PAM50Call_RNAseq,drop_first=True).astype(float);zf=(d[F]-d[F].mean())/d[F].std();X0=np.column_stack([std(d[BASE].to_numpy(float),d[BASE].to_numpy(float))[0],dm.to_numpy(),zf]);X1=np.column_stack([X0,dm.to_numpy()*zf.to_numpy()[:,None]])
 l0,l1=cox(X0,t,e).llf,cox(X1,t,e).llf;sub.append(dict(subtype='interaction LRT (3 df)',n=len(d),events=int(e.sum()),HR=np.nan,CI_low=np.nan,CI_high=np.nan,p=float(chi2.sf(2*(l1-l0),3))))
 # 共定位 vs TIL
 til=pd.read_csv(R/'results/t8_1b/coloc_vs_TIL_density_patients.csv')[['patient_id','til_percentage']];dt=load('TCGA-BRCA').merge(til,on='patient_id').dropna(subset=['til_percentage']).reset_index(drop=True);dt['logtil']=np.log1p(dt.til_percentage)
 c,p=repeated_cv(dt,{'M1 base':(BASE,'plain'),'M1 + log TIL%':(BASE+['logtil'],'plain'),'M1 + co-localization':(BASE+[F],'plain'),'M1 + both':(BASE+['logtil',F],'plain')},'TCGA-BRCA Saltz subset');cvs.append(c);pk.append(p)
 res+=[dict(cohort='TCGA-BRCA Saltz subset',n=len(dt),events=int(dt.PFI.sum()),**x) for x in []]
 cv=pd.concat(cvs);cv.to_csv(D/'clinical_review_repeatedCV.csv',index=False);pd.concat(pk).to_csv(D/'clinical_review_choices.csv',index=False);pd.DataFrame(res).to_csv(D/'clinical_review_LRT.csv',index=False);pd.DataFrame(sub).to_csv(D/'clinical_review_subtype.csv',index=False)
 # 汇总：每次重复内与 M1 的差
 base_c=cv[cv.model.eq('M1 base')].set_index(['set','rep']).C;cv['delta_vs_M1']=cv.apply(lambda r:r.C-base_c[(r.set,r.rep)],axis=1)
 summ=cv.groupby(['set','model'],as_index=False).agg(C_mean=('C','mean'),delta_mean=('delta_vs_M1','mean'),delta_q025=('delta_vs_M1',lambda v:v.quantile(.025)),delta_q975=('delta_vs_M1',lambda v:v.quantile(.975)),frac_delta_pos=('delta_vs_M1',lambda v:(v>0).mean()))
 summ.to_csv(D/'clinical_review_summary.csv',index=False);print(pd.DataFrame(res).round(4).to_string());print(summ.round(4).to_string());print(pd.DataFrame(sub).round(3).to_string())
 ch=pd.concat(pk);print(ch[ch.model.str.startswith('M3')].groupby(['set','choice']).size().sort_values(ascending=False).groupby(level=0).head(4))
