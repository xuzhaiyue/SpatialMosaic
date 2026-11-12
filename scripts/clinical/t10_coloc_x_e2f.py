"""T10 补充（2026-11-08 用户同意）：E2F–IFNγ 共定位的预后关联是否随肿瘤整体增殖水平变化（Nagalla 2013 假说的空间版本）。
预设：主检验 = Cox（年龄、二值分期、四均值、共定位）加入 共定位 × E2F 均值（均标准化），1 自由度似然比检验；
次要 = 按 E2F 均值三分位分层的共定位 HR；对照 = 共定位 × IFNγ 均值（检验特异性）。乳腺为主，肾同法报告。"""
from t10_common import *
from statsmodels.duration.hazard_regression import PHReg
from scipy.stats import chi2
MEANS=['E2F__mean','IFNg__mean','Hypoxia__mean','EMT__mean'];F='E2F__IFNg__coloc';BASE=['age','stage_high']+MEANS
def load(co):
 d=pd.read_csv(R/'results/t7/patient_spatial_clinical.csv');d=d[d.cohort.eq(co)&d.PFI.isin([0,1])&(d.PFI_time>0)&d.age.notna()&d.stage_I_IV.isin(['I','II','III','IV'])].copy()
 d['stage_high']=d.stage_I_IV.isin(['III','IV']).astype(int);return d.dropna(subset=MEANS+[F]).reset_index(drop=True)
def z(v):return (v-v.mean())/v.std()
def fit(d,X):return PHReg(d.PFI_time.to_numpy(),X,status=d.PFI.to_numpy().astype(int),ties='breslow').fit(disp=False)
rows=[];strata=[]
for co in ['TCGA-BRCA','TCGA-KIRC']:
 d=load(co);Z=pd.DataFrame({c:z(d[c].astype(float)) for c in BASE+[F]})
 m0=fit(d,Z[BASE+[F]].to_numpy())
 for mod in ['E2F__mean','IFNg__mean']:
  X1=np.column_stack([Z[BASE+[F]].to_numpy(),(Z[F]*Z[mod]).to_numpy()]);m1=fit(d,X1);lr=2*(m1.llf-m0.llf)
  b,se=m1.params[-1],m1.bse[-1]
  rows.append(dict(cohort=co,n=len(d),events=int(d.PFI.sum()),modifier=mod,interaction_coef=float(b),interaction_HR=float(np.exp(b)),CI_low=float(np.exp(b-1.96*se)),CI_high=float(np.exp(b+1.96*se)),LRT=float(lr),p=float(chi2.sf(lr,1)),coloc_HR_at_mean=float(np.exp(m1.params[len(BASE)]))))
 # E2F 均值三分位内的共定位 HR（同协变量，组内重新标准化）
 d['E2F_tertile']=pd.qcut(d['E2F__mean'],3,labels=['low','middle','high'])
 for t_,g in d.groupby('E2F_tertile',observed=True):
  X=np.column_stack([z(g[c].astype(float)).fillna(0) for c in BASE+[F]]);m=fit(g,X);b,se=m.params[-1],m.bse[-1]
  strata.append(dict(cohort=co,E2F_tertile=t_,n=len(g),events=int(g.PFI.sum()),HR=float(np.exp(b)),CI_low=float(np.exp(b-1.96*se)),CI_high=float(np.exp(b+1.96*se)),p=float(m.pvalues[-1])))
r=pd.DataFrame(rows);s=pd.DataFrame(strata);r.to_csv(D/'coloc_x_E2F_interaction.csv',index=False);s.to_csv(D/'coloc_by_E2F_tertile.csv',index=False)
print(r.round(4).to_string());print(s.round(3).to_string())
