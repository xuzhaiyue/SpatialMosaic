"""T10 补充核查：官方 UNI2-h patch 竖条状采样缺口是否制造了 E2F–IFNγ 共定位关联。
竖条指数 = 相邻两列 patch 数相对差 > 0.5 的列对比例（每张切片；患者取 patch 数加权均值）。24 例上 ≤0.05 与 ≥0.09 明显两群，规整阈值预设为 < 0.05。
检验：竖条指数与共定位的相关；Cox（同 Fig 5b 基础协变量）加入竖条指数；只在规整切片患者中重估共定位 HR。"""
from t10_common import *
import h5py
from statsmodels.duration.hazard_regression import PHReg
PP=E/'t7/patch_predictions'
OUT=R/'results/t10/qc/TCGA_BRCA_sampling_stripes.csv'
def stripe(path):
 with h5py.File(path) as f:xy=f['coords_top_left'][:];st=int(f.attrs['grid_step'])
 gx=(xy[:,0]//st).astype(int);c=np.bincount(gx-gx.min()).astype(float);a,b=c[:-1],c[1:];m=np.maximum(a,b);ok=m>0
 return float(np.mean(np.abs(a-b)[ok]/m[ok]>.5)) if ok.any() else np.nan,len(xy)
def value_stripe(path,cols=(0,1),k=7):
 """预测值竖条指数（Fig 5d 选例用）：各程序片内百分位减去 k×k 局部均值后，
 残差的列均值方差 / 行均值方差（只计 ≥10 个 patch 的列/行），取所给程序的最大值。采样完整但预测值呈竖条带时升高。"""
 from scipy.ndimage import uniform_filter
 with h5py.File(path) as f:xy=f['coords_top_left'][:];pr=f['predicted'][:];st=int(f.attrs['grid_step'])
 gx=(xy[:,0]//st).astype(int);gy=(xy[:,1]//st).astype(int);gx-=gx.min();gy-=gy.min();out=[]
 for j in cols:
  M=np.zeros((gy.max()+1,gx.max()+1));W=np.zeros_like(M);M[gy,gx]=pd.Series(pr[:,j]).rank(pct=True).to_numpy();W[gy,gx]=1
  Rr=np.where(W>0,M-uniform_filter(M,k)/np.maximum(uniform_filter(W,k),1e-9),np.nan)
  with np.errstate(all='ignore'):cm=np.nanmean(Rr[:,W.sum(0)>=10],0);rm=np.nanmean(Rr[W.sum(1)>=10],1)
  out.append(float(np.nanvar(cm)/np.nanvar(rm)))
 return max(out)
if __name__=='__main__':
 if not OUT.exists():
  rows=[]
  for p in sorted(x for x in (PP/'TCGA-BRCA').glob('*.h5') if not x.name.startswith('._')):  # 跳过 exFAT 的 ._ 资源文件
   s,n=stripe(p);rows.append(dict(slide=p.stem,patient_id=p.stem[:12],stripe=s,patches=n))
  sl=pd.DataFrame(rows);sl.to_csv(OUT.with_name('TCGA_BRCA_sampling_stripes_slides.csv'),index=False)
  pt=sl.groupby('patient_id').apply(lambda g:pd.Series(dict(stripe=np.average(g.stripe,weights=g.patches),slides=len(g)))).reset_index();pt.to_csv(OUT,index=False)
 pt=pd.read_csv(OUT)
 MEANS=['E2F__mean','IFNg__mean','Hypoxia__mean','EMT__mean'];F='E2F__IFNg__coloc';base=['age','stage_high']+MEANS
 d=pd.read_csv(R/'results/t7/patient_spatial_clinical.csv');d=d[d.cohort.eq('TCGA-BRCA')&d.PFI.isin([0,1])&(d.PFI_time>0)&d.age.notna()&d.stage_I_IV.isin(['I','II','III','IV'])].copy()
 d['stage_high']=d.stage_I_IV.isin(['III','IV']).astype(int);d=d.merge(pt,on='patient_id',how='left');assert d.stripe.notna().all()
 res=[]
 def fit(df,cols,label):
  x=df[cols].to_numpy(float);sd=x.std(0);sd[sd==0]=1;x=(x-x.mean(0))/sd
  m=PHReg(df.PFI_time.to_numpy(),x,status=df.PFI.to_numpy(),ties='breslow').fit(disp=False);b,se=m.params[-1],m.bse[-1]
  res.append(dict(model=label,n=len(df),events=int(df.PFI.sum()),HR=float(np.exp(b)),CI_low=float(np.exp(b-1.96*se)),CI_high=float(np.exp(b+1.96*se)),p=float(m.pvalues[-1])))
 fit(d,base+[F],'T7 model (all)');fit(d,base+['stripe',F],'+ stripe index')
 reg=d[d.stripe<.05];fit(reg,base+[F],'regular sampling only (stripe < 0.05)');fit(d[d.stripe>=.05],base+[F],'striped sampling only (stripe ≥ 0.05)')
 res=pd.DataFrame(res);res.to_csv(D/'sampling_stripes_cox.csv',index=False)
 summ=dict(patients=len(d),frac_striped=float((d.stripe>=.05).mean()),rho_stripe_coloc=corr(d.stripe,d[F]),coloc_median_regular=float(reg[F].median()),coloc_median_striped=float(d.loc[d.stripe>=.05,F].median()))
 pd.Series(summ).to_csv(D/'sampling_stripes_summary.csv');print(pd.Series(summ).round(3).to_string());print(res.round(3).to_string())
