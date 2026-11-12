"""T10 Fig 5：空间特征与预后，以及它在形态上是什么（TCGA，交叉拟合图）。
a 单特征 Cox 森林图（乳腺 / 肾各 18 项，调整年龄、二值分期、四程序均值）；
b E2F–IFNγ 共定位稳健性；
c 共定位三分位 KM（未调整、描述性）；d 共定位最高 / 最低病例（H&E + E2F / IFNγ 全片图），按排名依次取、跳过无 H&E 者及竖条采样者（竖条指数 ≥ 0.05，t10_sampling_stripes.py）
  及预测值竖条者（E2F / IFNγ 预测值竖条指数 ≥ 3，value_stripe；24 例中仅 1 例 > 3，次高 2.73）；
e 共定位与 Saltz 影像 TIL 密度、TIL 模式。
全片图空白为官方未采样 patch（部分切片呈竖条状）；b 中加入采样规整度敏感性分析。"""
from t10_common import *
import h5py
from PIL import Image
from statsmodels.duration.hazard_regression import PHReg
RS=R/'results'
# ---------- b 稳健性（d3.py 原逻辑） ----------
MEANS=['E2F__mean','IFNg__mean','Hypoxia__mean','EMT__mean'];F='E2F__IFNg__coloc'
d=pd.read_csv(RS/'t7/patient_spatial_clinical.csv');d=d[d.cohort.eq('TCGA-BRCA')&d.PFI.isin([0,1])&(d.PFI_time>0)&d.age.notna()&d.stage_I_IV.isin(['I','II','III','IV'])].copy()
d['stage_high']=d.stage_I_IV.isin(['III','IV']).astype(int);t=pd.read_csv(RS/'t8_1b/coloc_vs_TIL_density_patients.csv')[['patient_id','til_percentage','til_pattern4']]
d=d.merge(t,on='patient_id',how='left');d['tss']=d.patient_id.str.split('-').str[1];d['logtil']=np.log1p(d.til_percentage);rob=[]
def fit(df,cols,strata=None,label=''):
 df=df.dropna(subset=cols);x=df[cols].to_numpy(float);sd=x.std(0);sd[sd==0]=1;x=(x-x.mean(0))/sd
 m=PHReg(df.PFI_time.to_numpy(),x,status=df.PFI.to_numpy(),ties='breslow',strata=None if strata is None else df[strata].to_numpy()).fit(disp=False);b=m.params[-1];se=m.bse[-1]
 rob.append(dict(model=label,n=len(df),events=int(df.PFI.sum()),HR=float(np.exp(b)),CI_low=float(np.exp(b-1.96*se)),CI_high=float(np.exp(b+1.96*se)),p=float(m.pvalues[-1])))
base=['age','stage_high']+MEANS;dt=d[d.til_percentage.notna()]
fit(d,base+[F],label='T7 model');fit(dt,base+[F],label='Saltz-matched subset');fit(dt,base+['logtil',F],label='+ log TIL%')
dp=dt.copy();dp=pd.concat([dp,pd.get_dummies(dp.til_pattern4,prefix='pat',drop_first=True).astype(float)],axis=1);pc=[c for c in dp.columns if c.startswith('pat_')]
fit(dp,base+['logtil']+pc+[F],label='+ log TIL% + TIL pattern');fit(d,base+[F],strata='tss',label='stratified by tissue source site');fit(dt,base+['logtil',F],strata='tss',label='+ log TIL%, stratified by site')
# 官方采样规整度（部分全片图呈竖条状）：列填充率与包围盒填充率；只在有 Saltz 匹配的患者上计算
SQ=R/'results/t10/qc/TCGA_BRCA_sampling_regularity.csv'
if not SQ.exists():
 rows=[]
 for r in pd.read_csv(RS/'t8_1b/coloc_vs_TIL_density_patients.csv').itertuples():
  with h5py.File(r.prediction_path) as f:xy=f['coords_top_left'][:];st=int(f.attrs['grid_step'])
  g=np.unique((xy//st).astype(int),axis=0);ux=np.unique(g[:,0]);rows.append(dict(patient_id=r.patient_id,coloc=r.E2F__IFNg__coloc,colfill=len(ux)/(ux.max()-ux.min()+1),fill=len(g)/((ux.max()-ux.min()+1)*(g[:,1].max()-g[:,1].min()+1)),step=st,n=len(xy)))
 pd.DataFrame(rows).to_csv(SQ,index=False)
ds=dt.merge(pd.read_csv(SQ)[['patient_id','colfill','fill']],on='patient_id');fit(ds,base+['colfill','fill',F],label='+ sampling regularity')
rob=pd.DataFrame(rob);rob.to_csv(D/'Fig5b_robustness.csv',index=False)
# 自检：与先前诊断结果一致
assert abs(rob.set_index('model').HR['+ sampling regularity']-.795)<6e-4
for lab,hr in [('T7 model',.745),('+ log TIL%',.850),('stratified by tissue source site',.779),('+ log TIL%, stratified by site',.880)]:assert abs(rob.set_index('model').HR[lab]-hr)<6e-4,(lab,rob)

# 字号三档（面板字母 8 pt 粗体 / 面板标题 7 pt / 其余 6 pt）；上下两行等高（各约 80 mm）
from statsmodels.duration.survfunc import SurvfuncRight,survdiff
C=Canvas(180,170);A=C.ax;FS=6;TT=7
mpl.rcParams['legend.fontsize']=FS
def head(x,y,l,t):C.head(x,y,l,t,fs=TT)
def tk(a):ticks(a,FS,FS)
def logx(a,ticks_,lim):a.set_xscale('log');a.set_xticks(ticks_);a.set_xticklabels([f'{t:g}' for t in ticks_]);a.xaxis.set_minor_locator(mpl.ticker.NullLocator());a.set_xlim(*lim)
def ylab_groups(a,ylist,labels,headers,xr):
 # 左侧两级行标签：组名（粗体，左对齐 xr mm）+ 条目（右对齐到轴）
 a.set_yticks(ylist);a.set_yticklabels(labels,fontsize=FS);a.tick_params(axis='y',length=0,pad=2)
 tr=mpl.transforms.blended_transform_factory(C.fig.transFigure,a.transData)
 for y_,t_ in headers:a.text(xr/C.W,y_,t_,transform=tr,fontsize=FS,fontweight='bold',va='center',ha='left')
def grouped_rows(blocks):
 # blocks: [(组名, 条目列表)] → 每条 y 坐标、组名 y 坐标、组间分隔线 y
 ys=[];heads=[];seps=[];yy=0
 for gi,(ttl,items) in enumerate(blocks):
  if gi:seps.append(yy+.6)
  heads.append((yy,ttl));yy-=1
  for _ in items:ys.append(yy);yy-=1
  yy-=.2
 return ys,heads,seps,yy
DARK='#3E2766'  # 共定位主色（与 d 双变量"两者都高"一致）
R1,R1H=14,56  # 第一行坐标轴：y 14–70
R2=93         # 第二行内容起点

# ---------- a 单特征 Cox：两癌合并 ----------
head(2,5,'a','Spatial features and progression-free interval')
cx=pd.read_csv(RS/'t7/Cox_single_feature_associations.csv')
assert (cx[cx.cohort.eq('TCGA-BRCA')].feature.to_numpy()==cx[cx.cohort.eq('TCGA-KIRC')].feature.to_numpy()).all()
feat=cx[cx.cohort.eq('TCGA-BRCA')].feature.tolist();KIND=[('moran',"Moran's I"),('high_fraction','High fraction'),('entropy','Local entropy'),('coloc','Co-localization')]
def pname(f):return '–'.join({'IFNg':'IFNγ'}.get(x,x) for x in f.split('__')[:-1])
blocks=[(t,[f for f in feat if f.endswith('__'+k)]) for k,t in KIND];ys,heads,seps,yend=grouped_rows(blocks);rowy=dict(zip([f for _,b in blocks for f in b],ys))
a=A(24,R1,58,R1H);OFF=.2
for co,dy,fill in [('TCGA-BRCA',OFF,True),('TCGA-KIRC',-OFF,False)]:
 for r in cx[cx.cohort.eq(co)].itertuples():
  y_=rowy[r.feature]+dy;sig=r.q<.05;col=DARK if sig else ('#3a3a3a' if fill else '#8a8a8a')
  a.plot([r.CI_low,r.CI_high],[y_]*2,color=col,lw=.9 if sig else .6)
  a.scatter(r.HR,y_,s=12 if sig else 6,zorder=3,**(dict(color=col,lw=0) if fill else dict(facecolor='white',edgecolor=col,lw=.5)))
  if sig:a.text(r.CI_low/1.04,y_,f'q = {r.q:.3f}',va='center',ha='right',fontsize=FS,color=DARK)
a.axvline(1,color='#bbbbbb',lw=.5,ls=(0,(3,2)),zorder=0);logx(a,[.5,.7,1,1.4,2],(.4,2.3));a.set_ylim(yend+.5,.6)
for y_ in seps:a.axhline(y_,color='#e6e6e6',lw=.4,zorder=0)
a.set_xlabel('Hazard ratio per s.d. (95% CI)');tk(a);ylab_groups(a,list(rowy.values()),[pname(f) for f in rowy],heads,4);a.spines['left'].set_visible(False)
gb=cx[cx.cohort.eq('TCGA-BRCA')].iloc[0];gk=cx[cx.cohort.eq('TCGA-KIRC')].iloc[0]
lg=A(24,8.6,58,3.6);lg.set_xlim(0,58);lg.set_ylim(3.6,0);lg.set_axis_off()
lg.scatter([1],[1.8],s=6,color='#3a3a3a',lw=0);lg.text(2.4,1.8,f'TCGA-BRCA ({int(gb.n):,}; {int(gb.events)} events)',fontsize=FS,va='center')
lg.scatter([36],[1.8],s=6,facecolor='white',edgecolor='#8a8a8a',lw=.5);lg.text(37.4,1.8,f'TCGA-KIRC ({int(gk.n)}; {int(gk.events)})',fontsize=FS,va='center')

# ---------- b 稳健性 + 亚型内 + E2F 三分位内（亚型 HR 来自 t10_clinical_review.py；联合检验与嵌套选择同源） ----------
head(90,5,'b','Co-localization HR in sensitivity analyses, TCGA-BRCA')
sub=pd.read_csv(D/'clinical_review_subtype.csv');lrt=pd.read_csv(D/'clinical_review_LRT.csv');chs=pd.read_csv(D/'clinical_review_choices.csv')
sel=chs[(chs.set=='TCGA-BRCA')&chs.model.str.startswith('M3')];nsel=int((sel.choice==F).sum())
sb=sub[sub.subtype.isin(['Basal','Her2','LumA','LumB'])].rename(columns={'subtype':'model'})
# 采样竖条核查（t10_sampling_stripes.py）
st_=pd.read_csv(D/'sampling_stripes_cox.csv').set_index('model').loc[['+ stripe index','regular sampling only (stripe < 0.05)']].reset_index();st_['model']=['+ sampling stripe index','regular-sampling slides only']
rob_=pd.concat([rob[rob.model.ne('+ sampling regularity')],st_],ignore_index=True);rob_['model']=rob_.model.replace({'T7 model':'Primary model'})
# E2F 均值三分位内（t10_coloc_x_e2f.py）
et=pd.read_csv(D/'coloc_by_E2F_tertile.csv');et=et[et.cohort.eq('TCGA-BRCA')].copy();et['model']=et.E2F_tertile.str.capitalize();ix=pd.read_csv(D/'coloc_x_E2F_interaction.csv').set_index(['cohort','modifier']).p
cols=['model','n','events','HR','CI_low','CI_high','p'];bl=[('Adjustment and sampling',rob_[cols]),('Within PAM50 subtype',sb[cols]),('Within E2F-mean tertile',et[cols])]
ys,heads,seps,yend=grouped_rows([(t,b.model.tolist()) for t,b in bl]);rb=pd.concat([b.assign(block=t) for t,b in bl],ignore_index=True);rb['y']=ys
a=A(130,R1,19,R1H);tr=mpl.transforms.blended_transform_factory(C.fig.transFigure,a.transData)
for r in rb.itertuples():
 sig=r.p<.05;pm=r.model=='Primary model';col=DARK if pm else ('#3a3a3a' if sig else '#9a9a9a')
 a.plot([r.CI_low,r.CI_high],[r.y]*2,color=col,lw=.6);a.scatter(r.HR,r.y,s=12 if pm else 6,color=col,zorder=3,lw=0,marker='D' if pm else 'o')
 for xx,t_ in [(158,f'{r.HR:.2f}'),(168.5,f'{r.p:.3f}'),(179,f'{r.n:,}')]:a.text(xx/C.W,r.y,t_,transform=tr,ha='right',va='center',fontsize=FS,color=INK if sig else '#8a8a8a')
for xx,t_ in [(158,'HR'),(168.5,'p'),(179,'n')]:a.text(xx/C.W,.6,t_,transform=tr,ha='right',va='center',fontsize=FS,color=MUTED,style='normal' if t_=='HR' else 'italic')
a.axvline(1,color='#bbbbbb',lw=.5,ls=(0,(3,2)),zorder=0);logx(a,[.2,.5,1],(.15,1.7));a.set_ylim(yend+.5,1.1)
for y_ in seps:a.axhline(y_,color='#e6e6e6',lw=.4,zorder=0)
a.set_xlabel('HR per s.d.');tk(a);ylab_groups(a,rb.y.tolist(),rb.model.tolist(),heads,92);a.spines['left'].set_visible(False)
lr=lrt[lrt.cohort.eq('TCGA-BRCA')].set_index('df').p
NOTE=dict(LRT_p_18_features=float(lr[18]),nested_folds_selecting_coloc=f'{nsel}/{len(sel)}',subtype_interaction_p=float(sub[sub.subtype.str.startswith('interaction')].p.iloc[0]),E2F_mean_interaction_p=float(ix[('TCGA-BRCA','E2F__mean')]))

# ---------- c KM（共定位三分位，未调整，描述性；本 Cox 集 1,018 人内三分） ----------
head(2,R2-6,'c','PFI by co-localization tertile')
km=d[['patient_id','PFI','PFI_time',F]].copy();km['tert']=pd.qcut(km[F],3,labels=['Low','Middle','High']);km['yr']=km.PFI_time/365.25
lrt_chi,lrt_p=survdiff(km.PFI_time.to_numpy(),km.PFI.to_numpy(),km.tert.cat.codes.to_numpy())
TC={'Low':'#B9AAD3','Middle':'#7E69AC','High':DARK};KX,KY,KW,KH=16,R2,40,38;XMAX=10;kmrows=[]
a=A(KX,KY,KW,KH)
for t_ in ['Low','Middle','High']:
 g=km[km.tert.eq(t_)];sf=SurvfuncRight(g.yr.to_numpy(),g.PFI.to_numpy());tt=np.r_[0,sf.surv_times];ss=np.r_[1,sf.surv_prob]
 a.step(tt,ss,where='post',color=TC[t_],lw=1.0);cz=g[(g.PFI==0)&(g.yr<=XMAX)].yr.to_numpy();a.plot(cz,ss[np.searchsorted(tt,cz,side='right')-1],'|',color=TC[t_],ms=2,mew=.4)  # 删失标记，取阶梯函数右连续值
 kmrows+=[dict(tertile=t_,time_years=x,surv=y) for x,y in zip(tt,ss)]
a.set_xlim(0,XMAX);a.set_ylim(.5,1.01);a.set_xticks(range(0,XMAX+1,2));a.set_yticks([.5,.75,1]);a.set_xlabel('Years');a.set_ylabel('PFI probability');tk(a)
for i,t_ in enumerate(['High','Middle','Low']):a.plot([.4,1.4],[.64-i*.045]*2,color=TC[t_],lw=1);a.text(1.7,.64-i*.045,t_,fontsize=FS,va='center')
a.text(9.8,.985,f'log-rank p = {lrt_p:.4f}',fontsize=FS,va='top',ha='right')
C.text(KX-13,KY+KH+11,'No. at risk',fontsize=FS,fontweight='bold',ha='left',va='baseline')
for i,t_ in enumerate(['High','Middle','Low']):
 yb=KY+KH+14.5+i*3.2;C.text(KX-3,yb,t_,fontsize=FS,color=TC[t_],ha='right',va='center');g=km[km.tert.eq(t_)]
 for x_ in range(0,XMAX+1,2):C.text(KX+KW*x_/XMAX,yb,f'{int((g.yr>=x_).sum())}',fontsize=FS,ha='center',va='center')

# ---------- d 高 / 低共定位病例：E2F × IFNγ 双变量图 ----------
head(62,R2-6,'d','Extreme cases, bivariate maps')
hl=pd.read_csv(RS/'t8_1b/high_low_24_cases.csv');he=pd.read_csv(RS/'t8_1b/HE_index.csv').set_index('slide')
from t10_sampling_stripes import stripe,value_stripe
hl['stripe']=[stripe(x)[0] for x in hl.prediction_path];hl['vstripe']=[value_stripe(x) for x in hl.prediction_path]
picks=pd.concat([g[g.slide.isin(he.index)&(g.stripe<.05)&(g.vstripe<3)].sort_values('rank').head(2) for _,g in hl.groupby('group',sort=False)])
# 双变量配色：四角双线性插值（低/低 浅灰，仅 E2F 高 橙，仅 IFNγ 高 蓝，两者都高 深紫）
BV=np.array([[[236,236,236],[58,143,203]],[[230,130,58],[62,39,102]]])/255  # [E2F 低/高][IFNγ 低/高]
def bivar(u,v):
 u=u[...,None];v=v[...,None];return (1-u)*(1-v)*BV[0,0]+(1-u)*v*BV[0,1]+u*(1-v)*BV[1,0]+u*v*BV[1,1]
crows=[];DX0,IW,IH=64,23,12.5
yrow={('High',0):R2+2.5,('High',1):R2+20,('Low',0):R2+43,('Low',1):R2+60.5}
for grp in ['High','Low']:
 C.text(DX0,yrow[(grp,0)]-4,('Highest' if grp=='High' else 'Lowest')+' co-localization',fontsize=FS,fontweight='bold',va='baseline')
 for ci,r in enumerate(picks[picks.group.eq(grp)].itertuples()):
  y0=yrow[(grp,ci)];h=he.loc[r.slide];im=Image.open(h.png).convert('RGB')
  with h5py.File(r.prediction_path) as f:xy=f['coords_top_left'][:];pr=f['predicted'][:];step=int(f.attrs['grid_step'])
  gx=(xy[:,0]//step).astype(int);gy=(xy[:,1]//step).astype(int);ncol=int(np.ceil(h.width_level0/step));nrow=int(np.ceil(h.height_level0/step))
  a=A(DX0,y0,IW,IH);a.imshow(im,extent=[0,ncol,nrow,0]);a.set_xlim(0,ncol);a.set_ylim(nrow,0);a.set_axis_off()
  u=pd.Series(pr[:,0]).rank(pct=True).to_numpy();v=pd.Series(pr[:,1]).rank(pct=True).to_numpy();ok=(gy<nrow)&(gx<ncol)
  RGB=np.ones((nrow,ncol,3));RGB[gy[ok],gx[ok]]=bivar(u[ok],v[ok])
  b=A(DX0+IW+1,y0,IW,IH);b.imshow(RGB,interpolation='nearest',extent=[0,ncol,nrow,0]);b.set_xlim(0,ncol);b.set_ylim(nrow,0);b.set_axis_off()
  scalebar(b,2000/128,'2 mm',fs=FS,lw=.8,out=True)  # 网格单元 = 名义 128 µm patch
  C.text(DX0,y0+IH+2.4,r.patient_id.replace('TCGA-',''),fontsize=FS,va='baseline');C.text(DX0+IW,y0+IH+2.4,f'ρ = {getattr(r,"E2F__IFNg__coloc"):.2f}'.replace('-','−'),fontsize=FS,va='baseline',ha='right')
  crows.append(dict(group=grp,rank=r.rank,sampling_stripe_index=r.stripe,value_stripe_index=r.vstripe,patient_id=r.patient_id,slide=r.slide,coloc=getattr(r,'E2F__IFNg__coloc'),patches=r.n_patches))
# 双变量色键：8 mm 方块 + 四色样块
KQ=A(118.5,R2+2.5,8,8);uu,vv=np.meshgrid(np.linspace(0,1,64),np.linspace(0,1,64));KQ.imshow(bivar(uu,vv),origin='lower',extent=[0,1,0,1]);KQ.set_xticks([]);KQ.set_yticks([])
for s_ in KQ.spines.values():s_.set_visible(False)
KQ.text(.5,-.08,'E2F →',transform=KQ.transAxes,ha='center',va='top',fontsize=FS);KQ.text(-.08,.5,'IFNγ →',transform=KQ.transAxes,ha='right',va='center',rotation=90,fontsize=FS)
for i,(t_,c_) in enumerate([('both high',BV[1,1]),('E2F only',BV[1,0]),('IFNγ only',BV[0,1]),('both low',BV[0,0])]):
 yk=R2+17+i*2.9;sw=A(116,yk-.8,1.6,1.6);sw.add_patch(plt.Rectangle((0,0),1,1,color=c_));sw.set_axis_off();C.text(118.4,yk,t_,fontsize=FS,va='center')

# ---------- e TIL ----------
head(132,R2-6,'e','Co-localization and TILs')
tl=pd.read_csv(RS/'t8_1b/coloc_vs_TIL_density_patients.csv').dropna(subset=['til_percentage'])
a=A(143,R2,35,38);a.scatter(tl.til_percentage+.1,tl[F],s=1.4,color='#9a9a9a',alpha=.5,lw=0,rasterized=True);a.set_xscale('log');rt=corr(tl.til_percentage,tl[F])
# 分箱中位数线（TIL 十分位，描述性）
bn=pd.qcut(tl.til_percentage.rank(method='first'),10,labels=False);bm=tl.groupby(bn).agg(x=('til_percentage','median'),y=(F,'median'))
a.plot(bm.x+.1,bm.y,color=DARK,lw=1,marker='o',ms=1.8,mew=0)
a.set_xlabel('TIL density (% patches)');a.set_ylabel('Co-localization');tk(a);a.set_yticks([-.5,0,.5]);a.text(.03,.97,f'ρ = {rt:.2f}, n = {len(tl)}',transform=a.transAxes,fontsize=FS,va='top')
pct=pd.read_csv(RS/'t8_1b/coloc_tertiles_by_TIL_pattern_percent.csv').set_index('coloc_group').loc[['Low','Middle','High']];cnt=pd.read_csv(RS/'t8_1b/coloc_tertiles_by_TIL_pattern_counts.csv').set_index('coloc_group').loc[['Low','Middle','High']]
a=A(143,R2+56,30,13);left=np.zeros(3);PC={'brisk diffuse':'#0072B2','band-like':'#56B4E9','non-brisk':'#C9C9C9','none':'#EFEFEF'}
for c in pct.columns:a.barh(range(3),pct[c],left=left,color=PC[c],edgecolor='white',lw=.4,label=c,height=.72);left+=pct[c].to_numpy()
for i,g in enumerate(pct.index):a.text(102,i,f'{int(cnt.loc[g].sum())}',va='center',fontsize=FS)
a.text(102,2.8,'n',va='bottom',fontsize=FS,color=MUTED,style='italic')
a.set_yticks(range(3));a.set_yticklabels(list(pct.index));a.set_ylabel('Tertile');a.set_xlim(0,100);a.set_xticks([0,50,100]);a.set_xlabel('% patients by TIL pattern');tk(a);a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False)
a.legend(frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(-.3,1.06),handlelength=.8,handleheight=.8,handletextpad=.3,columnspacing=.8,borderaxespad=0,labelspacing=.25)
C.save('Fig5')
write_source('Fig5',{'a_Cox_single_feature':cx,'b_robustness':rb,'b_notes':pd.DataFrame([NOTE]),'c_KM_tertiles':pd.DataFrame(kmrows),'c_KM_patients':km,'d_cases':pd.DataFrame(crows),'e_TIL_density':tl[['patient_id','slide',F,'til_percentage','til_pattern4']],'e_TIL_pattern_counts':cnt.reset_index()})
print('FIG5 DONE',rob.round(3).to_dict('records'),'TIL rho',round(rt,3),'KM logrank',round(lrt_chi,2),lrt_p,NOTE,flush=True)
