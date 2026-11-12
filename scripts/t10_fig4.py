"""T10 Fig 4：虚拟分子地图与可读的空间结构。
a TCGA-BRCA / KIRC 全片四程序图（T7 交叉拟合图，即临床分析所用的同一批图）+ 真实 H&E 缩略图；
b 留出 ST 上空间量是否可读：b1 整片 E2F–IFNγ 共定位实测 vs 主模型；b2 Moran's I 实测 vs 主模型（留出患者）；
c 调整四程序均值后空间特征 × PAM50（T7 表，72 项）。HER2ST 只用 A–D。"""
from t10_common import *
import h5py
from PIL import Image
from matplotlib.colors import Normalize,TwoSlopeNorm
PP=E/'t7/patch_predictions'
cov=pd.read_csv(R/'results/t7/official_patch_tissue_coverage.csv');thumb=pd.read_csv(R/'results/t7/HE_thumbnail_index.csv').set_index('slide')
# 选例规则（预设、不看图）：T7 固定随机覆盖核查集中，每癌种 patch ≥ 5000、有缩略图、覆盖率最高的 2 张
cand=cov[cov.purposes.str.contains('random_coverage20')&(cov.patches>=5000)&cov.slide.isin(thumb.index)]
cases=pd.concat([g.sort_values('coverage',ascending=False).head(2) for _,g in cand.groupby('cohort')]).sort_values(['cohort','coverage'],ascending=[True,False])
# v3（2026-11-12）：与 Fig 5 v5 同一规则——字号三档（8 / 7 / 6 pt），标题左对齐，2 mm 比例尺。v2 存 results/t10/archive_Fig134_20261112/
C=Canvas(180,170);A=C.ax;fig=C.fig;FS,TT=6,7
def H(x,y,l,t):C.head(x,y,l,t,fs=TT)
def tk(a):ticks(a,FS,FS)

# ---------- a 全片图 ----------
arows=[];CX=[19+k*30.5 for k in range(5)];CW=29.5;RH=20
H(2,5,'a','Whole-slide programme maps, TCGA (cross-fitted)')
for k,t_ in enumerate(['H&E']+S):C.text(CX[k]+CW/2,11,t_,ha='center',va='baseline',fontsize=FS,fontweight='bold',color=INK if k==0 else PCOL[t_])
for c,r in enumerate(cases.itertuples()):
 with h5py.File(PP/r.cohort/(r.slide+'.h5')) as f:xy=f['coords_top_left'][:];pr=f['predicted'][:];cand_=f.attrs['candidate'];fold=int(f.attrs['fold']);step=int(f.attrs['grid_step'])
 t=thumb.loc[r.slide];im=Image.open(t.png).convert('RGB');sc=t.scale_to_level0;W,Hh=im.size;y0=13+c*(RH+1.2)
 gx=(xy[:,0]//step).astype(int);gy=(xy[:,1]//step).astype(int);ncol=int(np.ceil(W*sc/step));nrow=int(np.ceil(Hh*sc/step))
 a=A(CX[0],y0,CW,RH);a.imshow(im,extent=[0,ncol,nrow,0]);a.set_axis_off();scalebar(a,2000/128,'2 mm',fs=FS,lw=1)
 C.text(2,y0+RH/2-.6,r.patient_id.replace('TCGA-',''),va='baseline',fontsize=FS,fontweight='bold');C.text(2,y0+RH/2+2.6,r.cohort,va='baseline',fontsize=FS,color=MUTED)
 for j in range(4):
  M=np.full((nrow,ncol),np.nan);v=pd.Series(pr[:,j]).rank(pct=True).to_numpy();ok=(gy<nrow)&(gx<ncol);M[gy[ok],gx[ok]]=v[ok]
  b=A(CX[j+1],y0,CW,RH);b.imshow(M,cmap='viridis',vmin=0,vmax=1,interpolation='nearest',extent=[0,ncol,nrow,0]);b.set_axis_off()
 arows.append(dict(slide=r.slide,patient_id=r.patient_id,cohort=r.cohort,patches=len(pr),official_tissue_coverage=r.coverage,map_candidate=cand_,cross_fit_fold=fold))
cax=A(173,16,1.4,16);cb=fig.colorbar(plt.cm.ScalarMappable(Normalize(0,1),'viridis'),cax=cax);cb.set_ticks([0,1]);cb.set_ticklabels(['low','high']);cb.ax.tick_params(labelsize=FS,length=0,pad=1);cb.outline.set_visible(False)
C.text(173.7,14.5,'rank',ha='center',va='baseline',fontsize=FS)

# ---------- b 留出 ST 上的空间量 ----------
BY,BS=112,34
H(2,104,'b','Spatial readouts on held-out ST')
co=clean(pd.read_csv(D/'st_coloc_by_section.csv'));co=co[co.split.isin(['test','external'])&co.program_A.eq(P[0])&co.program_B.eq(P[1])]
a=A(14,BY,BS,BS);br=co[co.cohort.isin(['He2020','HER2ST'])];kd=co[co.cohort.eq('Meylan_ccRCC')]
for cohort,mk in [('He2020','o'),('HER2ST','s')]:
 d=br[br.cohort.eq(cohort)];a.scatter(d.observed_coloc,d.main_coloc,s=7,marker=mk,color='#555555',alpha=.85,lw=0)
a.scatter(kd.observed_coloc,kd.main_coloc,s=9,marker='^',facecolor='white',edgecolor='#555555',lw=.6)
rb=corr(br.observed_coloc,br.main_coloc);lim=[-.6,1];a.plot(lim,lim,ls=(0,(2,2)),color='#b0b0b0',lw=.5);a.set_xlim(lim);a.set_ylim(lim);a.set_xticks([-.5,0,.5,1]);a.set_yticks([-.5,0,.5,1])
a.set_xlabel('Measured');a.set_ylabel('Predicted');tk(a);a.text(.04,.97,f'breast ρ = {rb:.2f}',transform=a.transAxes,va='top',fontsize=FS)
C.text(14,109.5,'E2F–IFNγ co-localization',fontsize=FS,fontweight='bold',va='baseline')
lg=A(14,155,40,11);lg.set_xlim(0,40);lg.set_ylim(11,0);lg.set_axis_off()
for i,(mk,kw,lab) in enumerate([('o',dict(color='#555555',lw=0),f'He2020 test ({int(br.cohort.eq("He2020").sum())} sections)'),('s',dict(color='#555555',lw=0),f'HER2ST A–D ({int(br.cohort.eq("HER2ST").sum())})'),('^',dict(facecolor='white',edgecolor='#555555',lw=.6),f'Meylan test, kidney ({len(kd)})')]):
 lg.scatter([1],[1.5+i*3.3],s=8,marker=mk,**kw);lg.text(2.6,1.5+i*3.3,lab,fontsize=FS,va='center')
mo=clean(pd.read_csv(D/'st_moran_by_section.csv'));mo=mo[mo.split.isin(['test','external'])];mp=mo.groupby(['cohort','patient','program'],as_index=False)[['observed_moran','main_moran']].mean()
a=A(64,BY,BS,BS);mrows=[]
for j,p in enumerate(P):
 for ca,cohorts,fc in [('breast',['He2020','HER2ST'],PCOL[S[j]]),('kidney',['Meylan_ccRCC'],'white')]:
  d=mp[mp.program.eq(p)&mp.cohort.isin(cohorts)];a.scatter(d.observed_moran,d.main_moran,s=8,facecolor=fc,edgecolor=PCOL[S[j]],lw=.6,zorder=3);rr=corr(d.observed_moran,d.main_moran);mrows.append(dict(program=p,cancer=ca,patients=len(d),spearman=rr))
a.plot([-.1,1],[-.1,1],ls=(0,(2,2)),color='#b0b0b0',lw=.5);a.set_xlim(-.05,.9);a.set_ylim(-.05,.9);a.set_xticks([0,.4,.8]);a.set_yticks([0,.4,.8]);a.set_xlabel('Measured');a.set_ylabel('Predicted');tk(a)
C.text(64,109.5,"Moran's I, per patient",fontsize=FS,fontweight='bold',va='baseline')
mr=pd.DataFrame(mrows).set_index(['program','cancer']).spearman
lg=A(64,154.5,44,13);lg.set_xlim(0,44);lg.set_ylim(13,0);lg.set_axis_off()
lg.text(18,.8,'ρ breast',fontsize=FS,va='center',ha='right',color=MUTED);lg.text(26,.8,'kidney',fontsize=FS,va='center',ha='right',color=MUTED)
for j,p in enumerate(P):
 y=3.5+j*2.9;lg.scatter([1],[y],s=7,color=PCOL[S[j]],lw=0);lg.text(2.6,y,S[j],fontsize=FS,va='center');lg.text(18,y,f'{mr[(p,"breast")]:.2f}',fontsize=FS,va='center',ha='right');lg.text(26,y,f'{mr[(p,"kidney")]:.2f}',fontsize=FS,va='center',ha='right')
lg.scatter([29.5],[3.5],s=8,color=MUTED,lw=0);lg.text(31.1,3.5,'breast (10)',fontsize=FS,va='center');lg.scatter([29.5],[6.4],s=8,facecolor='white',edgecolor=MUTED,lw=.6);lg.text(31.1,6.4,'kidney (6)',fontsize=FS,va='center')

# ---------- c PAM50 ----------
pm=pd.read_csv(R/'results/t7/PAM50_single_feature_associations.csv');SUB=['Basal','Her2','LumA','LumB'];feats=list(dict.fromkeys(pm.feature))
LG=np.full((len(feats),4),np.nan);Q=np.full((len(feats),4),np.nan)
for i,f in enumerate(feats):
 for k,su in enumerate(SUB):
  z=pm[(pm.feature==f)&(pm.subtype==su)]
  if len(z):LG[i,k]=np.log2(z.OR.iloc[0]);Q[i,k]=z.q.iloc[0]
H(112,104,'c',f'Spatial features × PAM50 (TCGA-BRCA, n = {int(pm.n.iloc[0])})')
a=A(136,BY,26,46);lim=np.nanmax(np.abs(LG));im=a.imshow(LG,cmap='PuOr_r',vmin=-lim,vmax=lim,aspect='auto')
for i in range(len(feats)):
 for k in range(4):
  if Q[i,k]<.05:a.text(k,i+.2,'*',ha='center',va='center',fontsize=FS,color=INK)
def flab(f):
 parts=f.split('__');kind=parts[-1];names=[{'IFNg':'IFNγ'}.get(x,x) for x in parts[:-1]]
 return {'moran':f"{names[0]} Moran's I",'high_fraction':f'{names[0]} high fraction','entropy':f'{names[0]} entropy','coloc':'–'.join(names)+' co-loc.'}[kind]
a.set_yticks(range(len(feats)));a.set_yticklabels([flab(f) for f in feats]);a.set_xticks(range(4));a.set_xticklabels(SUB);a.xaxis.tick_top();tk(a);a.tick_params(length=0,pad=1.5)
for k in [3.5,7.5,11.5]:a.axhline(k,color='white',lw=1)
for s_ in a.spines.values():s_.set_visible(False)
cax=A(164.5,124,1.4,30);cb=fig.colorbar(im,cax=cax);cb.set_ticks([-.5,0,.5]);cb.ax.tick_params(labelsize=FS,length=1.5,pad=1);cb.outline.set_visible(False);cb.set_label('log2 OR per s.d.',fontsize=FS,labelpad=2)
C.text(112,166,f'* BH q < 0.05 ({int((Q<.05).sum())} of {int(np.isfinite(Q).sum())} tests); adjusted for 4 programme means',fontsize=FS,color=MUTED,va='baseline')
C.save('Fig4')
write_source('Fig4',{'a_example_slides':pd.DataFrame(arows),'b1_coloc_sections':co,'b2_moran_patients':mp,'b2_moran_spearman':pd.DataFrame(mrows),'c_PAM50_associations':pm})
print('FIG4 DONE breast coloc rho',round(rb,3),pd.DataFrame(mrows).round(3).to_dict('records'),flush=True)
