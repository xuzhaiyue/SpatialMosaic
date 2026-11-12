"""T10 Fig 2：H&E 在 spot 上能读出什么。
a 示例留出切片（实测 vs 主模型预测）；b 三个留出 ST 集的患者内 ρ；c 每程序预测最高 / 最低的 128 µm 图块（形态标签留空）；d 58 程序图谱。
HER2ST 一律只用 Patient A–D。"""
from t10_common import *
import openslide
from PIL import Image
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe
EX='SPA147'  # T5 按"实测 Moran's I 最大"规则选出的 HER2ST 示例（Patient B，非重复患者），不按预测好坏挑选
sp=pd.read_parquet(D/'st_spot_predictions.parquet');sec=clean(pd.read_csv(D/'st_section_rho.csv'));meta=manifest().set_index('sample')
held=sec[sec.split.isin(['test','external'])]
pat=held.groupby(['cohort','patient','program'],as_index=False).agg(rho_main=('rho_main','mean'),rho_T4=('rho_T4','mean'),sections=('sample','size'))
# v3（2026-11-11）："顶刊"版式：左对齐面板标题，去掉居中粗体标题 + 灰色副标题；队列名移到轴下；H&E 与图块加比例尺；d 用两级列标题、四个主程序以颜色标出。
# v2.3 存 archive_Fig2_20261111c/，v2.2 存 archive_Fig2_20261111b/，v1 存 archive_Fig2_20261111/
C=Canvas(180,170);A=C.ax

# ---------- a 示例切片 ----------
C.head(2,5,'a','Measured and predicted maps on an external section (HER2ST, patient B)')
g=sp[sp['sample'].eq(EX)];row=meta.loc[EX];sl=openslide.OpenSlide(row.image);pad=round(128/row.mpp)
x0,x1,y0,y1=g.x.min()-pad,g.x.max()+pad,g.y.min()-pad,g.y.max()+pad;lv=sl.get_best_level_for_downsample(max(x1-x0,y1-y0)/700);ds=sl.level_downsamples[lv]
img=sl.read_region((int(x0),int(y0)),lv,(int((x1-x0)/ds),int((y1-y0)/ds)));img=Image.alpha_composite(Image.new('RGBA',img.size,(255,255,255,255)),img).convert('RGB');sl.close()
ax=A(12,11,25,31);ax.imshow(img,extent=[x0,x1,y1,y0]);ax.set_xlim(x0,x1);ax.set_ylim(y1,y0);ax.set_aspect('equal');ax.set_axis_off()
ax.text(.5,1.02,'H&E',transform=ax.transAxes,ha='center',va='bottom',fontsize=6,fontweight='bold');scalebar(ax,1000/row.mpp,'1 mm',out=True)
rows_a=[];MX=[42.5+j*17.3 for j in range(4)];MW=16.6
for j,k in enumerate(SK):
 obs=g[f'obs_{k}'].rank(pct=True);pre=g[f'main_{k}'].rank(pct=True);r=corr(g[f'obs_{k}'],g[f'main_{k}']);rows_a.append(dict(sample=EX,program=P[j],rho_main=r,spots=len(g)))
 for i,(v,lab) in enumerate([(obs,'Measured'),(pre,'Predicted')]):
  a=A(MX[j],11+i*16.6,MW,15.8);a.scatter(g.x,g.y,c=v,cmap='viridis',norm=Normalize(0,1),s=2.6,marker='o',lw=0);a.set_xlim(x0,x1);a.set_ylim(y1,y0);a.set_aspect('equal');a.set_axis_off()
  if i==0:a.text(.5,1.02,S[j],transform=a.transAxes,ha='center',va='bottom',fontsize=6,color=PCOL[S[j]],fontweight='bold')
  else:a.text(.5,-.03,f'ρ = {r:.2f}',transform=a.transAxes,ha='center',va='top',fontsize=5.5)
  if j==0:a.text(-.03,.5,lab,transform=a.transAxes,rotation=90,ha='right',va='center',fontsize=5.5)
assert abs(rows_a[0]['rho_main']-sec[(sec['sample']==EX)&(sec.program==P[0])].rho_main.iloc[0])<1e-12
cax=A(16,47,17,1.1);cb=C.fig.colorbar(plt.cm.ScalarMappable(Normalize(0,1),'viridis'),cax=cax,orientation='horizontal');cb.set_ticks([]);cb.outline.set_visible(False)
cax.text(-.05,.5,'low',transform=cax.transAxes,ha='right',va='center',fontsize=5);cax.text(1.05,.5,'high',transform=cax.transAxes,ha='left',va='center',fontsize=5)
cax.text(.5,1.8,'within-section rank',transform=cax.transAxes,ha='center',va='bottom',fontsize=5)

# ---------- b 患者内 ρ（配对画法：每名患者 / 切片从参照连到主模型；虚线 = 噪声上限中位数，t10_ceiling.py） ----------
C.head(2,56,'b','Within-patient agreement on held-out and external sets')
ceil=pd.read_csv(D/'ceiling_summary_ST.csv');ceil=ceil[ceil.support.eq('native')];hc=pd.read_csv(D/'ceiling_by_section.csv');hc=hc[hc.set.eq('HEST')]
brows=[];BT=[('He2020','He2020 test','breast · 6 patients'),('Meylan_ccRCC','Meylan test','kidney · 6 patients'),('HER2ST','HER2ST A–D','breast · 4 patients'),('HEST','HEST 10x','breast · 6 sections')]
BX=[12,36.6,64.4,89];BW=23  # 组内间距 1.6 mm，组间 4.8 mm
BY,BH=64,25.5;DX=.19
def pair(a,j,ref,main,col,ref_open=False):
 # 参照（左）→ 主模型（右），细线连接同一患者 / 切片；短横 = 中位数
 for r_,m_ in zip(ref,main):a.plot([j-DX,j+DX],[r_,m_],color='#cfcfcf',lw=.4,zorder=2)
 if ref_open:a.scatter([j-DX]*len(ref),ref,s=5,facecolor='white',edgecolor=col,lw=.5,zorder=3)
 else:a.scatter([j-DX]*len(ref),ref,s=5,color=GREY,lw=0,zorder=3)
 a.scatter([j+DX]*len(main),main,s=6,color=col,lw=0,zorder=3)
 a.plot([j-DX-.12,j-DX+.12],[np.median(ref)]*2,color='#7a7a7a' if not ref_open else col,lw=1.1,zorder=4)
 a.plot([j+DX-.12,j+DX+.12],[np.median(main)]*2,color=col,lw=1.4,zorder=4)
for k,(co,t1,t2) in enumerate(BT):
 a=A(BX[k],BY,BW,BH)
 for j,p in enumerate(P):
  if co!='HEST':
   d=pat[(pat.cohort==co)&(pat.program==p)];pair(a,j,d.rho_T4.to_numpy(),d.rho_main.to_numpy(),PCOL[S[j]]);c_=ceil[(ceil.set==co)&(ceil.program==p)].ceiling.iloc[0]
  else:
   # 原始 spot（空心）与测量单元匹配后（实心：Visium spot+6 邻居合并；Visium HD 已为 128 µm 方格，即原始值）
   gg=hc[hc.program.eq(p)];nat=gg[gg.support.eq('native')].set_index('sample');agg=gg[gg.support.eq('spot+6 neighbours')].set_index('sample');mt=nat.rho.copy();mt.loc[agg.index]=agg.rho
   pair(a,j,nat.rho.to_numpy(),mt.reindex(nat.index).to_numpy(),PCOL[S[j]],ref_open=True);c_=nat.ceiling.median()
   brows+=[dict(set='HEST',sample=x,program=p,rho_native=nat.rho[x],rho_matched_support=mt[x],ceiling=nat.ceiling[x]) for x in nat.index]
  a.plot([j-.38,j+.38],[c_]*2,color='#4d4d4d',lw=.6,ls=(0,(2,1.2)),zorder=1)
 for yy in [.5,1]:a.axhline(yy,color='#eeeeee',lw=.4,zorder=0)
 a.axhline(0,color='#c8c8c8',lw=.5,zorder=0);a.set_xlim(-.55,3.55);a.set_ylim(-.2,1.02);a.set_yticks([0,.5,1]);a.set_xticks([]);ticks(a)
 a.spines['bottom'].set_visible(False)
 C.text(BX[k]+BW/2,BY+BH+1.2,t1,ha='center',va='top',fontsize=6,fontweight='bold');C.text(BX[k]+BW/2,BY+BH+3.9,t2,ha='center',va='top',fontsize=5,color=MUTED)
 if k==0:a.set_ylabel('Spearman ρ, measured vs predicted\n(within patient)',labelpad=1)
 else:a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0,labelleft=False)
for (x0_,x1_),gt in [((BX[0],BX[1]+BW),'Held-out test'),((BX[2],BX[3]+BW),'External')]:
 C.text((x0_+x1_)/2,60.6,gt,ha='center',va='bottom',fontsize=5.5,color=MUTED);C.line(x0_,61.2,x1_,61.2)
# 图例（两行，左对齐 x = 12）
lg=A(12,98.6,100,5.4);lg.set_xlim(0,100);lg.set_ylim(5.4,0);lg.set_axis_off()
for j,s_ in enumerate(S):lg.scatter([1+j*11.5],[1.2],s=6,color=PCOL[s_],lw=0);lg.text(2.4+j*11.5,1.2,s_,fontsize=5,va='center')
lg.plot([50,53],[1.2,1.2],color=INK,lw=1.4);lg.text(54.4,1.2,'median',fontsize=5,va='center')
lg.plot([66,69.5],[1.2,1.2],color='#4d4d4d',lw=.6,ls=(0,(2,1.2)));lg.text(70.9,1.2,'noise ceiling (median)',fontsize=5,va='center')
def leg_pair(x,y,open_,lab):
 lg.plot([x,x+3],[y,y],color='#cfcfcf',lw=.4);lg.scatter([x],[y],s=5,**(dict(facecolor='white',edgecolor=INK,lw=.5) if open_ else dict(color=GREY,lw=0)));lg.scatter([x+3],[y],s=6,color=INK,lw=0);lg.text(x+4.4,y,lab,fontsize=5,va='center')
leg_pair(1,4.4,False,'ST-only → dual-scale, per patient');leg_pair(50,4.4,True,'HEST: native spot → matched support')

# ---------- c 形态图库 ----------
C.head(2,109,'c','Held-out breast tiles with the highest and lowest predicted scores')
# 留出乳腺 spot（He2020 测试 + HER2ST A–D），按主模型预测排序；每程序最高 / 最低各 4 个，每名患者每端最多 2 个
hs=clean(sp.merge(meta[['cohort','patient','split','image','mpp']].reset_index(),on='sample'));hs=hs[hs.cohort.isin(['He2020','HER2ST'])&hs.split.isin(['test','external'])].copy()
# v2.1：排除无组织纹理的图块（玻片背景；128 µm 图块灰度标准差 < 0.02，scripts 内联计算见 results/t10/qc/heldout_spot_tissue_metrics.csv）。
# 留出 38,364 个 spot 中仅 5 个，去掉后切片内 ρ 变化 ≤ 0.013，故只用于画廊选例，主分析不变。
tq=pd.read_csv(O/'qc/heldout_spot_tissue_metrics.csv');hs=hs.merge(tq[['sample','spot_id','gray_sd']],on=['sample','spot_id'],how='left');assert hs.gray_sd.notna().all();hs=hs[hs.gray_sd>=.02].copy()
for k in SK:hs[f'obs_pct_{k}']=hs.groupby('sample')[f'obs_{k}'].rank(pct=True)
NT=4;picks=[]
for j,k in enumerate(SK):
 for end,asc in [('high',False),('low',True)]:
  cnt={};n=0
  for r in hs.sort_values(f'main_{k}',ascending=asc).itertuples():
   if cnt.get(r.patient,0)>=2:continue
   cnt[r.patient]=cnt.get(r.patient,0)+1;picks.append(dict(program=P[j],end=end,rank=n+1,sample=r.sample,patient=r.patient,spot_id=r.spot_id,x=r.x,y=r.y,predicted=getattr(r,f'main_{k}'),measured=getattr(r,f'obs_{k}'),measured_pct_in_section=getattr(r,f'obs_pct_{k}'),image=r.image,mpp=r.mpp));n+=1
   if n==NT:break
picks=pd.DataFrame(picks);slides={}
TW,TG,EG,TY=11.9,.3,2.6,115.4;EOFF=NT*(TW+TG)+EG
C.text(12,113.6,'Highest predicted',fontsize=5.5,va='baseline');C.text(12+EOFF,113.6,'Lowest predicted',fontsize=5.5,va='baseline')
C.text(12+EOFF+NT*(TW+TG)-TG,113.6,'inset: measured percentile in section',fontsize=5,color=MUTED,va='baseline',ha='right')
STROKE=[pe.withStroke(linewidth=.9,foreground='#1a1a1a',alpha=.75)]
for j,p in enumerate(P):
 for e,end in enumerate(['high','low']):
  for t,r in enumerate(picks[(picks.program==p)&(picks.end==end)].itertuples()):
   sl=slides.setdefault(r.image,openslide.OpenSlide(r.image));size=round(128/r.mpp);x,y=np.rint(np.array([r.x,r.y])-size/2).astype(int)
   im=sl.read_region((int(x),int(y)),0,(size,size));im=Image.alpha_composite(Image.new('RGBA',im.size,(255,255,255,255)),im).convert('RGB').resize((160,160),Image.BICUBIC)
   a=A(12+t*(TW+TG)+e*EOFF,TY+j*(TW+.8),TW,TW);a.imshow(im);a.set_xticks([]);a.set_yticks([])
   for s in a.spines.values():s.set_visible(True);s.set_color(PCOL[S[j]] if end=='high' else '#a6a6a6');s.set_linewidth(.8)
   a.text(.05,.05,f'{100*r.measured_pct_in_section:.0f}',transform=a.transAxes,fontsize=5,color='white',va='bottom',ha='left',path_effects=STROKE)
   if t==0 and e==0:a.text(-.08,.5,S[j],transform=a.transAxes,rotation=90,ha='right',va='center',fontsize=6,color=PCOL[S[j]],fontweight='bold')
   if j==len(P)-1 and e==1 and t==NT-1:scalebar(a,160*50/128,'50 µm',color='white',lw=1)
for s in slides.values():s.close()

# ---------- d 58 程序图谱 ----------
C.head(116,5,'d','Atlas of 58 programmes')
at=pd.read_csv(R/'results/t9/atlas/atlas_summary.csv').set_index('program');ss=clean(pd.read_csv(R/'results/t9/atlas/atlas/ST_section_spearman.csv'))
h2=ss[ss.cohort.eq('HER2ST')].groupby(['patient_key','program']).spearman.mean().groupby('program').median();at['HER2ST (ST)']=h2.reindex(at.index)
assert ss[ss.cohort.eq('HER2ST')].patient_key.nunique()==4
COLS=[('He2020 test (ST)','He2020'),('HER2ST (ST)','HER2ST'),('TCGA-BRCA (bulk)','TCGA'),('CPTAC-BRCA (bulk)','CPTAC'),('Meylan ccRCC test (ST)','Meylan'),('TCGA-KIRC (bulk)','TCGA'),('CPTAC-CCRCC (bulk)','CPTAC')]
at['breast_ST_mean']=at[['He2020 test (ST)','HER2ST (ST)']].mean(1);at=at.sort_values('breast_ST_mean',ascending=False)
HX,HY,HW,HH=148,22.5,28,127;CW=HW/len(COLS)
a=A(HX,HY,HW,HH);M=at[[c for c,_ in COLS]].to_numpy(float);im=a.imshow(M,cmap='RdBu_r',vmin=-.7,vmax=.7,aspect='auto',interpolation='nearest')
a.set_xticks(range(len(COLS)));a.set_xticklabels([l for _,l in COLS],fontsize=5,rotation=90,ha='center',va='bottom');a.xaxis.tick_top();a.tick_params(axis='x',length=0,pad=1.5)
# 两级列标题：癌种 / 测量尺度
for (c0,c1),t in [((0,3),'Breast'),((4,6),'Kidney')]:
 C.text(HX+(c0+c1+1)/2*CW,8.6,t,ha='center',va='baseline',fontsize=5.5,fontweight='bold');C.line(HX+c0*CW+.4,9.6,HX+(c1+1)*CW-.4,9.6)
for (c0,c1),t in [((0,1),'ST'),((2,3),'bulk'),((4,4),'ST'),((5,6),'bulk')]:C.text(HX+(c0+c1+1)/2*CW,12.4,t,ha='center',va='baseline',fontsize=5,color=MUTED)
NAMEFIX=[('Myc','MYC'),('Tnfa','TNFα'),('Nfkb','NF-κB'),('Il2','IL2'),('Il6','IL6'),('Jak','JAK'),('Stat3','STAT3'),('Stat5','STAT5'),('Kras','KRAS'),('Dna','DNA'),('Uv','UV'),('Mtorc1','mTORC1'),('Pi3K','PI3K'),('Akt','AKT'),('Mtor','mTOR'),('Tgf','TGF'),('Wnt','WNT'),('Il','IL'),('Cd8','CD8'),('Nk','NK')]
def nice(p):
 t=p.replace('CELL_','Cell: ').replace('_',' ').title()
 for a_,b_ in NAMEFIX:t=t.replace(a_,b_)
 return t
lab=[nice(p)+(' †' if n<4 else '') for p,n in zip(at.index,at.members)]
a.set_yticks(range(len(at)));a.set_yticklabels(lab,fontsize=5);a.tick_params(axis='y',length=0,pad=1)
for t_,p in zip(a.get_yticklabels(),at.index):
 if p in P:t_.set_color(PCOL[S[P.index(p)]]);t_.set_fontweight('bold')
for k in [1.5,3.5,4.5]:a.axvline(k,color='white',lw=1.2 if k==3.5 else .6)
for s in a.spines.values():s.set_visible(False)
cax=A(151,156.5,22,1.1);cb=C.fig.colorbar(im,cax=cax,orientation='horizontal');cb.set_ticks([-.6,0,.6]);cb.set_ticklabels(['−0.6','0','0.6']);cb.ax.tick_params(labelsize=5,length=1.5,width=.5,pad=1);cb.outline.set_visible(False)
cax.text(.5,1.8,'Spearman ρ',transform=cax.transAxes,ha='center',va='bottom',fontsize=5)
C.text(116,163.5,'ST: median within-patient ρ; bulk: across patients\n† fewer than 4 marker genes in background',fontsize=5,color=MUTED,va='top',linespacing=1.4)
C.save('Fig2')
write_source('Fig2',{'a_example_section':pd.DataFrame(rows_a),'a_example_spots':g[['sample','spot_id','x','y']+[f'obs_{k}' for k in SK]+[f'main_{k}' for k in SK]],
 'b_patient_rho':pat,'b_HEST':pd.DataFrame(brows),'b_ceiling':ceil,'b_section_rho':held,'c_gallery_tiles':picks.drop(columns=['image']),'d_atlas':at.reset_index()[['program','members']+[c for c,_ in COLS]]})
print('FIG2 DONE',pat.groupby(['cohort','program']).rho_main.median().round(3).to_dict(),flush=True)
