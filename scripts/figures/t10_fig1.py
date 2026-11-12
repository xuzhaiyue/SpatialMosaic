"""T10 Fig 1 v2（2026-11-11）：研究设计与计算流程总览，版式与图形摘要 v3 统一（浅色底带、无黑框、字号四档、小写面板字母）。
所有数字从结果表读取并写入 Fig1_source_data/numbers.csv；示意中的 H&E、spot、程序图均为本项目真实图像。
相对 v1 的更正：损失写作 λ·L_ST + (1−λ)·L_bulk（λ = 1 为纯 ST，与 Methods 一致；v1 写反）；空间特征改为实际的四类
（Moran's I、高分比例、局部熵、共定位；均值只作校正协变量，不在 18 个特征内）；programme 拼写与正文一致。v1 存于 results/t10/archive_Fig1_20261111/。"""
from t10_common import *
import openslide,h5py
from PIL import Image
from matplotlib.patches import FancyBboxPatch,Circle,Polygon,Rectangle
RS=R/'results'
# ---------- 数字 ----------
man=manifest();clean_man=clean(man)
def st_counts(co):
 m=clean_man[clean_man.cohort.eq(co)];return {s:int(m[m.split.eq(s)].patient_key.nunique()) for s in ['train','val','test','external']}|{'sections':len(m)}
he,me,h2=st_counts('He2020'),st_counts('Meylan_ccRCC'),st_counts('HER2ST')
w=pd.read_csv(RS/'t5/wsi_prediction_manifest.csv');wc=w.groupby('cohort').agg(slides=('slide','nunique'),patients=('patient_id','nunique'),patches=('patches','sum'))
bk=pd.read_csv(RS/'t7/bulk_correlations_all_candidates.csv');bn=bk[bk.candidate=='lambda_0.5'].groupby('cohort').n.first()
cx=pd.read_csv(RS/'t7/Cox_single_feature_associations.csv').groupby('cohort')[['n','events']].first()
pe=pd.read_csv(RS/'t9/pam50/PAM50_external_CPTAC.csv').set_index('analysis').loc['primary_genefu_labels'];til=pd.read_csv(RS/'t8_1b/coloc_vs_TIL_density_patients.csv').til_percentage.notna().sum()
N=[('He2020 patients train/val/test','/'.join(str(he[s]) for s in ['train','val','test']),'results/t4/input_manifest.csv'),('He2020 sections',he['sections'],'results/t4/input_manifest.csv'),
 ('Meylan patients train/val/test','/'.join(str(me[s]) for s in ['train','val','test']),'results/t4/input_manifest.csv'),('HER2ST A–D patients',h2['external'],'results/t4/input_manifest.csv minus results/t10/qc duplicates'),('HER2ST A–D sections',h2['sections'],'same'),
 ('TCGA-BRCA bulk patients',int(bn['TCGA-BRCA']),'results/t7/bulk_correlations_all_candidates.csv'),('TCGA-KIRC bulk patients',int(bn['TCGA-KIRC']),'same'),('CPTAC-BRCA bulk patients',int(bn['CPTAC-BRCA']),'same'),('CPTAC-CCRCC bulk patients',int(bn['CPTAC-CCRCC']),'same')]
for co in wc.index:N+= [(f'{co} WSI patients',int(wc.loc[co,'patients']),'results/t5/wsi_prediction_manifest.csv'),(f'{co} WSI slides',int(wc.loc[co,'slides']),'same')]
N+=[('total slides',int(wc.slides.sum()),'same'),('total patches',int(wc.patches.sum()),'same'),('TCGA-BRCA PFI n/events',f"{cx.loc['TCGA-BRCA','n']}/{cx.loc['TCGA-BRCA','events']}",'results/t7/Cox_single_feature_associations.csv'),
 ('TCGA-KIRC PFI n/events',f"{cx.loc['TCGA-KIRC','n']}/{cx.loc['TCGA-KIRC','events']}",'same'),('TCGA-BRCA PAM50 training n',int(pe.train_n),'results/t9/pam50/PAM50_external_CPTAC.csv'),('CPTAC-BRCA PAM50 n',int(pe.CPTAC_n),'same'),('Saltz TIL matched n',int(til),'results/t8_1b/coloc_vs_TIL_density_patients.csv')]
num=dict((k,v) for k,v,_ in N)
def pfi(co):n,e=num[co+' PFI n/events'].split('/');return f'{int(n):,} pts, {int(e)} events'

# ---------- 画布（mm）与样式：同图形摘要 v3 ----------
Wd,Ht=180,170;fig=plt.figure(figsize=(Wd*MM,Ht*MM));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,Wd);ax.set_ylim(Ht,0);ax.set_axis_off()
ST_C,BK_C,MD_C,HL='#3d6b33','#8a5a00','#2C5D9A','#a5481b'
# v3（2026-11-12）：字号统一三档——面板字母 8 pt 粗体、面板标题 7 pt、其余一律 6 pt（β 符号按 7 pt）。v2 存 results/t10/archive_Fig134_20261112/
def txt(x,y,s,size=6,**k):size=8 if size>=8 and k.get('fontweight')=='bold' and len(s)==1 else (7 if size>=7 else 6);return ax.text(x,y,s,fontsize=size,ha=k.pop('ha','center'),va=k.pop('va','center'),zorder=k.pop('zorder',8),**k)
def arrow(x0,y0,x1,y1,c='#444444',lw=.9):ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>,head_length=.4,head_width=.22',color=c,lw=lw,shrinkA=0,shrinkB=0),zorder=7)
def img(a,x,y,w,h=None,z=4,frame=None):
 a=np.asarray(a);h=h if h is not None else w*a.shape[0]/a.shape[1];ax.imshow(a,extent=[x,x+w,y+h,y],zorder=z,interpolation='bilinear')
 if frame:ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=frame,lw=.5,zorder=z+1))
 return h
def fit(a,x,y,w,h,**k):
 a=np.asarray(a);r=a.shape[0]/a.shape[1];ww,hh=(w,w*r) if w*r<=h else (h/r,h);img(a,x+(w-ww)/2,y+(h-hh)/2,ww,hh,**k);return x+(w-ww)/2,y+(h-hh)/2,ww,hh
def band(y,h,letter,title,fc):
 ax.add_patch(FancyBboxPatch((2,y),Wd-4,h,boxstyle='round,pad=0,rounding_size=2',fc=fc,ec='none',zorder=0))
 txt(4.5,y+4,letter,8,ha='left',fontweight='bold',color=INK);txt(10,y+4,title,7,ha='left',color=INK)
 ax.plot([41,41],[y+9,y+h-3],color='#d8d8d8',lw=.5,zorder=1)
def cohorts(y,groups):
 # 左栏：队列列表（纯文字，无框）
 for head,lines in groups:
  txt(6,y,head,5.8,ha='left',fontweight='bold',color=MD_C);y+=3.6
  for l in lines:txt(6,y,l,5.2,ha='left');y+=3.2
  y+=1.6
def spot_img(df,col):
 f2=plt.figure(figsize=(3,3*asp),dpi=200);a2=f2.add_axes([0,0,1,1]);a2.imshow(hei,extent=[x0,x1,y1,y0],alpha=.3);a2.scatter(df.x,df.y,c=pd.Series(df[col]).rank(pct=True),cmap='viridis',s=46,lw=0)
 a2.set_xlim(x0,x1);a2.set_ylim(y1,y0);a2.set_axis_off();f2.canvas.draw();im=np.asarray(f2.canvas.buffer_rgba()).copy();plt.close(f2);return im
def gridmap(xy,v,step):
 gx=(xy[:,0]//step).astype(int);gy=(xy[:,1]//step).astype(int);gx-=gx.min();gy-=gy.min();M=np.zeros((gy.max()+1,gx.max()+1,4));M[gy,gx]=plt.cm.viridis(pd.Series(v).rank(pct=True).to_numpy());return M

# ---------- 素材 ----------
sp=pd.read_parquet(D/'st_spot_predictions.parquet');EX='SPA147';g=sp[sp['sample'].eq(EX)];row=man.set_index('sample').loc[EX];sl=openslide.OpenSlide(row.image)
pad=round(128/row.mpp);x0,x1,y0,y1=g.x.min()-pad,g.x.max()+pad,g.y.min()-pad,g.y.max()+pad;lv=sl.get_best_level_for_downsample(max(x1-x0,y1-y0)/600);ds=sl.level_downsamples[lv]
hei=sl.read_region((int(x0),int(y0)),lv,(int((x1-x0)/ds),int((y1-y0)/ds)));hei=Image.alpha_composite(Image.new('RGBA',hei.size,(255,255,255,255)),hei).convert('RGB');asp=(y1-y0)/(x1-x0)
c0=g.iloc[np.argmax(g.obs_E2F.to_numpy())]
tile=sl.read_region((int(c0.x-pad/2),int(c0.y-pad/2)),0,(pad,pad));tile=Image.alpha_composite(Image.new('RGBA',tile.size,(255,255,255,255)),tile).convert('RGB').resize((240,240));sl.close()
sp_obs,sp_pred=spot_img(g,'obs_E2F'),spot_img(g,'main_E2F')
TS='TCGA-AR-A24H-01Z-00-DX1.5CFC7E16-3F38-4531-968C-A4E4C9D00659';th=pd.read_csv(RS/'t7/HE_thumbnail_index.csv').set_index('slide').loc[TS];tim=Image.open(th.png).convert('RGB')
with h5py.File(E/'t7/patch_predictions/TCGA-BRCA'/(TS+'.h5')) as f:txy=f['coords_top_left'][:];tpr=f['predicted'][:];tstep=int(f.attrs['grid_step'])
tmaps=[gridmap(txy,tpr[:,j],tstep) for j in range(4)]

# ================= a：双尺度读出 =================
yA=2;hA=66;band(yA,hA,'a','Dual-scale molecular readout','#F6F7F9')
cohorts(yA+12,[('ST supervision',[f"He2020 breast  {he['train']}/{he['val']}/{he['test']}",f"Meylan ccRCC  {me['train']}/{me['val']}/{me['test']}",'patients, train/val/test']),
 ('External ST',[f"HER2ST A–D  {h2['external']} patients"]),('Bulk supervision',[f"TCGA-BRCA  {num['TCGA-BRCA bulk patients']:,} patients",f"TCGA-KIRC  {num['TCGA-KIRC bulk patients']:,} patients"])])
yS,yB=yA+21,yA+50
txt(45,yA+11.5,'Spot level · spatial transcriptomics',6.2,ha='left',fontweight='bold',color=ST_C)
txt(45,yA+38.5,'Slide level · bulk RNA-seq',6.2,ha='left',fontweight='bold',color=BK_C)
fit(hei,44,yS-8,20,16);txt(54,yS+9.6,'ST section',5,color=MUTED)
arrow(64.5,yS,67.5,yS);img(tile,68,yS-6,12,12,frame='#333333');txt(74,yS+9.6,'tile per spot',5,color=MUTED)
fit(tim,44,yB-8,20,16);txt(54,yB+9.6,'TCGA whole slide',5,color=MUTED)
arrow(64.5,yB,67.5,yB)
for i in range(5):
 for j in range(5):ax.add_patch(Rectangle((68+i*2.4,yB-6+j*2.4),2.2,2.2,fc='#e3cde3' if (i*3+j)%4 else '#c49ac4',ec='none',zorder=4))
txt(74,yB+9.6,'128 µm patches',5,color=MUTED)
# 共享编码器与读出（跨两行）
for yy in (yS,yB):arrow(80.5,yy,84,yy)
ax.add_patch(Polygon([[84.3,yS-7],[92,yS-3],[92,yB+3],[84.3,yB+7]],closed=True,fc='#9DB8DE',ec=MD_C,lw=.6,zorder=4));txt(88.2,(yS+yB)/2,'UNI2-h (frozen) · 1,536-d',5.4,rotation=90,color=INK)
for yy in (yS,yB):arrow(92.3,yy,96,yy)
ax.add_patch(FancyBboxPatch((96.3,yS-6),18,yB-yS+12,boxstyle='round,pad=0,rounding_size=1',fc='white',ec='#555555',lw=.6,zorder=4))
txt(105.3,yS-2,'shared linear',5.8,fontweight='bold',color=MD_C);txt(105.3,yS+1.2,'readout',5.8,fontweight='bold',color=MD_C);txt(105.3,(yS+yB)/2-1.5,'β',7,color=MD_C)
txt(105.3,(yS+yB)/2+6,'ridge',5,color=MUTED)
txt(105.3,yB+1.2,r'$\overline{f(x)}=f(\overline{x})$',6.2,color=INK);txt(105.3,yB+4.3,'linear identity',5,color=MUTED)
# 输出 ↔ 测量
for yy in (yS,yB):arrow(114.6,yy,118,yy)
fit(sp_pred,118.5,yS-8,13,16);txt(125,yS+9.6,'predicted',5,color=MUTED);txt(133.5,yS,'↔',7,color='#666666');fit(sp_obs,136,yS-8,13,16);txt(142.5,yS+9.6,'measured spots',5,color=MUTED)
ax.add_patch(Rectangle((119,yB-2),11,4,fc=plt.cm.viridis(.62),ec='none',zorder=4));txt(124.5,yB+5,'mean of patch\npredictions',5,color=MUTED)
txt(133.5,yB,'↔',7,color='#666666')
for i,hh in enumerate([3,6.5,4.5,9,5.5]):ax.add_patch(Rectangle((137+i*2.2,yB+3-hh),1.7,hh,fc=plt.cm.viridis(hh/10),ec='none',zorder=4))
txt(142.5,yB+5.8,'bulk RNA\nper patient',5,color=MUTED)
# 损失
arrow(150,yS,154.3,yA+30,c=ST_C);arrow(150,yB,154.3,yA+42.5,c=BK_C)
txt(155,yA+25,'$L_{ST}$',6,color=ST_C,ha='left');txt(155,yA+47.5,'$L_{bulk}$',6,color=BK_C,ha='left')
ax.add_patch(FancyBboxPatch((154.5,yA+28.5),23,15.5,boxstyle='round,pad=0,rounding_size=1',fc='white',ec='#555555',lw=.6,zorder=4))
txt(166,yA+32,r'$\lambda\,L_{ST}+(1-\lambda)\,L_{bulk}$',6.2,color=INK);txt(166,yA+35.6,'λ = 1: ST only',5,color=MUTED);txt(166,yA+38.4,'λ = 0: bulk only',5,color=MUTED);txt(166,yA+41.2,'chosen on validation',5,color=MUTED)
txt(166,yA+52,'4 Hallmark programmes',5.6,fontweight='bold')
for j,s in enumerate(S):ax.add_patch(Circle((156+(j%2)*12,yA+56.2+(j//2)*3.6),1,fc=PCOL[s],ec='none',zorder=5));txt(157.8+(j%2)*12,yA+56.2+(j//2)*3.6,s,5.2,ha='left')
txt(166,yA+63.4,'+ 58-programme atlas',5,color=MUTED)

# ================= b：全片图 =================
yB0=yA+hA+2;hB=52;band(yB0,hB,'b','Virtual molecular maps at scale','#FBF7EC')
b=lambda co:f"{num[co+' WSI patients']:,} patients · {num[co+' WSI slides']:,} slides"
cohorts(yB0+12,[('Discovery (cross-fitted)',['TCGA-BRCA',b('TCGA-BRCA'),'TCGA-KIRC',b('TCGA-KIRC')]),('External',['CPTAC-BRCA',b('CPTAC-BRCA'),'CPTAC-CCRCC',b('CPTAC-CCRCC')])])
y0b=yB0+11.5
txt(55,y0b,'Whole-slide H&E',6.2,fontweight='bold');fit(tim,44,y0b+4,22,30);arrow(66.5,y0b+19,70,y0b+19)
txt(88,y0b,'Programme maps',6.2,fontweight='bold')
for j in range(4):
 xx,yy=70.5+(j%2)*17.5,y0b+3+(j//2)*17;fit(tmaps[j],xx,yy,17,13);txt(xx+8.5,yy+14.4,S[j],5.4,color=PCOL[S[j]],fontweight='bold')
arrow(106.5,y0b+19,110,y0b+19)
txt(129,y0b,'18 spatial features per cancer',6.2,fontweight='bold')
F4=[("Moran's I",'clustering, ×4'),('High fraction','above 75th pct, ×4'),('Local entropy','mixing, ×4'),('Co-localization','programme pairs, ×6')];rng=np.random.default_rng(3)
for k,(n1,n2) in enumerate(F4):
 xx=111+(k%2)*19;yy=y0b+4+(k//2)*15;q=1.8
 for i in range(4):
  for j in range(4):
   if k==0:fc=PCOL['IFNγ'] if (i<2 and j<2) else '#e6e6e6'
   elif k==1:fc=PCOL['E2F'] if (i,j) in {(0,3),(2,1),(3,3),(1,0)} else '#e6e6e6'
   elif k==2:fc=[PCOL['E2F'],PCOL['Hypoxia'],PCOL['EMT'],PCOL['IFNγ']][rng.integers(4)]
   else:fc={0:'#3B1F6E',1:PCOL['E2F'],2:PCOL['IFNγ']}.get((i+j)%5,'#e6e6e6')
   ax.add_patch(Rectangle((xx+i*q,yy+j*q),q-.25,q-.25,fc=fc,ec='none',zorder=5))
 txt(xx+3.5,yy+9.3,n1,5.2,fontweight='bold');txt(xx+3.5,yy+11.9,n2,5,color=MUTED)
txt(129,y0b+37.5,'adjusted for programme means',5,color=MUTED)
txt(164,y0b,'Validation',6.2,fontweight='bold',color=HL)
for k,(t1,t2) in enumerate([('Slide mean vs bulk RNA','TCGA cross-fitted,\nCPTAC external'),('Spatial structure','held-out and\nexternal ST')]):
 yy=y0b+5+k*14;ax.add_patch(FancyBboxPatch((150.7,yy),26.6,11.5,boxstyle='round,pad=0,rounding_size=1',fc='white',ec='#d9b98a',lw=.6,zorder=4))
 txt(164,yy+3,t1,5.2,fontweight='bold');txt(164,yy+7.6,t2,5,color=MUTED)
txt(164,y0b+34.8,f"{num['total slides']:,} slides",5.2,color=MUTED);txt(164,y0b+37.8,f"{num['total patches']/1e6:.1f} M patches",5.2,color=MUTED)
ax.plot([149.5,149.5],[y0b-1.5,y0b+39],color='#e6dcc4',lw=.5,zorder=1)

# ================= c：生物学问题 =================
yC=yB0+hB+2;hC=Ht-yC-2;band(yC,hC,'c','Subtype, outcome and morphology','#FBF4EF')
cohorts(yC+12,[('TCGA',[f"BRCA PAM50  n = {num['TCGA-BRCA PAM50 training n']:,}",f"BRCA PFI  {pfi('TCGA-BRCA')}",f"KIRC PFI  {pfi('TCGA-KIRC')}"]),('External',[f"CPTAC-BRCA PAM50  n = {num['CPTAC-BRCA PAM50 n']}"]),('Reference',[f"Saltz TIL maps  n = {num['Saltz TIL matched n']}"])])
sub=[('Molecular subtype','programme means → PAM50,\nexternal CPTAC test;\nspatial features by subtype'),('Outcome','spatial feature → PFI (Cox),\nadjusted for age, stage\nand programme means'),('Morphology','high / low cases;\nTIL density and pattern')]
for k,(t1,t2) in enumerate(sub):
 xx=44+k*45;cx_,cy_=xx+20,yC+20
 txt(cx_,yC+10.5,t1,6.2,fontweight='bold',color=HL)
 if k==0:
  ax.add_patch(Rectangle((cx_-6,cy_-5),12,12,fill=False,ec='#999999',lw=.5,zorder=4));tt=np.linspace(0,1,40)
  for c_,pw in [('#4D4D4D',.25),(PCOL['IFNγ'],.5)]:ax.plot(cx_-6+12*tt,cy_+7-12*tt**pw,color=c_,lw=.9,zorder=5)
  ax.plot([cx_-6,cx_+6],[cy_+7,cy_-5],ls=':',color='#bbbbbb',lw=.6,zorder=5)
 if k==1:
  for i,(hr,lo,hi_) in enumerate([(.95,.8,1.1),(.74,.62,.9),(1.05,.9,1.25),(.9,.75,1.05)]):
   yy=cy_-3.5+i*3.3;X=lambda v:cx_+np.log(v)*20;c_=HL if i==1 else '#999999';ax.plot([X(lo),X(hi_)],[yy,yy],color=c_,lw=.8,zorder=5);ax.add_patch(Circle((X(hr),yy),.6,fc=c_,ec='none',zorder=6))
  ax.plot([cx_,cx_],[cy_-5.5,cy_+7],ls='--',color='#bbbbbb',lw=.6,zorder=4)
 if k==2:fit(tim,cx_-14,cy_-5,13.5,12);fit(tmaps[0],cx_+.5,cy_-5,13.5,12)
 txt(cx_,yC+35,t2,5.2)
 if k<2:ax.plot([xx+43,xx+43],[yC+9,yC+hC-3],color='#ecd9cc',lw=.5,zorder=1)

for ext in ['pdf','svg','png']:fig.savefig(O/f'Fig1.{ext}',dpi=300 if ext=='png' else None,facecolor='white')
plt.close(fig)
write_source('Fig1',{'numbers':pd.DataFrame(N,columns=['quantity','value','source'])})
print('FIG1 v2 DONE',flush=True)
