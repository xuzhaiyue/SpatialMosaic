"""T10 补充图（4 张 + 1 个附件）。
S1 数据与标签质控（队列、HER2ST 重复剔除、PAM50 标签一致性与官方标签外部 ROC）；S2 骨干对比；
S3 读出上限与 HEST 外部验证；S4 临床稳健性（设计审查、采样缺口）；SupplementaryData1 全部留出 ST 切片实测 vs 预测（多页 PDF）。
HER2ST 一律 A–D。用法：python t10_supp.py [S1 S2 S3 S4 SD1]（缺省全部）。"""
from t10_common import *
import sys,h5py,json
from PIL import Image
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
RS=R/'results';HEST=Path('/Volumes/INTENSO_BIG/HEST1k')
def pimg(ax,path,title=None):
 ax.imshow(Image.open(path).convert('RGB'));ax.set_axis_off()
 if title:ax.set_title(title,fontsize=6.5,loc='left')
def km(ax,t,e,label,color):
 o=np.argsort(t);t,e=t[o],e[o];s=1.;xs=[0];ys=[1.]
 for tt in np.unique(t[e==1]):
  n=(t>=tt).sum();d=((t==tt)&(e==1)).sum();s*=1-d/n;xs+=[tt,tt];ys+=[ys[-1],s]
 xs.append(t.max());ys.append(ys[-1]);ax.plot(np.array(xs)/365.25,ys,color=color,lw=1.1,label=label)

def table(ax,df,cols,widths,fs=5.6,scale=(1,1.2)):
 t=ax.table(cellText=df.astype(str).values.tolist(),colLabels=cols,loc='center',cellLoc='center',colWidths=widths);t.auto_set_font_size(False);t.set_fontsize(fs);t.scale(*scale)
 for (r,c),cell in t.get_celld().items():
  cell.set_edgecolor('#cccccc');cell.set_linewidth(.4)
  if r==0:cell.set_text_props(fontweight='bold')
 ax.set_axis_off()

DARKP='#3E2766'  # 与 Fig 5 共定位主色一致
# ---------- 版式：字号只用三档（面板字母 8 pt 粗体 / 面板标题 7 pt / 其余 6 pt），宽 180 mm、高 ≤ 170 mm；
# 表格为三线表；图内长注释放在图注。 ----------
FS,TT=6,7
def H(C,x,y,l,t):C.head(x,y,l,t,fs=TT)
def tk(a):ticks(a,FS,FS)
def booktabs(C,x0,y0,cols,rows,rh=3.4):
 """三线表：cols = [(表头, x mm, 'l'/'r')]，rows = 行列表（字符串）。返回表底 y。"""
 for t,x,al in cols:C.text(x,y0+2.2,t,fontsize=FS,fontweight='bold',ha='left' if al=='l' else 'right',va='baseline')
 xl=min(x for _,x,al in cols if al=='l')-1;xr=max(x for _,x,al in cols if al=='r')+1
 C.line(xl,y0-.6,xr,y0-.6,color=INK,lw=.6);C.line(xl,y0+3.4,xr,y0+3.4,color=INK,lw=.4)
 for i,r in enumerate(rows):
  for (t,x,al),v in zip(cols,r):C.text(x,y0+3.4+rh*(i+1)-.9,v,fontsize=FS,ha='left' if al=='l' else 'right',va='baseline')
 yb=y0+3.4+rh*len(rows)+.6;C.line(xl,yb,xr,yb,color=INK,lw=.6);return yb
def flab(f):
 parts=f.split('__');kind=parts[-1];names=[{'IFNg':'IFNγ'}.get(x,x) for x in parts[:-1]]
 return {'moran':f"{names[0]} Moran's I",'high_fraction':f'{names[0]} high fraction','entropy':f'{names[0]} entropy','coloc':'–'.join(names)+' co-loc.','mean':f'{names[0]} mean'}.get(kind,f)

# ---------------- S1 数据与标签质控 ----------------
def S1():
 man=manifest();ov=pd.read_csv(RS/'t10/qc/HER2ST_He2020_overlap.csv');rows=[]
 tech={'He2020':('legacy ST','100 µm'),'HER2ST':('legacy ST','100 µm'),'Meylan_ccRCC':('Visium','55 µm')}
 for co in ['He2020','HER2ST','Meylan_ccRCC']:
  g=man[man.cohort.eq(co)]
  for sp_ in [s for s in ['train','val','test','external'] if s in set(g.split)]:
   h=g[g.split.eq(sp_)];keep=clean(h) if co=='HER2ST' else h
   rows.append(dict(cohort=co.replace('_ccRCC',' ccRCC'),technology=tech[co][0],spot=tech[co][1],split=sp_,patients=keep.patient_key.nunique(),sections=len(keep),spots=int(keep.spots.sum()),excluded=len(h)-len(keep)))
 tb=pd.DataFrame(rows)
 C=Canvas(180,166);A=C.ax
 # a 三线表
 H(C,2,5,'a','Spatial transcriptomics cohorts')
 cols=[('Cohort',4,'l'),('Technology',23,'l'),('Spot',39,'l'),('Split',50,'l'),('Patients',67,'r'),('Sections',79,'r'),('Spots',88,'r'),('Excluded',101,'r')];prev=None;cells=[]
 for r in tb.itertuples():
  first=r.cohort!=prev;prev=r.cohort
  cells.append([r.cohort if first else '',r.technology if first else '',r.spot if first else '',r.split,f'{r.patients}',f'{r.sections}',f'{r.spots:,}',f'{r.excluded}'])
 booktabs(C,2,10,cols,cells)
 # b 指纹距离（按 HER2ST 患者分组）
 H(C,2,49,'b','HER2ST sections duplicated in He2020')
 o=ov.copy();o['pt']=o.HER2ST_patient.str.split().str[-1];o=o.sort_values(['pt','HER2ST_sample']).reset_index(drop=True);o['x']=np.arange(len(o))+o.pt.rank(method='dense').to_numpy()*.8
 a=A(17,55,82,30)
 a.scatter(o.x,o.max_quantile_diff,c=o.duplicate.map({True:'#D55E00',False:'#4d4d4d'}),s=7,lw=0,zorder=3);a.axhline(.03,ls=(0,(3,2)),color='#9a9a9a',lw=.5);a.set_yscale('log')
 a.text(o.x.max()+.6,.03,'0.03',fontsize=FS,va='bottom',ha='right',color=MUTED)
 ctr=o.groupby('pt').x.agg(['min','max']);a.set_xticks((ctr['min']+ctr['max'])/2);a.set_xticklabels(ctr.index);a.tick_params(axis='x',length=0);a.set_xlim(o.x.min()-1,o.x.max()+1)
 for i,(lo,hi) in enumerate(zip(ctr['min'],ctr['max'])):
  if i:a.axvline(lo-.9,color='#eeeeee',lw=.4,zorder=0)
 a.set_xlabel('HER2ST patient (sections)');a.set_ylabel('Max quantile difference\nto nearest He2020 section');tk(a)
 a.scatter([],[],s=7,color='#D55E00',lw=0,label='identical section (excluded)');a.scatter([],[],s=7,color='#4d4d4d',lw=0,label='distinct section')
 a.legend(frameon=False,loc='upper right',fontsize=FS,handletextpad=.2,borderaxespad=.2,labelspacing=.3)
 # c 配对 H&E（E–H 各一对重复切片 + 患者 B 非重复对照，即 Fig 2a 示例切片）
 H(C,108,5,'c','Matched pairs and a control')
 PAIRS=['SPA128','SPA125','SPA124','SPA120','SPA147'];ovs=ov.set_index('HER2ST_sample');assert ovs.loc[PAIRS[:4]].duplicate.all() and not ovs.loc['SPA147'].duplicate
 import openslide
 def thumb(s,ax):
  r=man.set_index('sample').loc[s];sl=openslide.OpenSlide(r.image);lv=sl.level_count-1;im=sl.read_region((0,0),lv,sl.level_dimensions[lv]);sl.close()
  im=Image.alpha_composite(Image.new('RGBA',im.size,(255,255,255,255)),im).convert('RGB');im.thumbnail((400,400));ax.imshow(im);ax.set_axis_off()
 C.text(133,11.5,'HER2ST',fontsize=FS,fontweight='bold',ha='center',va='baseline');C.text(152,11.5,'He2020',fontsize=FS,fontweight='bold',ha='center',va='baseline')
 for i,s in enumerate(PAIRS):
  y0=13.5+i*15.8;r=ovs.loc[s]
  for k,(ss,lab) in enumerate([(s,s),(r.best_He2020_sample,f'{r.best_He2020_sample}, {r.He2020_split}')]):
   ax=A(125+k*19,y0,16,11.5);thumb(ss,ax);C.text(133+k*19,y0+13.6,lab,fontsize=FS,ha='center',va='baseline')
  C.text(122,y0+5.8,f'{r.HER2ST_patient.split()[-1]}',fontsize=FS,fontweight='bold',ha='right',va='center');C.text(122,y0+8.6,'duplicate' if r.duplicate else 'control',fontsize=FS,ha='right',va='center',color='#D55E00' if r.duplicate else MUTED)
  C.text(172,y0+5.8,f'{r.max_quantile_diff:.3f}',fontsize=FS,ha='left',va='center')
 C.text(172,11.5,'diff.',fontsize=FS,fontweight='bold',ha='left',va='baseline')
 # d PAM50 标签一致性
 cm=pd.read_csv(RS/'t9/pam50/selfcheck_TCGA_genefu_vs_official_confusion.csv',index_col=0);ag=json.loads((RS/'t9/pam50/TCGA_label_agreement.json').read_text())
 H(C,2,98,'d','TCGA-BRCA PAM50 labels');C.text(6,103,f'agreement {ag["agreement"]:.3f}, κ = {ag["kappa"]:.3f}, n = {ag["n"]}',fontsize=FS,va='baseline',color=MUTED)
 a=A(20,108,44,44);M=cm.to_numpy();a.imshow(M,cmap='Greys',vmin=0,vmax=M.max()*1.15,aspect='auto')
 for i in range(M.shape[0]):
  for j in range(M.shape[1]):a.text(j,i,int(M[i,j]),ha='center',va='center',fontsize=FS,color='white' if M[i,j]>M.max()/2 else INK)
 a.set_xticks(range(M.shape[1]));a.set_xticklabels([c.replace('genefu_','') for c in cm.columns]);a.set_yticks(range(M.shape[0]));a.set_yticklabels([c.replace('official_','') for c in cm.index])
 a.set_xlabel('genefu call (primary labels)');a.set_ylabel('TCGA official PAM50');tk(a);a.tick_params(length=0)
 for s_ in a.spines.values():s_.set_visible(False)
 # e CPTAC 外部 ROC（官方标签训练）
 from sklearn.metrics import roc_curve,roc_auc_score
 ext=pd.read_csv(RS/'t9/pam50/PAM50_external_CPTAC.csv').set_index('analysis').loc['secondary_official_labels'];pr=pd.read_csv(RS/'t9/pam50/PAM50_CPTAC_predictions.csv');pr=pr[pr.analysis=='secondary_official_labels']
 H(C,84,98,'e','CPTAC-BRCA, model trained on official labels');C.text(88,103,f'macro-AUROC {ext.CPTAC_macro_AUROC:.3f} (TCGA CV {ext.TCGA_CV_macro_AUROC_mean:.3f})',fontsize=FS,va='baseline',color=MUTED)
 a=A(98,108,44,44);CC={'Basal':'#4D4D4D','Her2':'#E69F00','LumA':'#56B4E9','LumB':'#0072B2'};roc=[]
 C.text(148,111,'subtype (n)',fontsize=FS,fontweight='bold',va='baseline');C.text(176,111,'AUROC',fontsize=FS,fontweight='bold',va='baseline',ha='right')
 for i,cl in enumerate(['Basal','Her2','LumA','LumB']):
  y=(pr.label==cl).astype(int);f,t,_=roc_curve(y,pr[f'prob_{cl}']);au=roc_auc_score(y,pr[f'prob_{cl}']);a.plot(f,t,color=CC[cl],lw=1);roc.append(dict(subtype=cl,n=int(y.sum()),AUROC=au))
  yy=115+i*3.4;C.line(148,yy-.7,151,yy-.7,color=CC[cl],lw=1);C.text(152,yy,f'{cl} ({y.sum()})',fontsize=FS,va='baseline');C.text(176,yy,f'{au:.2f}',fontsize=FS,va='baseline',ha='right')
 a.plot([0,1],[0,1],ls=':',color='#bbbbbb',lw=.6);a.set_xlim(0,1);a.set_ylim(0,1.01);a.set_xticks([0,.5,1]);a.set_yticks([0,.5,1]);a.set_xlabel('False positive rate');a.set_ylabel('True positive rate');tk(a)
 C.save('FigS1');write_source('FigS1',{'a_cohorts':tb,'b_overlap':ov,'d_confusion':cm.reset_index(),'e_predictions':pr,'e_AUROC':pd.DataFrame(roc)})

# ---------------- S2 骨干对比 ----------------
def S2():
 B=['UNI2h','conch','densenet121','stnet_finetune'];LAB={'UNI2h':'UNI2-h','conch':'CONCH','densenet121':'DenseNet-121, frozen','stnet_finetune':'ST-Net, fine-tuned'};COL={'UNI2h':'#0072B2','conch':'#E69F00','densenet121':'#9a9a9a','stnet_finetune':'#3a3a3a'}
 sec=clean(pd.concat([pd.read_csv(RS/f't9/backbones/{b}/heldout_section_spearman.csv') for b in B]))
 pat=sec.groupby(['backbone','evaluation','patient','program'],as_index=False).spearman.mean()
 EV=[('He2020_test','He2020 test','breast'),('HER2ST_external','HER2ST A–D','breast, external'),('Meylan_ccRCC_test','Meylan test','kidney')]
 C=Canvas(180,60);A=C.ax;w=.8/len(B);BX=[14,69,124];BW=52
 for k_,b in enumerate(B):
  x=[14,30,46,77][k_];C.line(x,5,x+3,5,color=COL[b],lw=1.4);C.text(x+4,5,LAB[b],fontsize=FS,va='center')
 for i,(ev,t1,t2) in enumerate(EV):
  a=A(BX[i],15,BW,34)
  for k,b in enumerate(B):
   for j,p in enumerate(P):
    v=pat[(pat.backbone==b)&(pat.evaluation==ev)&(pat.program==p)].spearman.to_numpy();x=j+(k-(len(B)-1)/2)*w
    a.scatter(np.full(len(v),x)+np.linspace(-w*.18,w*.18,len(v)),v,s=4,color=COL[b],alpha=.55,lw=0);a.plot([x-w*.4,x+w*.4],[np.median(v)]*2,color=COL[b],lw=1.4)
  for j in range(1,4):a.axvline(j-.5,color='#eeeeee',lw=.4,zorder=0)
  a.axhline(0,color='#c8c8c8',lw=.5);a.set_xlim(-.5,3.5);a.set_ylim(-.1,.7);a.set_yticks([0,.2,.4,.6]);a.set_xticks(range(4));a.set_xticklabels(S);tk(a);a.tick_params(axis='x',length=0)
  n=pat[pat.evaluation==ev].patient.nunique();C.text(BX[i],12,t1,fontsize=FS,fontweight='bold',va='baseline');C.text(BX[i]+BW,12,f'{t2} · {n} patients',fontsize=FS,va='baseline',ha='right',color=MUTED)
  if i==0:a.set_ylabel('Within-patient Spearman ρ')
  else:a.set_yticklabels([])
 med=pat.groupby(['backbone','evaluation','program'],as_index=False).spearman.median();mm=med.groupby('backbone').spearman.mean()
 C.text(14,57,'Mean of 12 medians: '+'; '.join(f'{LAB[b]} {mm[b]:.3f}' for b in B),fontsize=FS,color=MUTED,va='baseline')
 C.save('FigS2');write_source('FigS2',{'patient_rho':pat,'medians':med})

# ---------------- S3 读出上限与 HEST 外部 ----------------
SM3=['TENX39','TENX53','TENX68','TENX161','TENX162','TENX180']
def S3d_data(s):
 """d 面板每张切片的实测 / 预测图。首次读取 HEST 原始数据；结果缓存到 results/t10/data/S3d_cache/。"""
 cf=D/'S3d_cache'/f'{s}.npz'
 if cf.exists():z=np.load(cf,allow_pickle=True);return {k:z[k] for k in z.files}
 assert HEST.exists(),f'{HEST} 未挂载，且无缓存 {cf}'
 from t10_ceiling import cap_ranks,ucell,aggregate,neighbor_groups,members as MB,bg as BG,mapping as MP
 from t8_measure import gene_columns,read_counts
 from scipy import sparse
 ps=pd.read_csv(RS/'t8/paired_spots.csv');md=json.loads((HEST/'metadata'/f'{s}.json').read_text())
 with h5py.File(HEST/'st'/f'{s}.h5ad','r') as f:bar=[x_.decode() if isinstance(x_,bytes) else str(x_) for x_ in f['obs']['_index'][:]];cols=gene_columns(f,MP,BG)
 present=[x_ for x_ in BG if x_ in cols];X=read_counts(HEST/'st'/f'{s}.h5ad',np.array([cols[x_] for x_ in present]));q=ps[ps['sample'].eq(s)].set_index('barcode');keep=np.array([b in q.index for b in bar]);Xk=X[keep];qq=q.loc[[b for b in bar if b in q.index]]
 pred=qq[['E2F_predicted','IFNg_predicted','Hypoxia_predicted','EMT_predicted']].to_numpy();xy=qq[['x_um','y_um']].to_numpy();vis=md.get('st_technology')=='Visium'
 if vis:grp=neighbor_groups(xy,105.0);cap,ui=cap_ranks(aggregate(sparse.csr_matrix(Xk),grp),present);pred=np.vstack([pred[x_].mean(0) for x_ in grp])
 else:cap,ui=cap_ranks(Xk,present)
 obs=np.column_stack([ucell(cap,ui,[g for g in MB[p] if g in ui]) for p in P])
 with h5py.File(HEST/'patches'/f'{s}.h5','r') as f:im=f['img'][len(f['img'])//2]
 out=dict(xy=xy,obs=obs,pred=pred,im=im,tech=md.get('st_technology'),pres=md.get('preservation_method'))
 cf.parent.mkdir(exist_ok=True);np.savez_compressed(cf,**out);return out
def S3():
 from scipy.spatial import cKDTree;import matplotlib.patheffects as pe
 c=pd.read_csv(D/'ceiling_by_section.csv');cs=c[c.set.ne('HEST')]
 C=Canvas(180,170);A=C.ax;MK={'He2020':'o','HER2ST':'s','Meylan_ccRCC':'^'};TY,TH=11,34
 # a 读出 vs 噪声上限
 H(C,2,5,'a','Readout vs noise ceiling');a=A(14,TY,40,TH)
 for st,g in cs[cs.support.eq('native')].groupby('set'):
  for j,p in enumerate(P):d=g[g.program.eq(p)];a.scatter(d.ceiling,d.rho,s=5,marker=MK[st],color=PCOL[S[j]],alpha=.75,lw=0)
 a.plot([0,1],[0,1],ls=(0,(2,2)),color='#b0b0b0',lw=.5);a.set_xlim(0,1);a.set_ylim(-.2,1);a.set_xlabel('Noise ceiling (√ split-half reliability)');a.set_ylabel('Section Spearman ρ');tk(a)
 a.legend(handles=[Line2D([],[],marker=MK[k],ls='',color=MUTED,ms=2.6,mew=0,label=l) for k,l in [('He2020','He2020 test'),('HER2ST','HER2ST A–D'),('Meylan_ccRCC','Meylan test')]],frameon=False,fontsize=FS,loc='upper left',handletextpad=.1,borderaxespad=.2,labelspacing=.3)
 # b ρ / 上限
 H(C,66,5,'b','Fraction of ceiling reached');a=A(78,TY,42,TH);sm=pd.read_csv(D/'ceiling_summary_ST.csv');sm=sm[sm.support.eq('native')];x=0;tx=[]
 for st in ['He2020','HER2ST','Meylan_ccRCC']:
  for j,p in enumerate(P):a.bar(x,sm[(sm.set==st)&(sm.program==p)].rho_over_ceiling.iloc[0],color=PCOL[S[j]],width=.8,lw=0);x+=1
  tx.append(x-2.5);x+=1
 a.set_xticks(tx);a.set_xticklabels(['He2020','HER2ST A–D','Meylan']);a.set_ylabel('Median ρ / noise ceiling');a.set_ylim(0,.6);a.set_xlim(-.8,x-1.2);tk(a);a.tick_params(axis='x',length=0)
 a.legend(handles=[plt.Rectangle((0,0),1,1,color=PCOL[k],lw=0) for k in S],labels=S,frameon=False,fontsize=FS,ncol=2,loc='upper left',handlelength=1,handleheight=.8,handletextpad=.3,columnspacing=.8,borderaxespad=.2)
 # c 相邻切片 Moran's I
 rp=pd.read_csv(RS/'t5/He2020_between_section_patterns.csv');rp=rp[rp.consecutive_numbered_within_block];mo=pd.read_csv(D/'st_moran_by_section.csv');test=set(mo[(mo.cohort=='He2020')&(mo.split=='test')]['sample'])
 rp=rp[rp.sample_A.isin(test)&rp.sample_B.isin(test)].sort_values(['patient','sample_A','sample_B']).groupby(['patient','program'],as_index=False).first();m=mo.set_index(['sample','program']).main_moran
 rp['main_A']=[m[(x_,p)] for x_,p in zip(rp.sample_A,rp.program)];rp['main_B']=[m[(x_,p)] for x_,p in zip(rp.sample_B,rp.program)]
 H(C,132,5,'c',f"Adjacent sections, He2020 test");a=A(144,TY,34,TH)
 for j,p in enumerate(P):d=rp[rp.program.eq(p)];a.scatter(d.observed_A,d.observed_B,s=7,facecolor='white',edgecolor=PCOL[S[j]],lw=.6);a.scatter(d.main_A,d.main_B,s=7,color=PCOL[S[j]],lw=0)
 a.plot([0,1],[0,1],ls=(0,(2,2)),color='#b0b0b0',lw=.5);a.set_xlim(0,.8);a.set_ylim(0,.8);a.set_xticks([0,.4,.8]);a.set_yticks([0,.4,.8]);a.set_xlabel("Moran's I, section A");a.set_ylabel("Moran's I, section B");tk(a)
 a.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=MUTED,mew=.6,ms=3,label='measured'),Line2D([],[],marker='o',ls='',color=MUTED,mew=0,ms=3,label='predicted')],frameon=False,fontsize=FS,loc='upper left',handletextpad=.1,borderaxespad=.2,labelspacing=.3)
 # d HEST 外部
 Y0,PITCH,MS=70,16.4,12.6;X0=39;PG,GG=.5,5;out=[]
 H(C,2,59,'d','HEST external breast sections (not used for training), matched support')
 for j in range(4):
  xc=X0+j*(2*MS+PG+GG);C.text(xc+MS+PG/2,63.5,S[j],fontsize=FS,fontweight='bold',ha='center',va='baseline',color=PCOL[S[j]])
  for m_,t in enumerate(['measured','predicted']):C.text(xc+m_*(MS+PG)+MS/2,67,t,fontsize=FS,ha='center',va='baseline',color=MUTED)
 C.text(X0-14.5+MS/2,67,'H&E, 112 µm',fontsize=FS,ha='center',va='baseline',color=MUTED)
 for i,s in enumerate(SM3):
  z=S3d_data(s);xy=z['xy'].astype(float);xy[:,1]*=-1;vis=str(z['tech'])=='Visium';y=Y0+i*PITCH
  nn=np.median(cKDTree(xy).query(xy,k=2)[0][:,1]);ctr=(xy.max(0)+xy.min(0))/2;Rg=np.ptp(xy,0).max()+nn
  C.text(2,y+3,s,fontsize=FS,fontweight='bold',va='baseline')
  C.text(2,y+6,str(z['tech']),fontsize=FS,va='baseline',color=MUTED);C.text(2,y+9,str(z['pres']),fontsize=FS,va='baseline',color=MUTED)
  L=1000/Rg*MS;C.line(2,y+11.8,2+L,y+11.8,color=INK,lw=1);C.text(2+L+1,y+11.8,'1 mm',fontsize=FS,va='center')
  a=A(X0-14.5,y,MS,MS);im=z['im'];a.imshow(im);a.set_axis_off();scalebar(a,im.shape[1]*50/112,'50 µm',color='white',fs=FS);[t.set_path_effects([pe.withStroke(linewidth=1.2,foreground='#333333')]) for t in a.texts[-1:]+a.lines[-1:]]
  dpt=nn/Rg*MS*MM*72
  for j in range(4):
   r=corr(z['obs'][:,j],z['pred'][:,j]);out.append(dict(sample=s,program=P[j],support='spot+6 neighbours' if vis else 'native 128 µm bin',rho=r,spots=len(xy)))
   xc=X0+j*(2*MS+PG+GG)
   for m_,v in enumerate([z['obs'][:,j],z['pred'][:,j]]):
    a=A(xc+m_*(MS+PG),y,MS,MS);a.scatter(xy[:,0],xy[:,1],c=pd.Series(v).rank(pct=True),cmap='viridis',vmin=0,vmax=1,s=(dpt*(1.0 if vis else 1.12))**2,marker='o' if vis else 's',lw=0)
    a.set_xlim(ctr[0]-Rg/2,ctr[0]+Rg/2);a.set_ylim(ctr[1]-Rg/2,ctr[1]+Rg/2);a.set_axis_off()
   C.text(xc+MS+PG/2,y+MS+2.4,f'ρ = {r+0:.2f}'.replace('-0.00','0.00').replace('-','−'),fontsize=FS,ha='center',va='baseline')
 cb=A(X0+4*(2*MS+PG)+3*GG+3,Y0,1.6,MS*1.6);cb.imshow(np.linspace(1,0,100)[:,None],aspect='auto',cmap='viridis',extent=[0,1,0,1]);cb.set_xticks([]);cb.yaxis.tick_right();cb.set_yticks([0,1]);cb.set_yticklabels(['low','high']);tk(cb);[sp.set_visible(False) for sp in cb.spines.values()]
 cb.set_ylabel('Within-section rank',fontsize=FS,rotation=90,labelpad=1);cb.yaxis.set_label_position('right')
 dd=pd.DataFrame(out);old=pd.read_csv(O/'FigS3_source_data/d_HEST_matched.csv') if (O/'FigS3_source_data/d_HEST_matched.csv').exists() else None
 if old is not None:assert np.allclose(old.rho.to_numpy(),dd.rho.to_numpy(),atol=1e-9),'d 的 ρ 与旧版不一致'
 C.save('FigS3');write_source('FigS3',{'a_sections':cs,'b_summary':sm,'c_repeated_sections':rp,'d_HEST_matched':dd,'d_HEST_all':c[c.set.eq('HEST')]})
# ---------------- S4 关联与区分 ----------------
def S4():
 cv=pd.read_csv(D/'clinical_review_repeatedCV.csv');lr=pd.read_csv(D/'clinical_review_LRT.csv');ch=pd.read_csv(D/'clinical_review_choices.csv');old=pd.read_csv(RS/'t6/clinical_CV_fold_metrics.csv')
 b=cv[cv.model.eq('M1 base')].set_index(['set','rep']).C;cv['dC']=[r.C-b[(r.set,r.rep)] for r in cv.itertuples()]
 C=Canvas(180,164);A=C.ax
 # a 原设计（单次 5 折）
 H(C,2,5,'a','Original design, single 5-fold');a=A(16,13,40,34);o=old[(old.source=='T6_dual')&(old.task=='PFI')];xt=[];xl=[]
 for k,co in enumerate(['TCGA-BRCA','TCGA-KIRC']):
  for m,(mod,c_,lab) in enumerate([('mean','#9a9a9a','means'),('mean_plus_spatial','#3a3a3a','+spatial')]):
   v=o[(o.cohort==co)&(o.model==mod)].value;x=k*3+m*1.3;a.scatter(np.full(len(v),x)+np.linspace(-.12,.12,len(v)),v,s=6,color=c_,lw=0);a.plot([x-.3,x+.3],[v.mean()]*2,color=c_,lw=1.4);xt.append(x);xl.append(lab)
 a.set_xticks(xt);a.set_xticklabels(xl);a.set_xlim(-.7,5);a.set_ylabel('Fold C-index');tk(a);a.tick_params(axis='x',length=0)
 for k,co in enumerate(['BRCA','KIRC']):C.text(16+40*(k*3+.65+.7)/5.7,54,co,fontsize=FS,ha='center',va='baseline',fontweight='bold')
 # b 修订设计：重复 20 × 5 折 ΔC
 H(C,58,5,'b','Revised design, 20 × 5-fold pooled out-of-fold C-index');a=A(72,13,106,34);x=0;tick=[];lab=[];grp=[]
 NM={'M2 +18 spatial (tuned ridge)':'+18 spatial\n(ridge)','M3 +1 spatial (nested selection)':'+1 spatial\n(nested)','M1 + log TIL%':'+ log TIL%','M1 + co-localization':'+ co-loc.','M1 + both':'+ both'}
 for st,gl in [('TCGA-BRCA','TCGA-BRCA'),('TCGA-KIRC','TCGA-KIRC'),('TCGA-BRCA Saltz subset','TCGA-BRCA, Saltz subset')]:
  x0=x
  for m in [m for m in cv[cv.set.eq(st)].model.unique() if m!='M1 base']:
   v=cv[(cv.set==st)&(cv.model==m)].dC;a.scatter(np.full(len(v),x)+np.linspace(-.15,.15,len(v)),v,s=4,color='#7a7a7a',alpha=.6,lw=0);a.plot([x-.3,x+.3],[v.mean()]*2,color=INK,lw=1.4);tick.append(x);lab.append(NM[m]);x+=1
  grp.append((x0,x-1,gl));x+=.7
 a.axhline(0,color='#9a9a9a',lw=.5,ls=(0,(3,2)));a.set_xticks(tick);a.set_xticklabels(lab);a.set_xlim(-.6,x-.7+.6-.7);a.set_ylabel('ΔC-index vs base');tk(a);a.tick_params(axis='x',length=0)
 X0,X1=a.get_xlim()
 for x0,x1,gl in grp:
  m0=72+106*(x0-.35-X0)/(X1-X0);m1=72+106*(x1+.35-X0)/(X1-X0);C.line(m0,57,m1,57);C.text((m0+m1)/2,60,gl,fontsize=FS,ha='center',va='baseline')
 C.text(178,11,'base = age + stage + 4 programme means',fontsize=FS,ha='right',va='baseline',color=MUTED)
 # c 嵌套似然比检验（三线表）
 H(C,2,70,'c','Nested likelihood-ratio tests')
 t=lr.copy();t['added']=t.added.str.replace('E2F__IFNg__coloc','E2F–IFNγ co-loc.').str.replace(' spatial features',' spatial features')
 cols=[('Cohort',4,'l'),('Added to base',19,'l'),('df',51,'r'),('LR',60,'r'),('p',71,'r')]
 booktabs(C,2,75,cols,[[r.cohort,r.added,f'{r.df}',f'{r.LR:.2f}',f'{r.p:.4f}'] for r in t.itertuples()])
 # d 嵌套选择被选特征
 H(C,84,70,'d','Feature chosen by nested selection (100 training folds)');sel=ch[ch.model.str.startswith('M3')];rows=[];heads=[];yy=0
 for co in ['TCGA-BRCA','TCGA-KIRC']:
  heads.append((yy,co));yy-=1
  for name,n in sel[sel.set.eq(co)].choice.value_counts().head(6).items():rows.append(dict(cohort=co,feature=name,n=n,y=yy));yy-=1
  yy-=.3
 rows=pd.DataFrame(rows);a=A(122,76,50,40)
 a.barh(rows.y,rows.n,color=[DARKP if f=='E2F__IFNg__coloc' else '#9a9a9a' for f in rows.feature],height=.7)
 for r in rows.itertuples():a.text(r.n+1.2,r.y,f'{r.n}',fontsize=FS,va='center')
 a.set_yticks(rows.y);a.set_yticklabels([flab(f) for f in rows.feature]);a.tick_params(axis='y',length=0);a.set_ylim(yy+.5,.6);a.set_xlim(0,100);a.set_xticks([0,50,100]);a.set_xlabel('Times selected');tk(a);a.spines['left'].set_visible(False);a.tick_params(axis='y',length=0)
 tr=mpl.transforms.blended_transform_factory(C.fig.transFigure,a.transData)
 for y_,t_ in heads:a.text(88/180,y_,t_,transform=tr,fontsize=FS,fontweight='bold',va='center')
 # e 采样竖条：散点 + Cox HR 小森林图
 sq=pd.read_csv(RS/'t10/qc/TCGA_BRCA_sampling_stripes.csv').merge(pd.read_csv(RS/'t7/patient_spatial_clinical.csv')[['patient_id','E2F__IFNg__coloc']],on='patient_id').rename(columns={'E2F__IFNg__coloc':'coloc'})
 sx=pd.read_csv(D/'sampling_stripes_cox.csv')
 H(C,2,124,'e','Striped sampling of the official patch grid (TCGA-BRCA)')
 a=A(16,130,44,22);a.scatter(sq.stripe,sq.coloc,s=1.4,color='#7a7a7a',alpha=.5,lw=0,rasterized=True);a.axvline(.05,color='#9a9a9a',lw=.5,ls=(0,(3,2)))
 a.set_xlabel('Sampling stripe index');a.set_ylabel('Co-localization');a.set_ylim(-.6,1.05);a.set_yticks([-.5,0,.5]);tk(a);a.text(.97,.97,f'ρ = {corr(sq.stripe,sq.coloc):.2f}\nstriped (≥ 0.05): {(sq.stripe>=.05).mean()*100:.0f}%',transform=a.transAxes,fontsize=FS,ha='right',va='top')
 LB={'T7 model (all)':'Primary model','+ stripe index':'+ stripe index','regular sampling only (stripe < 0.05)':'Regular sampling only','striped sampling only (stripe ≥ 0.05)':'Striped sampling only'}
 a=A(110,131,34,18);y=np.arange(len(sx))[::-1];trr=mpl.transforms.blended_transform_factory(C.fig.transFigure,a.transData)
 for yi,r in zip(y,sx.itertuples()):
  col=DARKP if r.model.startswith('T7') else INK;a.plot([r.CI_low,r.CI_high],[yi]*2,color=col,lw=.6);a.scatter(r.HR,yi,s=10 if r.model.startswith('T7') else 6,color=col,lw=0,zorder=3,marker='D' if r.model.startswith('T7') else 'o')
  for xx,t_ in [(155,f'{r.HR:.2f}'),(166,f'{r.p:.3f}'),(177,f'{r.n:,}')]:a.text(xx/180,yi,t_,transform=trr,fontsize=FS,ha='right',va='center')
 for xx,t_ in [(155,'HR'),(166,'p'),(177,'n')]:a.text(xx/180,len(sx)-.3,t_,transform=trr,fontsize=FS,ha='right',va='center',color=MUTED,style='normal' if t_=='HR' else 'italic')
 a.axvline(1,color='#bbbbbb',lw=.5,ls=(0,(3,2)));a.set_xscale('log');a.set_xticks([.5,1]);a.set_xticklabels(['0.5','1']);a.xaxis.set_minor_locator(mpl.ticker.NullLocator());a.set_xlim(.45,1.2)
 a.set_yticks(y);a.set_yticklabels([LB[m] for m in sx.model]);a.set_ylim(-.6,len(sx)-.1);a.set_xlabel('Co-localization HR per s.d.');tk(a);a.tick_params(axis='y',length=0);a.spines['left'].set_visible(False)
 C.save('FigS4');write_source('FigS4',{'a_original_folds':o,'b_repeatedCV':cv,'c_LRT':lr,'d_choices':sel,'e_sampling':sq,'e_cox':sx})

def SD1():
 from matplotlib.backends.backend_pdf import PdfPages
 pdf=PdfPages(O/'SupplementaryData1_heldout_sections.pdf')
 sp=pd.read_parquet(D/'st_spot_predictions.parquet');sec=clean(pd.read_csv(D/'st_section_rho.csv'));sec=sec[sec.split.isin(['test','external'])]
 for tag,co in [('a','He2020'),('b','HER2ST'),('c','Meylan_ccRCC')]:
  ss=sec[sec.cohort.eq(co)].drop_duplicates('sample').sort_values(['patient','sample']);n=len(ss);blocks=2 if n>8 else 1;rows=int(np.ceil(n/blocks))
  fig=plt.figure(figsize=(FULL,(14*rows+14)*MM));gs=GridSpec(rows,blocks*9,figure=fig,wspace=.05,hspace=.25,width_ratios=([1]*8+[.7])*blocks)
  for i,r in enumerate(ss.itertuples()):
   g=sp[sp['sample'].eq(r.sample)];b,rr=divmod(i,rows);rho=sec[sec['sample'].eq(r.sample)].set_index('program').rho_main
   for j,k in enumerate(SK):
    for m,(pref,lab) in enumerate([('obs','meas.'),('main','pred.')]):
     a=fig.add_subplot(gs[rr,b*9+j*2+m]);a.scatter(g.x,-g.y,c=g[f'{pref}_{k}'].rank(pct=True),cmap='viridis',s=.8 if co=='Meylan_ccRCC' else 1.3,lw=0);a.set_aspect('equal');a.set_axis_off()
     if rr==0:a.set_title(f'{S[j]}\n{lab}',fontsize=5.3,color=PCOL[S[j]])
     if m==1:a.text(.5,-.06,f'{rho[P[j]]:.2f}',transform=a.transAxes,ha='center',va='top',fontsize=4.8)
    if j==0:a0=fig.axes[-2];a0.text(-.15,.5,f'{r.sample}\n{r.patient.split("::")[-1]}',transform=a0.transAxes,rotation=90,ha='right',va='center',fontsize=4.6)
  name={'He2020':'He2020 test','HER2ST':'HER2ST A–D external','Meylan_ccRCC':'Meylan ccRCC test'}[co]
  fig.suptitle(f'{name}: measured vs predicted (dual-scale model), within-section rank; number = section Spearman ρ',fontsize=6.5,y=.995)
  pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
 pdf.close();write_source('SupplementaryData1',{'section_rho':sec})


if __name__=='__main__':
 todo=sys.argv[1:] or ['S1','S2','S3','S4','SD1']
 for t in todo:globals()[t]();print('DONE',t,flush=True)
