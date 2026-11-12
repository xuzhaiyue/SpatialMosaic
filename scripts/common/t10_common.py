"""T10 定稿图公共部分：路径、主模型、统一样式。只读已有结果；不训练、不下载。
主模型 = T7 验证选择、全部 TCGA 训练（fold0）：乳腺 λ=0.5（与 T6 相同），肾 λ=0.75。T4 ST-only 只作灰色对照。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',MPLBACKEND='Agg')
from pathlib import Path
import json,datetime
import numpy as np,pandas as pd,joblib
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[2];O=R/'results/t10';D=O/'data';D.mkdir(parents=True,exist_ok=True)
E=Path('/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts');E10=E/'t10';E10.mkdir(exist_ok=True)
P=['E2F_TARGETS','INTERFERON_GAMMA_RESPONSE','HYPOXIA','EPITHELIAL_MESENCHYMAL_TRANSITION'];S=['E2F','IFNγ','Hypoxia','EMT'];SK=['E2F','IFNg','Hypoxia','EMT']
MAIN={'TCGA-BRCA':'lambda_0.5','TCGA-KIRC':'lambda_0.75'};ST2TCGA={'He2020':'TCGA-BRCA','HER2ST':'TCGA-BRCA','Meylan_ccRCC':'TCGA-KIRC'}
def stamp():return datetime.datetime.now().astimezone().isoformat()
def corr(a,b):return float(spearmanr(a,b).statistic) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan
def main_model(tcga):return joblib.load(E/'t7/models'/f'{tcga}_{MAIN[tcga]}_fold0.joblib')
def predict(m,x,source):return (x@m['coef'].T+m['intercept'])*m[source+'_scale']+m[source+'_mean']
def manifest():return pd.read_csv(R/'results/t4/input_manifest.csv')

# ---------- 统一样式 ----------
import matplotlib as mpl
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
 'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','figure.dpi':150,'savefig.dpi':300})
import matplotlib.pyplot as plt
# 四程序配色（Okabe–Ito 子集，色盲安全）；ST-only 统一浅灰
PCOL={'E2F':'#D55E00','IFNγ':'#0072B2','Hypoxia':'#009E73','EMT':'#CC79A7'};GREY='#B8B8B8';INK='#222222';MUTED='#6b6b6b'
MM=1/25.4;FULL=180*MM;HALF=88*MM
def panel(ax,letter,x=-0.18,y=1.06):ax.text(x,y,letter,transform=ax.transAxes,fontsize=9,fontweight='bold',va='bottom',ha='left',color=INK)
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(O/f'{name}.{ext}',bbox_inches='tight',pad_inches=0.02)
 plt.close(fig)
def write_source(name,sheets):
 # 环境无 openpyxl（不另装），source data 以每面板一个 CSV 的目录形式给出
 d=O/f'{name}_source_data';d.mkdir(exist_ok=True)
 for k,v in sheets.items():v.to_csv(d/f'{k}.csv',index=False)

# ---------- HER2ST 外部 = Patient A–D（E–H 与 He2020 为同一批切片，E/F/G 在训练集） ----------
_ov=pd.read_csv(R/'results/t10/qc/HER2ST_He2020_overlap.csv')
HER2ST_DUP=set(_ov.loc[_ov.duplicate,'HER2ST_sample']);assert len(HER2ST_DUP)==12
def clean(df,col='sample'):return df[~df[col].isin(HER2ST_DUP)].copy()

# ---------- 版式：mm 画布、左对齐面板标题（字母 8 pt 粗体 + 标题 6.5 pt 常规）、细轴线、比例尺 ----------
mpl.rcParams.update({'axes.linewidth':.5,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2,'ytick.major.size':2,'xtick.major.pad':1.5,'ytick.major.pad':1.5,
 'axes.labelpad':2,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,'axes.labelcolor':INK,'text.color':INK,'lines.solid_capstyle':'butt'})
class Canvas:
 """固定 mm 画布；坐标一律以左上角为原点、单位 mm。"""
 def __init__(s,w,h):s.W,s.H=w,h;s.fig=plt.figure(figsize=(w*MM,h*MM))
 def ax(s,x,y,w,h):return s.fig.add_axes([x/s.W,1-(y+h)/s.H,w/s.W,h/s.H])
 def text(s,x,y,t,**k):return s.fig.text(x/s.W,1-y/s.H,t,**k)
 def head(s,x,y,letter,title=None,fs=6.5):
  # 面板字母与标题同一基线，左对齐
  s.text(x,y,letter,fontsize=8,fontweight='bold',va='baseline',ha='left')
  if title:s.text(x+4,y,title,fontsize=fs,va='baseline',ha='left')
 def line(s,x0,y0,x1,y1,**k):s.fig.add_artist(plt.Line2D([x0/s.W,x1/s.W],[1-y0/s.H,1-y1/s.H],**{'color':'#9a9a9a','lw':.5,**k}))
 def save(s,name):
  for ext in ['pdf','svg','png']:s.fig.savefig(O/f'{name}.{ext}',dpi=300 if ext=='png' else None,facecolor='white')
  plt.close(s.fig)
def ticks(a,size=5.5,label=6):a.tick_params(labelsize=size,length=2,width=.5,pad=1.5);a.xaxis.label.set_size(label);a.yaxis.label.set_size(label)
def scalebar(a,px,label,loc='lr',color=INK,lw=1.2,fs=5,out=False):
 """数据坐标为像素时画比例尺：px = 比例尺长度（像素）。loc: lr / ll。"""
 x0,x1=a.get_xlim();y0,y1=a.get_ylim();W=abs(x1-x0);Hh=abs(y1-y0);top=min(y0,y1);bot=max(y0,y1)
 xs=(max(x0,x1)-(0 if out else .04*W)-px) if loc=='lr' else (min(x0,x1)+(0 if out else .04*W));ys=bot+.05*Hh if out else bot-.05*Hh
 a.plot([xs,xs+px],[ys,ys],color=color,lw=lw,solid_capstyle='butt',clip_on=False,zorder=10)
 # out=True：比例尺画在图像下方，文字放在比例尺左侧
 if out:a.text(xs-.03*W,ys,label,ha='right',va='center',fontsize=fs,color=color,zorder=10)
 else:a.text(xs+px/2,ys-.025*Hh,label,ha='center',va='bottom',fontsize=fs,color=color,zorder=10)
