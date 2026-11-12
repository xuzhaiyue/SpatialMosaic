"""T10 Fig 3：双尺度监督——ST 管空间，bulk 管水平。
a 权衡曲线（ST 患者内 ρ vs TCGA OOF bulk ρ，λ 0→1 与分解模型，标出验证选择）；b 全片均值 vs bulk：ST-only → 主模型；c CPTAC-BRCA 外部 PAM50。
HER2ST 只用 Patient A–D；全部数值来自 T7 / T9 已有表，不重算模型。"""
from t10_common import *
from sklearn.metrics import roc_curve,roc_auc_score
LAMS=['lambda_0','lambda_0.1','lambda_0.25','lambda_0.5','lambda_0.75','lambda_0.9','lambda_1'];LV=[0,.1,.25,.5,.75,.9,1]
st=clean(pd.read_csv(R/'results/t7/ST_section_all_candidates.csv'))
stp=st.groupby(['cohort','patient_key','program','candidate'],as_index=False).rho.mean().groupby(['cohort','program','candidate'],as_index=False).agg(ST_median=('rho','median'),patients=('patient_key','nunique'))
bk=pd.read_csv(R/'results/t7/bulk_correlations_all_candidates.csv')
# 自检：主模型 bulk ρ 与 T6 报告一致（乳腺 λ=0.5 = T6）
assert abs(bk[(bk.cohort=='TCGA-BRCA')&(bk.candidate=='lambda_0.5')&(bk.program==P[0])].rho.iloc[0]-0.680)<5e-4
# 字号三档（面板字母 8 pt 粗体 / 标题 7 pt / 其余 6 pt），同行面板等高，标题左对齐
C=Canvas(180,137);A=C.ax;FS,TT=6,7
def H(x,y,l,t):C.head(x,y,l,t,fs=TT)
def tk(a):ticks(a,FS,FS)
def sub(x0,w,y,t1,t2):C.text(x0,y,t1,fontsize=FS,fontweight='bold',va='baseline');C.text(x0+w,y,t2,fontsize=FS,color=MUTED,va='baseline',ha='right')

# ---------- a 权衡曲线 ----------
PANELS=[('He2020','TCGA-BRCA','He2020 test','breast · TCGA-BRCA'),('HER2ST','TCGA-BRCA','HER2ST A–D','breast · TCGA-BRCA'),('Meylan_ccRCC','TCGA-KIRC','Meylan test','kidney · TCGA-KIRC')]
AX=[16,72,128];AW=48;arows=[]
H(2,5,'a','Spatial and bulk accuracy across the ST weight λ')
for k,(sc,bc,t1,t2) in enumerate(PANELS):
 a=A(AX[k],14,AW,38);sel=MAIN[bc];sub(AX[k],AW,11.5,t1,t2)
 for j,p in enumerate(P):
  xs=[stp[(stp.cohort==sc)&(stp.program==p)&(stp.candidate==c)].ST_median.iloc[0] for c in LAMS];ys=[bk[(bk.cohort==bc)&(bk.program==p)&(bk.candidate==c)].rho.iloc[0] for c in LAMS]
  a.plot(xs,ys,'-',color=PCOL[S[j]],lw=.8,alpha=.9,zorder=2);a.scatter(xs,ys,s=[3+9*v for v in LV],color=PCOL[S[j]],lw=0,zorder=3)
  i=LAMS.index(sel);a.scatter(xs[i],ys[i],s=34,facecolor='none',edgecolor=INK,lw=.6,zorder=4)
  xd=stp[(stp.cohort==sc)&(stp.program==p)&(stp.candidate=='decomposed')].ST_median.iloc[0];yd=bk[(bk.cohort==bc)&(bk.program==p)&(bk.candidate=='decomposed')].rho.iloc[0];a.scatter(xd,yd,marker='D',s=8,facecolor='white',edgecolor=PCOL[S[j]],lw=.6,zorder=3)
  for c,x,y in zip(LAMS,xs,ys):arows.append(dict(ST_set=sc,bulk_cohort=bc,program=p,candidate=c,ST_patient_median_rho=x,bulk_rho=y,selected=c==sel))
  arows.append(dict(ST_set=sc,bulk_cohort=bc,program=p,candidate='decomposed',ST_patient_median_rho=xd,bulk_rho=yd,selected=False))
 a.axhline(0,color='#dddddd',lw=.5,zorder=1);a.axvline(0,color='#dddddd',lw=.5,zorder=1);a.set_xlim(-.1,.5);a.set_ylim(-.1,.75);a.set_xticks([0,.2,.4]);a.set_yticks([0,.2,.4,.6])
 a.set_xlabel('ST within-patient ρ (median)');tk(a)
 if k==0:a.set_ylabel('Bulk patient-level ρ (TCGA)')
 else:a.set_yticklabels([])
# 图例（a、b 共用颜色键），一行
lg=A(0,59,180,5);lg.set_xlim(0,180);lg.set_ylim(5,0);lg.set_axis_off()
for j,(s_,x) in enumerate(zip(S,[16,26,37,51])):lg.scatter([x],[2.5],s=7,color=PCOL[s_],lw=0);lg.text(x+1.4,2.5,s_,fontsize=FS,va='center')
lg.text(62,2.5,'λ = 0',fontsize=FS,va='center')
for i,v in enumerate(LV):lg.scatter([70+i*2.2],[2.5],s=3+9*v,color=MUTED,lw=0)
lg.text(85,2.5,'λ = 1 (bulk only → ST only)',fontsize=FS,va='center')
lg.scatter([121],[2.5],s=34,facecolor='none',edgecolor=INK,lw=.6);lg.text(123.5,2.5,'validation-selected λ',fontsize=FS,va='center')
lg.scatter([151],[2.5],marker='D',s=8,facecolor='white',edgecolor=MUTED,lw=.6);lg.text(153,2.5,'decomposed model',fontsize=FS,va='center')

# ---------- b 全片均值 vs bulk ----------
COH=[('TCGA-BRCA','cross-fitted'),('CPTAC-BRCA','external'),('TCGA-KIRC','cross-fitted'),('CPTAC-CCRCC','external')];brows=[]
BX=[16,42,72,98];BW=24;BY,BH=88,36
H(2,72,'b','Slide-mean prediction vs bulk RNA')
for c,(co,sub_) in enumerate(COH):
 a=A(BX[c],BY,BW,BH);sel=MAIN['TCGA-BRCA' if 'BRCA' in co else 'TCGA-KIRC']
 for j,p in enumerate(P):
  r=bk[(bk.cohort==co)&(bk.program==p)];v1=r[r.candidate=='lambda_1'].rho.iloc[0];vm=r[r.candidate==sel].rho.iloc[0];n=int(r.n.iloc[0])
  a.plot([j,j],[v1,vm],color=PCOL[S[j]],lw=1,alpha=.5,zorder=2);a.scatter(j,v1,s=9,color=GREY,lw=0,zorder=3);a.scatter(j,vm,s=12,color=PCOL[S[j]],lw=0,zorder=4)
  brows.append(dict(cohort=co,program=p,n=n,ST_only_rho=v1,main_rho=vm,main_candidate=sel))
 C.text(BX[c]+BW/2,82,co,fontsize=FS,fontweight='bold',ha='center',va='baseline');C.text(BX[c]+BW/2,85.4,f'n = {n:,}',fontsize=FS,color=MUTED,ha='center',va='baseline')
 a.axhline(0,color='#d9d9d9',lw=.5,zorder=0);a.set_xlim(-.6,3.6);a.set_ylim(-.15,.75);a.set_xticks([]);a.set_yticks([0,.2,.4,.6]);tk(a)
 if c==0:a.set_ylabel('Patient-level Spearman ρ')
 else:a.set_yticklabels([]);a.spines['left'].set_color('#bbbbbb')
for (x0_,x1_),gt in [((BX[0],BX[1]+BW),'Breast'),((BX[2],BX[3]+BW),'Kidney')]:
 C.text((x0_+x1_)/2,77.4,gt,ha='center',va='baseline',fontsize=FS);C.line(x0_,78.6,x1_,78.6)
lg=A(16,126.5,110,5);lg.set_xlim(0,110);lg.set_ylim(5,0);lg.set_axis_off()
lg.plot([1,1],[4.2,1],color=MUTED,lw=1,alpha=.5);lg.scatter([1],[4.2],s=9,color=GREY,lw=0);lg.scatter([1],[1],s=12,color=INK,lw=0)
lg.text(2.8,2.6,'grey, ST-only (λ = 1); coloured, dual-scale. TCGA cross-fitted; CPTAC external',fontsize=FS,va='center')

# ---------- c CPTAC 外部 PAM50 ----------
ext=pd.read_csv(R/'results/t9/pam50/PAM50_external_CPTAC.csv').set_index('analysis').loc['primary_genefu_labels'];pr=pd.read_csv(R/'results/t9/pam50/PAM50_CPTAC_predictions.csv');pr=pr[pr.analysis=='primary_genefu_labels']
H(124,72,'c','External PAM50 subtype');a=A(136,BY,38,BH);crows=[];CL=['Basal','Her2','LumA','LumB'];CC={'Basal':'#4D4D4D','Her2':'#E69F00','LumA':'#56B4E9','LumB':'#0072B2'}
for cl in CL:
 y=(pr.label==cl).astype(int);s=pr[f'prob_{cl}'];fpr,tpr,_=roc_curve(y,s);auc=roc_auc_score(y,s);assert abs(auc-ext[f'CPTAC_AUROC_{cl}'])<1e-9
 a.plot(fpr,tpr,color=CC[cl],lw=1,label=f'{cl} ({int(y.sum())})  {auc:.2f}');crows.append(dict(subtype=cl,n=int(y.sum()),AUROC=auc))
a.plot([0,1],[0,1],ls=':',color='#bbbbbb',lw=.6);a.set_xlim(0,1);a.set_ylim(0,1.01);a.set_xticks([0,.5,1]);a.set_yticks([0,.5,1]);a.set_xlabel('False positive rate');a.set_ylabel('True positive rate');tk(a)
sub(136,38,82,'CPTAC-BRCA',f'external · n = {int(ext.CPTAC_n)}');C.text(136,85.4,f'macro-AUROC {ext.CPTAC_macro_AUROC:.2f}; TCGA CV {ext.TCGA_CV_macro_AUROC_mean:.2f} ± {ext.TCGA_CV_SD:.2f}',fontsize=FS,color=MUTED,va='baseline')
a.legend(frameon=False,loc='lower right',fontsize=FS,handlelength=1.1,handletextpad=.4,borderaxespad=.1,labelspacing=.25,title='subtype (n), AUROC',title_fontsize=FS)
pass
C.save('Fig3')
write_source('Fig3',{'a_tradeoff':pd.DataFrame(arows),'b_bulk_correspondence':pd.DataFrame(brows),'c_CPTAC_PAM50':pd.concat([pd.DataFrame(crows),pd.DataFrame([dict(subtype='macro',n=int(ext.CPTAC_n),AUROC=ext.CPTAC_macro_AUROC),dict(subtype='TCGA_CV_macro_mean',n=int(ext.train_n),AUROC=ext.TCGA_CV_macro_AUROC_mean)])]),'c_CPTAC_predictions':pr})
print('FIG3 DONE',flush=True)
