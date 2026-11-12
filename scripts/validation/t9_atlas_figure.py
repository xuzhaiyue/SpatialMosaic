"""T9-3 第三步：程序图谱汇总表与热图。ST = 患者内 Spearman（切片平均到患者）的患者中位数；bulk = 患者 Spearman。不做阈值筛选。"""
from t9_common import *
import matplotlib.pyplot as plt
D=O/'atlas';pat=pd.read_csv(D/'ST_patient_spearman.csv');bulk=pd.read_csv(D/'bulk_correlations.csv');mem=pd.read_csv(O/'program_members_summary.csv')
st=pat.groupby(['cohort','program']).spearman.median().unstack(0)
st=st.rename(columns={'He2020':'He2020 test (ST)','HER2ST':'HER2ST (ST)','Meylan_ccRCC':'Meylan ccRCC test (ST)'})
bk=bulk.pivot(index='program',columns='cohort',values='Spearman').rename(columns=lambda c:c+' (bulk)')
cols=['He2020 test (ST)','HER2ST (ST)','TCGA-BRCA (bulk)','CPTAC-BRCA (bulk)','Meylan ccRCC test (ST)','TCGA-KIRC (bulk)','CPTAC-CCRCC (bulk)']
t=st.join(bk)[cols];t['members']=mem.set_index('program').members;t['breast_ST_mean']=t[['He2020 test (ST)','HER2ST (ST)']].mean(1)
t=t.sort_values('breast_ST_mean',ascending=False);t.to_csv(O/'atlas_summary.csv')
lab=[(p.replace('CELL_','Cell: ').replace('_',' ') if p.startswith('CELL_') else p.replace('_',' '))+('  †' if n<4 else '') for p,n in zip(t.index,t.members)]
v=t[cols].to_numpy(float);fig,ax=plt.subplots(figsize=(8.2,0.24*len(t)+2.2))
im=ax.imshow(v,cmap='RdBu_r',vmin=-0.7,vmax=0.7,aspect='auto')
for i in range(v.shape[0]):
 for j in range(v.shape[1]):
  if np.isfinite(v[i,j]):ax.text(j,i,f'{v[i,j]:.2f}',ha='center',va='center',fontsize=6.5,color='white' if abs(v[i,j])>0.45 else '#222222')
ax.set_xticks(range(len(cols)),[c.replace('Meylan ccRCC test','Meylan\nccRCC test').replace(' (','\n(') for c in cols],fontsize=7);ax.xaxis.tick_top();ax.set_yticks(range(len(t)),lab,fontsize=7)
ax.axvline(3.5,color='#333333',lw=1);ax.tick_params(length=0)
for s in ax.spines.values():s.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02);cb.set_label('Spearman ρ (measured vs H&E-predicted)',fontsize=7.5);cb.ax.tick_params(labelsize=7)
fig.text(0.01,0.005,'Breast | Kidney. ST: median over held-out patients of within-patient spot ρ; bulk: patient-level ρ (TCGA cross-fitted, CPTAC external).\n† fewer than 4 marker genes in the shared 9,036-gene background. Sorted by mean breast ST readout.',fontsize=6.5,color='#444444')
fig.tight_layout(rect=(0,0.03,1,1))
for ext in ['png','pdf','svg']:fig.savefig(O/f'Fig2d_program_atlas.{ext}',dpi=300)
print(t.round(3).to_string())
