"""T9-1 第一步：取 PAM50 基因的 log2 表达，每患者一份（原发肿瘤样本取均值）。只读已有 RNA。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'results/t9/pam50';O.mkdir(parents=True,exist_ok=True)
REF=Path('/Volumes/INTENSO/SpatialMosaic_data/03_references/tcga_endpoints_rna')
# genefu pam50 centroid 名称（旧符号）；表达表里的现行符号 -> 质心名
PAM50=['ACTR3B','ANLN','BAG1','BCL2','BIRC5','BLVRA','CCNB1','CCNE1','CDC20','CDC6','CDCA1','CDH3','CENPF','CEP55','CXXC5','EGFR','ERBB2','ESR1','EXO1','FGFR4','FOXA1','FOXC1','GPR160','GRB7','KIF2C','KNTC2','KRT14','KRT17','KRT5','MAPT','MDM2','MELK','MIA','MKI67','MLPH','MMP11','MYBL2','MYC','NAT1','ORC6L','PGR','PHGDH','PTTG1','RRM2','SFRP1','SLC39A6','TMEM45B','TYMS','UBE2C','UBE2T']
ALIAS={'NUF2':'CDCA1','NDC80':'KNTC2','ORC6':'ORC6L'}
def to_centroid(symbols):return [ALIAS.get(s,s) for s in symbols]
rows=[]
# TCGA：Xena HiSeqV2，已是 log2(norm+1)；原发肿瘤 = 样本类型 01
x=pd.read_csv(REF/'BRCA_HiSeqV2.gz',sep='\t',index_col=0);x.index=to_centroid(x.index.astype(str))
assert not x.index.duplicated().any()
cols=[c for c in x.columns if c.startswith('TCGA-') and c.split('-')[3][:2]=='01']
t=x.loc[[g for g in PAM50 if g in x.index],cols].T;t['patient_id']=[c[:12] for c in cols]
tcga=t.groupby('patient_id').mean();rows.append(dict(cohort='TCGA-BRCA',samples=len(cols),patients=len(tcga),genes=tcga.shape[1],missing=';'.join(g for g in PAM50 if g not in tcga.columns)))
# CPTAC：STAR tpm_unstranded，log2(x+1)；原发肿瘤；患者号与 T5/T6 相同（manifest.patient）
files=pd.read_csv(R/'docs/worklog/2026-10-26_T3/download_manifest.csv');sample=pd.read_csv(R/'results/t3/CPTAC_RNA_sample_manifest.csv')
sample=sample[sample.cohort.eq('CPTAC-BRCA')&sample.sample_type.eq('Primary Tumor')].drop_duplicates('file_id').merge(files[['version','destination_file']],left_on='file_id',right_on='version',validate='one_to_one')
vals=[]
for r in sample.itertuples():
 tab=pd.read_csv(r.destination_file,sep='\t',comment='#');tab=tab[tab.gene_type.eq('protein_coding')&tab.gene_name.notna()]
 tab['g']=to_centroid(tab.gene_name.astype(str));tab=tab[tab.g.isin(PAM50)]
 assert not tab.g.duplicated().any(),r.destination_file
 v=np.log2(tab.set_index('g').tpm_unstranded.astype(float)+1);v.name=r.patient;vals.append(v)
c=pd.DataFrame(vals);c.index.name='patient_id';cptac=c.groupby(level=0).mean()
rows.append(dict(cohort='CPTAC-BRCA',samples=len(sample),patients=len(cptac),genes=cptac.shape[1],missing=';'.join(g for g in PAM50 if g not in cptac.columns)))
tcga.to_csv(O/'TCGA-BRCA_PAM50_log2_expression.csv');cptac.to_csv(O/'CPTAC-BRCA_PAM50_log2_expression.csv');pd.DataFrame(rows).to_csv(O/'PAM50_expression_inventory.csv',index=False)
print(pd.DataFrame(rows).to_string())
