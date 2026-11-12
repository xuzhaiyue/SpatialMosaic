"""Fig3 ST spatial readouts and fixed-background bulk UCell without new fitting."""
from t5_common import *
import anndata as ad,h5py,gzip,re,itertools
from scipy.stats import spearmanr
if not (O/'He2020_between_section_patterns.csv').exists():
 meta=pd.read_csv(R/C['T4_manifest']);st=[];pairs=[]
 for row in meta.itertuples():
  d=np.load(Path(C['T4_prediction_root'])/(row.sample+'.npz'),allow_pickle=False);a=ad.read_h5ad(row.rna,backed='r');rc=a.obs[['array_row','array_col']].to_numpy();a.file.close();lookup={tuple(v):i for i,v in enumerate(rc)};edges=[];offsets=[(0,2),(1,1),(1,-1)] if row.cohort=='Meylan_ccRCC' else [(1,0),(0,1)]
  for i,v in enumerate(rc):
   for shift in offsets:
    j=lookup.get(tuple(v+shift))
    if j is not None:edges.append((i,j))
  edges=np.asarray(edges,dtype=int).reshape(-1,2);actual=d['observed'];pred=d['predicted'];Ia=moran(actual,edges);Ip=moran(pred,edges);ca=coloc(actual);cp=coloc(pred)
  for j,p in enumerate(P):st.append({'cohort':row.cohort,'patient_key':row.patient_key,'patient':row.patient,'sample':row.sample,'split':row.split,'program':p,'observed_moran':Ia[j],'predicted_moran':Ip[j],'spots':len(rc)})
  for (i,j),va,vp in zip(PAIRS,ca,cp):pairs.append({'cohort':row.cohort,'patient_key':row.patient_key,'sample':row.sample,'split':row.split,'program_A':P[i],'program_B':P[j],'observed_coloc':va,'predicted_coloc':vp})
 pd.DataFrame(st).to_csv(O/'ST_moran_by_slice.csv',index=False);patient=pd.DataFrame(st).groupby(['cohort','patient_key','split','program'],as_index=False)[['observed_moran','predicted_moran']].mean();patient.to_csv(O/'ST_moran_by_patient.csv',index=False);pd.DataFrame(pairs).to_csv(O/'ST_colocalization_by_slice.csv',index=False);pd.DataFrame(pairs).groupby(['cohort','patient_key','split','program_A','program_B'],as_index=False)[['observed_coloc','predicted_coloc']].mean().to_csv(O/'ST_colocalization_by_patient.csv',index=False)
 # Compare scalar spatial patterns across same-patient sections; do not assume pixel registration.
 source=pd.read_csv(R/'results/t0/hest_manifest.csv').set_index('id');repeat=[]
 for (patient_id,program),g in pd.DataFrame(st).query("cohort=='He2020'").groupby(['patient','program']):
  for a,b in itertools.combinations(g.to_dict('records'),2):
   label_a=str(source.loc[a['sample'],'subseries']);label_b=str(source.loc[b['sample'],'subseries']);ma=re.search(r'_([A-Z]+)(\d+)$',label_a);mb=re.search(r'_([A-Z]+)(\d+)$',label_b);same_block=bool(ma and mb and ma.group(1)==mb.group(1));consecutive=bool(same_block and abs(int(ma.group(2))-int(mb.group(2)))==1)
   repeat.append({'patient':patient_id,'program':program,'sample_A':a['sample'],'sample_B':b['sample'],'source_label_A':label_a,'source_label_B':label_b,'same_named_block':same_block,'consecutive_numbered_within_block':consecutive,'observed_A':a['observed_moran'],'observed_B':b['observed_moran'],'predicted_A':a['predicted_moran'],'predicted_B':b['predicted_moran']})
 pd.DataFrame(repeat).to_csv(O/'He2020_between_section_patterns.csv',index=False);print('ST SPATIAL COMPLETE',flush=True)
bg=(R/C['background']).read_text().splitlines();bgset=set(bg);members=json.loads((R/C['members']).read_text());mapping=json.loads((R/C['symbol_mapping']).read_text());scores=[]
for cohort,name in [('TCGA-BRCA','BRCA_HiSeqV2.gz'),('TCGA-KIRC','KIRC_HiSeqV2.gz')]:
 table=pd.read_csv(Path(C['reference_root'])/name,sep='\t',index_col=0);ids=[mapping.get(str(g)) for g in table.index];freq=pd.Series(ids).value_counts();index={g:i for i,g in enumerate(ids) if g in bgset and freq[g]==1};assert set(index)==set(bg);cols=[c for c in table.columns if c.startswith('TCGA-') and c.split('-')[3][:2]=='01'];values=table.iloc[[index[g] for g in bg]][cols].to_numpy(dtype='float64').T;sc=ucell(values,bg,members)
 for name,v in zip(cols,sc):scores.append({'cohort':cohort,'patient_id':name[:12],'RNA_sample':name,**dict(zip(MEANS,v))})
 print('BULK',cohort,len(cols),flush=True)
files=pd.read_csv(R/'docs/worklog/2026-10-26_T3/download_manifest.csv');sample=pd.read_csv(R/'results/t3/CPTAC_RNA_sample_manifest.csv');assert sample.groupby('file_id').patient.nunique().max()==1;sample=sample.groupby(['file_id','file','cohort','patient'],as_index=False).agg(sample=('sample',lambda x:';'.join(sorted(set(x)))));sample.to_csv(O/'CPTAC_unique_RNA_file_references.csv',index=False);sample=sample.merge(files[['version','destination_file']],left_on='file_id',right_on='version',validate='one_to_one')
for row in sample.itertuples():
 table=pd.read_csv(row.destination_file,sep='\t',comment='#');table['id']=table.gene_id.str.split('.').str[0];freq=table.id.value_counts();lookup={g:i for i,g in enumerate(table.id) if g in bgset and freq[g]==1};assert set(lookup)==set(bg);v=table.iloc[[lookup[g] for g in bg]][C['CPTAC_bulk_column']].to_numpy(dtype=float);sc=ucell(v[None,:],bg,members)[0];scores.append({'cohort':row.cohort,'patient_id':row.patient,'RNA_sample':row.sample,**dict(zip(MEANS,sc))})
all_scores=pd.DataFrame(scores);all_scores.to_csv(O/'bulk_UCell_by_sample.csv',index=False);patient=all_scores.groupby(['cohort','patient_id'],as_index=False)[MEANS].mean();patient.to_csv(O/'bulk_UCell_by_patient.csv',index=False);print('BULK COMPLETE',len(patient),flush=True)
