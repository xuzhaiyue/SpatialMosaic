"""T10 数据：主模型在全部 ST 切片上的 spot 预测与空间量（Moran's I、共定位），口径与 T5 t5_st_bulk.py 完全相同。
自检：留出 / 外部切片的切片内 Spearman 必须与 results/t7/ST_section_all_candidates.csv 中主模型的值一致（差 < 1e-9）；
T4 一侧的 Moran / 共定位必须与 T5 表逐项一致。"""
from t10_common import *
import h5py,anndata as ad
from t5_common import moran,coloc,PAIRS
T4PRED=E/'t4/predictions'
def edges_for(row):
 a=ad.read_h5ad(row.rna,backed='r');rc=a.obs[['array_row','array_col']].to_numpy();ids=a.obs_names.astype(str).to_numpy();a.file.close()
 lookup={tuple(v):i for i,v in enumerate(rc)};e=[];offsets=[(0,2),(1,1),(1,-1)] if row.cohort=='Meylan_ccRCC' else [(1,0),(0,1)]
 for i,v in enumerate(rc):
  for s in offsets:
   j=lookup.get(tuple(v+s))
   if j is not None:e.append((i,j))
 return np.asarray(e,dtype=int).reshape(-1,2),ids
if __name__=='__main__':
 meta=manifest();models={k:main_model(k) for k in MAIN};spots=[];sec=[];mor=[];col=[]
 for row in meta.itertuples():
  with h5py.File(row.features) as f:x=f['features'][:].astype('float64');sid=f['spot_ids'].asstr()[:];xy=f['coordinates'][:]
  with np.load(row.targets,allow_pickle=True) as t:y=t['scores'];assert np.array_equal(sid,t['spot_ids'])
  pm=predict(models[ST2TCGA[row.cohort]],x,'ST')
  with np.load(T4PRED/(row.sample+'.npz')) as t4:p4=t4['predicted'];assert np.allclose(t4['observed'],y)
  e,ids=edges_for(row);assert np.array_equal(ids,sid)
  im,i4,io=moran(pm,e),moran(p4,e),moran(y,e);cm,c4,co=coloc(pm),coloc(p4),coloc(y)
  base=dict(cohort=row.cohort,patient_key=row.patient_key,patient=row.patient,sample=row.sample,split=row.split)
  for j,p in enumerate(P):
   sec.append({**base,'program':p,'spots':len(y),'rho_main':corr(y[:,j],pm[:,j]),'rho_T4':corr(y[:,j],p4[:,j])})
   mor.append({**base,'program':p,'observed_moran':io[j],'main_moran':im[j],'T4_moran':i4[j]})
  for (a,b),vo,vm,v4 in zip(PAIRS,co,cm,c4):col.append({**base,'program_A':P[a],'program_B':P[b],'observed_coloc':vo,'main_coloc':vm,'T4_coloc':v4})
  spots.append(pd.DataFrame({'sample':row.sample,'spot_id':sid,'x':xy[:,0],'y':xy[:,1],**{f'obs_{k}':y[:,j] for j,k in enumerate(SK)},**{f'main_{k}':pm[:,j] for j,k in enumerate(SK)},**{f'T4_{k}':p4[:,j] for j,k in enumerate(SK)}}))
  print('ST',row.sample,flush=True)
 sec=pd.DataFrame(sec);mor=pd.DataFrame(mor);col=pd.DataFrame(col)
 pd.concat(spots,ignore_index=True).to_parquet(D/'st_spot_predictions.parquet',index=False);sec.to_csv(D/'st_section_rho.csv',index=False);mor.to_csv(D/'st_moran_by_section.csv',index=False);col.to_csv(D/'st_coloc_by_section.csv',index=False)
 # 自检 1：主模型留出切片 ρ 与 T7 表一致
 t7=pd.read_csv(R/'results/t7/ST_section_all_candidates.csv');t7=t7[t7.apply(lambda r:r.candidate==MAIN[ST2TCGA[r.cohort]],axis=1)];chk=sec.merge(t7[['sample','program','rho']],on=['sample','program'],validate='one_to_one')
 d1=float((chk.rho_main-chk.rho).abs().max());assert len(chk)==len(sec[sec.split.isin(['test','external'])]) and d1<1e-9,d1
 # 自检 2：T4 一侧 Moran 与共定位与 T5 表一致
 t5=pd.read_csv(R/'results/t5/ST_moran_by_slice.csv').merge(mor,on=['sample','program']);d2=float(max((t5.observed_moran_x-t5.observed_moran_y).abs().max(),(t5.predicted_moran-t5.T4_moran).abs().max()))
 t5c=pd.read_csv(R/'results/t5/ST_colocalization_by_slice.csv').merge(col,on=['sample','program_A','program_B']);d3=float(max((t5c.observed_coloc_x-t5c.observed_coloc_y).abs().max(),(t5c.predicted_coloc-t5c.T4_coloc).abs().max()))
 assert d2<1e-9 and d3<1e-9,(d2,d3)
 (D/'st_selfcheck.json').write_text(json.dumps({'time':stamp(),'heldout_sections_checked':int(len(chk)//4),'max_diff_rho_vs_T7':d1,'max_diff_moran_vs_T5':d2,'max_diff_coloc_vs_T5':d3},indent=2));print('SELFCHECK OK',d1,d2,d3,flush=True)
