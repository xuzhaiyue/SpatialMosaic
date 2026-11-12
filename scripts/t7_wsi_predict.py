"""Apply only validation-selected models; reuse cached endpoint/decomposed maps when exact."""
from t7_common import *
from t7_fit import candidate_predict
import h5py,joblib,tarfile,io,shutil,time
manifest=pd.read_csv(R/'results/t6/wsi_prediction_manifest.csv');by={(r.archive,r.slide):r for r in manifest.itertuples()};selection=pd.read_csv(O/'candidate_validation_selection.csv');selection=selection[selection.selected];registry=json.loads((O/'model_registry.json').read_text());registry={(r['cohort'],r['fold'],r['candidate']):r for r in registry};folds={(r.cohort,r.patient_id):r.fold for r in pd.read_csv(R/'results/t6/TCGA_patient_folds.csv').itertuples()};selected={(r.cohort,r.outer_fold):r.candidate for r in selection.itertuples()};loaded={};outrecords=[]
with h5py.File(E6/'slide_feature_means.h5') as means:
 for arc in C['archives']:
  with tarfile.open(arc['path'],'r|gz') as tar:
   for member in tar:
    name=Path(member.name).name
    if not member.isfile() or not name.endswith('.h5') or name.startswith('.'):continue
    row=by.get((arc['name'],name[:-3]))
    if row is None:continue
    tc='TCGA-BRCA' if 'BRCA' in row.cohort else 'TCGA-KIRC';fold=folds[row.cohort,row.patient_id] if row.cohort.startswith('TCGA') else 0;key=selected[tc,fold];rec=registry[tc,fold,key];model=loaded.setdefault((tc,fold),joblib.load(rec['path']));dest=E/'patch_predictions'/row.cohort/(row.slide+'.h5');dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():
     with h5py.File(row.path) as old:xy=old['coords_top_left'][:];step=int(old.attrs['grid_step'])
     n=len(xy);part=dest.with_suffix('.partial.h5')
     if key in ['lambda_0.5','lambda_1','decomposed']:
      src=Path(row.path) if key=='lambda_0.5' else E5/'patch_predictions'/row.cohort/(row.slide+'.h5')
      with h5py.File(src) as f:v=f['predicted'][:]
      if key=='decomposed':v=v-v.mean(0)+candidate_predict(model,means[row.cohort+'/'+row.slide][:][None,:],'bulk')
      with h5py.File(part,'w') as f:f.create_dataset('predicted',data=v);f.create_dataset('coords_top_left',data=xy);f.attrs.update(complete=True,grid_step=step,candidate=key,fold=fold,model_sha256=rec['sha256'])
     else:
      stream=tar.extractfile(member);temp=None
      if member.size<=512*1024**2:payload=io.BytesIO(stream.read())
      else:
       d=Path(C['large_temp_root']);d.mkdir(exist_ok=True,parents=True);temp=d/(row.slide+'_T7.h5')
       with temp.open('wb') as f:shutil.copyfileobj(stream,f,4*1024**2)
       payload=temp
      with h5py.File(payload) as raw,h5py.File(part,'a') as f:
       data=raw['features'];assert data.shape[-2]==n
       if 'predicted' not in f:f.create_dataset('predicted',(n,4),dtype='float64');f.create_dataset('coords_top_left',data=xy);f.attrs.update(rows_done=0,complete=False,grid_step=step,candidate=key,fold=fold,model_sha256=rec['sha256'])
       for l in range(int(f.attrs['rows_done']),n,C['prediction_batch']):
        h=min(n,l+C['prediction_batch']);x=(data[0,l:h] if data.ndim==3 else data[l:h]).astype('float64');v=candidate_predict(model,x,'bulk');assert np.isfinite(v).all();f['predicted'][l:h]=v;f.flush();f.attrs['rows_done']=h;f.flush();time.sleep(C['pause_seconds'])
       f.attrs['complete']=True;f.flush()
      if hasattr(payload,'close'):payload.close()
      if temp:temp.unlink()
     part.replace(dest)
    with h5py.File(dest) as f:assert f.attrs['complete'];n=len(f['predicted']);step=int(f.attrs['grid_step'])
    outrecords.append(dict(cohort=row.cohort,patient_id=row.patient_id,slide=row.slide,patches=n,path=str(dest),archive=arc['name'],grid_step=step,candidate=key,fold=fold));status(phase='selected_WSI',slides_done=len(outrecords),slides_total=3081);print('WSI',len(outrecords),key,row.slide,flush=True)
assert len(outrecords)==3081;pd.DataFrame(outrecords).to_csv(O/'wsi_prediction_manifest.csv',index=False);(W/'WSI_complete.json').write_text(json.dumps({'complete':True,'slides':3081,'time':stamp()}));print('WSI COMPLETE',flush=True)
