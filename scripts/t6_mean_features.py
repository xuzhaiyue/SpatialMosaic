"""One streaming pass obtains exact per-slide and equal-slide patient feature means."""
from t6_common import *
import tarfile,io,h5py,shutil,time
inventory=pd.read_csv(R/C['feature_inventory']);lookup={(r.archive,r.slide):r for r in inventory.itertuples()};cache=E/'slide_feature_means.h5';records=[]
with h5py.File(cache,'a') as output:
 for archive in C['archives']:
  with tarfile.open(archive['path'],'r|gz') as tar:
   for member in tar:
    name=Path(member.name).name
    if not member.isfile() or not name.endswith('.h5') or name.startswith('.'):continue
    slide=name[:-3];row=lookup.get((archive['name'],slide))
    if row is None:continue
    key=row.cohort+'/'+slide
    if key not in output:
     stream=tar.extractfile(member);temp=None
     if member.size<=512*1024**2:payload=io.BytesIO(stream.read())
     else:
      directory=Path(C['large_temp_root']);directory.mkdir(parents=True,exist_ok=True);temp=directory/(slide+'_T6mean.h5')
      with temp.open('wb') as f:shutil.copyfileobj(stream,f,4*1024**2)
      payload=temp
     with h5py.File(payload,'r') as raw:
      data=raw['features'];n=data.shape[-2];total=np.zeros(1536,dtype='float64')
      for start in range(0,n,C['prediction_batch']):
       x=data[0,start:start+C['prediction_batch']] if data.ndim==3 else data[start:start+C['prediction_batch']];total+=x.sum(0,dtype='float64');time.sleep(C['pause_seconds'])
      item=output.create_dataset(key,data=total/n);item.attrs.update(cohort=row.cohort,patient_id=row.patient_id,slide=slide,n_patches=n)
     if hasattr(payload,'close'):payload.close()
     if temp:temp.unlink()
     output.flush()
    item=output[key];records.append({'cohort':row.cohort,'patient_id':row.patient_id,'slide':slide,'n_patches':int(item.attrs['n_patches']),'h5_key':key});status('mean_feature_status.json',phase='mean_features',slides_done=len(records),slides_total=C['expected_slides']);print('MEAN',len(records),slide,flush=True)
 assert len(records)==C['expected_slides'];df=pd.DataFrame(records);df.to_csv(O/'slide_mean_feature_index.csv',index=False);patient=[];arrays=[]
 for (cohort,pid),g in df.groupby(['cohort','patient_id'],sort=True):
  arrays.append(np.mean([output[k][:] for k in g.h5_key],axis=0));patient.append({'cohort':cohort,'patient_id':pid,'n_slides':len(g),'row':len(arrays)-1})
 np.save(E/'patient_mean_features.npy',np.vstack(arrays));pd.DataFrame(patient).to_csv(O/'patient_mean_feature_index.csv',index=False)
status('mean_feature_status.json',phase='complete',slides=len(records),patients=len(patient));print('MEAN FEATURES COMPLETE',flush=True)
