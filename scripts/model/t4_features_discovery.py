"""Resumable UNI2-h feature extraction for ST spots on MPS."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
import json,time,hashlib,subprocess,shlex,fcntl,datetime
from pathlib import Path
import numpy as np,pandas as pd,h5py,torch,openslide
from PIL import Image
from uni2h_inference import load_encoder,prepare
from t4_display_provenance import allowed_hashes
R=Path(__file__).resolve().parents[2];W=R/'docs/worklog/2026-10-26_T4';O=R/'results/t4'
def stamp():return datetime.datetime.now().astimezone().isoformat()
def gpu_snapshot():
 rows=[]
 for line in subprocess.check_output(['ps','-axo','pid=,%cpu=,command='],text=True).splitlines():
  try:pid,cpu,cmd=line.strip().split(None,2);parts=shlex.split(cmd)
  except ValueError:continue
  if not parts or 'python' not in parts[0] or int(pid)==os.getpid():continue
  names=[Path(x).name for x in parts[1:] if x.endswith('.py')]
  if any('cellvit' in n.lower() or ('sr_' in n and 'pipeline' in n) or n in ['t4_features.py','t3b_functional_check_mps.py'] for n in names):rows.append({'pid':int(pid),'cpu':float(cpu),'scripts':names})
 return rows

def main():
 cfg=json.loads((R/'config/t4_frozen.json').read_text());plan=(R/'PLAN.md').read_bytes();assert plan.splitlines()[0]==b'status: frozen_2026-10-26';plan_sha=hashlib.sha256(plan).hexdigest();cfg_sha=hashlib.sha256((R/'config/t4_frozen.json').read_bytes()).hexdigest();meta=pd.read_csv(R/cfg['input_manifest']);torch.set_num_threads(2);allowed_plans,allowed_configs=allowed_hashes()
 lock=open('/tmp/zhaiyue_spatialmosaic_mps.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
 while True:
  snapshot=gpu_snapshot();active=[x for x in snapshot if any('cellvit' in n.lower() for n in x['scripts']) or x['cpu']>1]
  # Idle resident servers/pipelines are recorded; active external inference blocks MPS.
  status={'date':stamp(),'phase':'waiting_for_gpu' if active else 'gpu_preflight_ready','other_gpu_processes':snapshot,'active_conflicts':active,'plan_sha256':plan_sha}
  (O/'extraction_status.json').write_text(json.dumps(status,indent=2))
  if not active:
   util=subprocess.run(['ioreg','-r','-c','AGXAccelerator','-l'],capture_output=True,text=True).stdout
   import re
   usage=[int(x) for x in re.findall(r'"Device Utilization %"=(\d+)',util)]
   status['device_utilization']=usage
   with (W/'gpu_preflight.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
   # The resident SR pipeline is only considered inactive after idle GPU samples.
   if not snapshot or (usage and max(usage)<=3):break
  print('WAIT GPU',json.dumps(snapshot),flush=True);time.sleep(30)
 device='mps';assert torch.backends.mps.is_available();model=load_encoder(cfg['weights']).to(device);completed=0
 for row in meta.sort_values(['cohort','patient_key','sample']).itertuples():
  # RNA is opened only for barcode/coordinate metadata; feature extraction does not read expression values.
  import anndata as ad
  a=ad.read_h5ad(row.rna,backed='r');ids=a.obs_names.astype(str).to_numpy();xy=np.asarray(a.obsm['spatial']);a.file.close();assert len(ids)==row.spots
  size=round(cfg['field_um']/row.mpp);slide=openslide.OpenSlide(row.image);dest=Path(row.features);dest.parent.mkdir(parents=True,exist_ok=True)
  with h5py.File(dest,'a') as f:
   if 'features' not in f:
    f.attrs.update(config_sha256=cfg_sha,plan_sha256=plan_sha,image=row.image,mpp=row.mpp,crop_pixels=size,device=device,precision=cfg['precision'])
    f.create_dataset('features',(len(ids),1536),dtype='float32');f.create_dataset('done',data=np.zeros(len(ids),dtype=bool));f.create_dataset('spot_ids',data=np.asarray(ids,dtype=object),dtype=h5py.string_dtype());f.create_dataset('coordinates',data=xy);f.create_dataset('tissue_coverage',(len(ids),),dtype='float32');f.create_dataset('in_image_fraction',(len(ids),),dtype='float32')
   else:
    assert f.attrs['config_sha256'] in allowed_configs and f.attrs['plan_sha256'] in allowed_plans;assert np.array_equal(f['spot_ids'].asstr()[:],ids);assert np.array_equal(f['coordinates'][:],xy)
   for start in range(0,len(ids),cfg['batch_size']):
    ix=[i for i in range(start,min(len(ids),start+cfg['batch_size'])) if not f['done'][i]]
    if not ix:continue
    while True:
     conflicts=[x for x in gpu_snapshot() if any('cellvit' in n.lower() for n in x['scripts']) or x['cpu']>1]
     if not conflicts:break
     (O/'extraction_status.json').write_text(json.dumps({'date':stamp(),'phase':'paused_for_other_gpu','sample':row.sample,'conflicts':conflicts}));time.sleep(30)
    tensors=[];coverage=[];valid=[]
    for i in ix:
     x,y=np.rint(xy[i]-size/2).astype(int);raw=slide.read_region((int(x),int(y)),0,(size,size));white=Image.new('RGBA',raw.size,(255,255,255,255));image=Image.alpha_composite(white,raw).convert('RGB');small=image.resize((256,256),Image.Resampling.BICUBIC);rgb=np.asarray(small,dtype=np.float64)/255;mx=rgb.max(2);mn=rgb.min(2);sat=(mx-mn)/np.maximum(mx,1e-8);coverage.append(float(np.mean((sat>.05)&(mx<.98))));valid.append(max(0,min(x+size,slide.dimensions[0])-max(x,0))*max(0,min(y+size,slide.dimensions[1])-max(y,0))/size**2);tensors.append(prepare(image,'pil_bicubic'))
    with torch.inference_mode(),torch.autocast('mps',dtype=torch.bfloat16):values=model(torch.cat(tensors).to('mps')).float().cpu().numpy()
    torch.mps.synchronize();assert np.isfinite(values).all()
    for k,i in enumerate(ix):f['features'][i]=values[k];f['tissue_coverage'][i]=coverage[k];f['in_image_fraction'][i]=valid[k]
    f.flush()
    for i in ix:f['done'][i]=True
    f.flush();count=int(f['done'][:].sum());(O/'extraction_status.json').write_text(json.dumps({'date':stamp(),'phase':'extracting','sample':row.sample,'sample_done':count,'sample_total':len(ids),'completed_prior_slides_spots':completed,'total_spots':int(meta.spots.sum()),'device':'mps','batch':cfg['batch_size'],'threads':2,'pause_seconds':cfg['pause_seconds']}));time.sleep(cfg['pause_seconds'])
   assert f['done'][:].all();features=f['features'][:];assert np.isfinite(features).all();f.attrs['complete']=True;f.flush()
  slide.close();completed+=len(ids);print('FEATURE COMPLETE',row.sample,completed,flush=True)
 (O/'features_complete.json').write_text(json.dumps({'complete':True,'spots':completed,'slides':len(meta),'plan_sha256':plan_sha,'config_sha256':cfg_sha}));print('ALL FEATURES COMPLETE',flush=True)
if __name__=='__main__':main()
