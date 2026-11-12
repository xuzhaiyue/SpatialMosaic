"""T9-2：换骨干提取 ST spot 特征（CONCH / DenseNet-121），裁剪链与 T4 完全相同，可断点续跑。
裁剪：128 µm 视野、round(center-size/2)、越界补白、RGBA 叠白底转 RGB（同 t4_features_discovery.py）。
预处理：DenseNet-121 与 UNI2-h 相同（PIL bicubic 256 → bilinear 224 中心裁剪，ImageNet 归一化）；
CONCH 用官方 create_model_from_pretrained 返回的 preprocess（448），特征为 encode_image(proj_contrast=False, normalize=False)。
用 smbackbone 环境运行（hemorph 克隆 + conch 代码包），nice 19、2 线程、小批量、批间暂停。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TORCH_HOME='/Volumes/INTENSO/SpatialMosaic_data/03_references/torch_hub')
import json,time,argparse,datetime,subprocess,shlex
from pathlib import Path
import numpy as np,pandas as pd,h5py,torch,openslide
from PIL import Image
from uni2h_inference import prepare
R=Path(__file__).resolve().parents[2];O=R/'results/t9/backbones';E=Path('/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts/t9/backbones')
CONCH='/Users/Zhaiyue/.cache/huggingface/hub/models--MahmoodLab--conch/snapshots/f9ca9f877171a28ade80228fb195ac5d79003357/pytorch_model.bin'
DIM={'densenet121':1024,'conch':512}
def stamp():return datetime.datetime.now().astimezone().isoformat()

def busy_gpu():
 # 其他 GPU 推理进程存在时不使用 MPS
 rows=[]
 for line in subprocess.check_output(['ps','-axo','pid=,%cpu=,command='],text=True).splitlines():
  try:pid,cpu,cmd=line.strip().split(None,2)
  except ValueError:continue
  if int(pid)==os.getpid():continue
  low=cmd.lower();names=[Path(x).name for x in cmd.split() if x.endswith('.py')]
  if 'cellvit' in low or 'ray::' in low or any(n.startswith('sr0') and 'server' not in n for n in names):rows.append({'pid':int(pid),'cpu':float(cpu),'cmd':cmd[:120]})
 return rows

def load_backbone(name,device):
 torch.set_num_threads(2)
 if name=='densenet121':
  import torchvision
  net=torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
  class Pool(torch.nn.Module):
   # ST-Net 骨干：features → ReLU → 全局平均池化，1024 维
   def __init__(s,m):super().__init__();s.f=m.features
   def forward(s,x):return torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.relu(s.f(x)),1).flatten(1)
  model=Pool(net).eval().to(device);tf=lambda im:prepare(im,'pil_bicubic')
 elif name=='conch':
  from conch.open_clip_custom import create_model_from_pretrained
  m,pre=create_model_from_pretrained('conch_ViT-B-16',checkpoint_path=CONCH,device=device);m.eval()
  class Enc(torch.nn.Module):
   def __init__(s,m):super().__init__();s.m=m
   def forward(s,x):return s.m.encode_image(x,proj_contrast=False,normalize=False)
  model=Enc(m).eval();tf=lambda im:pre(im).unsqueeze(0)
 else:raise ValueError(name)
 return model,tf

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backbone',required=True,choices=list(DIM));ap.add_argument('--device',default='cpu',choices=['cpu','mps']);ap.add_argument('--samples',nargs='*');ap.add_argument('--limit',type=int);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--pause',type=float,default=0.5);ap.add_argument('--out',default=None)
 a=ap.parse_args();meta=pd.read_csv(R/'results/t4/input_manifest.csv');field=json.loads((R/'config/t4_frozen.json').read_text())['field_um'];assert field==128.0
 if a.samples:meta=meta[meta['sample'].isin(a.samples)];assert len(meta)==len(a.samples)
 out=Path(a.out) if a.out else E/a.backbone/'features';out.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
 model,tf=load_backbone(a.backbone,a.device);status=O/f'extraction_status_{a.backbone}.json';total=int(meta.spots.sum());done_prior=0;t0=time.time();n_new=0
 for row in meta.sort_values(['cohort','patient_key','sample']).itertuples():
  # 只读 RNA 的条码与坐标，不读表达值
  import anndata as ad
  ann=ad.read_h5ad(row.rna,backed='r');ids=ann.obs_names.astype(str).to_numpy();xy=np.asarray(ann.obsm['spatial']);ann.file.close();assert len(ids)==row.spots
  with h5py.File(row.features) as t4:assert np.array_equal(t4['spot_ids'].asstr()[:],ids) and np.array_equal(t4['coordinates'][:],xy);t4_size=int(t4.attrs['crop_pixels'])
  size=round(field/row.mpp);assert size==t4_size;n=len(ids) if a.limit is None else min(a.limit,len(ids));slide=openslide.OpenSlide(row.image);dest=out/(row.sample+'.h5')
  with h5py.File(dest,'a') as f:
   if 'features' not in f:
    f.attrs.update(backbone=a.backbone,image=row.image,mpp=row.mpp,crop_pixels=size,device=a.device,n=n)
    f.create_dataset('features',(n,DIM[a.backbone]),dtype='float32');f.create_dataset('done',data=np.zeros(n,dtype=bool));f.create_dataset('spot_ids',data=np.asarray(ids[:n],dtype=object),dtype=h5py.string_dtype());f.create_dataset('coordinates',data=xy[:n])
   else:assert f.attrs['backbone']==a.backbone and int(f.attrs['n'])==n and np.array_equal(f['spot_ids'].asstr()[:],ids[:n])
   for start in range(0,n,a.batch):
    ix=[i for i in range(start,min(n,start+a.batch)) if not f['done'][i]]
    if not ix:continue
    while a.device=='mps' and busy_gpu():
     status.write_text(json.dumps({'date':stamp(),'phase':'paused_for_other_gpu','sample':row.sample,'conflicts':busy_gpu()}));time.sleep(30)
    tensors=[]
    for i in ix:
     x,y=np.rint(xy[i]-size/2).astype(int);raw=slide.read_region((int(x),int(y)),0,(size,size));image=Image.alpha_composite(Image.new('RGBA',raw.size,(255,255,255,255)),raw).convert('RGB');tensors.append(tf(image))
    with torch.inference_mode():values=model(torch.cat(tensors).to(a.device)).float().cpu().numpy()
    assert values.shape==(len(ix),DIM[a.backbone]) and np.isfinite(values).all()
    for k,i in enumerate(ix):f['features'][i]=values[k]
    f.flush()
    for i in ix:f['done'][i]=True
    f.flush();n_new+=len(ix);status.write_text(json.dumps({'date':stamp(),'phase':'extracting','backbone':a.backbone,'device':a.device,'sample':row.sample,'sample_done':int(f['done'][:].sum()),'sample_total':n,'prior_spots':done_prior,'total_spots':total,'sec_per_spot_this_run':(time.time()-t0)/max(n_new,1)}));time.sleep(a.pause)
   assert f['done'][:].all();f.attrs['complete']=True
  slide.close();done_prior+=n;print('DONE',a.backbone,row.sample,done_prior,'/',total,f'{(time.time()-t0)/max(n_new,1):.3f}s/spot',flush=True)
 print('ALL DONE',a.backbone,flush=True)
if __name__=='__main__':main()
