"""T9-2 可选项：ST-Net 原方法（DenseNet-121 ImageNet 预训练、端到端微调），目标为四程序 UCell。
超参按 ST-Net 官方 README（bryanhe/ST-Net：--model densenet121 --pretrain --lr 1e-6 --batch 32 --norm --average，50 epoch，SGD momentum 0.9）。
与官方的差别（为与其他骨干公平比较，写明）：视野/裁剪链同 T4（128 µm → 256 bicubic → 224）；输出 4 个程序而非 250 基因；
训练时随机 90° 旋转 + 翻转；epoch 由验证集按 T4 同一指标（切片内 Spearman→患者内等权→验证患者等权，再四程序平均）选择，
只用 train 训练（神经网络不再 train+val 重拟合）；验证与测试都做 8 向旋转/翻转平均（--average）。
步骤：cache（裁剪成 224 uint8 存外盘）→ check（冻结骨干在缓存上重算 SPA149 前 32 个 spot，须与 T9-2 DenseNet 特征一致）→ train。
smbackbone 环境，nice 19、2 线程、MPS，对 SR 支线让路。"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TORCH_HOME='/Volumes/INTENSO/SpatialMosaic_data/03_references/torch_hub')
import json,time,argparse,datetime
from pathlib import Path
import numpy as np,pandas as pd,h5py,torch,torchvision,openslide
from PIL import Image
from t9_backbone_features import busy_gpu,load_backbone
from t9_backbone_fit import corr
R=Path(__file__).resolve().parents[2];O=R/'results/t9/backbones/stnet_finetune';E=Path('/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts/t9/stnet')
CFG=dict(lr=1e-6,momentum=0.9,batch=32,epochs=50,seed=20261027,pause_every=50,pause_seconds=0.2)
MEAN=torch.tensor([.485,.456,.406]).view(1,3,1,1);STD=torch.tensor([.229,.224,.225]).view(1,3,1,1)
def stamp():return datetime.datetime.now().astimezone().isoformat()
SHARE=dict(on=False,every=5,seconds=1.0)
def wait_gpu(status,what):
 # --share-gpu：与其他进程共享 MPS，每 5 批暂停 1 s；只影响速度，不影响结果
 if SHARE['on']:status.write_text(json.dumps({'date':stamp(),'phase':'training_shared_gpu','at':what,'others_on_gpu':len(busy_gpu())}));time.sleep(SHARE['seconds']);return
 while busy_gpu():status.write_text(json.dumps({'date':stamp(),'phase':'paused_for_other_gpu','at':what}));time.sleep(30)

def crop224(slide,center,size):
 # 与 uni2h_inference.prepare 的几何部分完全相同：PIL bicubic 256 → bilinear 224 → 中心裁剪；ToTensor 前的 uint8 图像可精确缓存
 x,y=np.rint(center-size/2).astype(int);raw=slide.read_region((int(x),int(y)),0,(size,size));im=Image.alpha_composite(Image.new('RGBA',raw.size,(255,255,255,255)),raw).convert('RGB').resize((256,256),Image.Resampling.BICUBIC)
 im=torchvision.transforms.Resize(224,interpolation=torchvision.transforms.InterpolationMode.BILINEAR)(im);im=torchvision.transforms.CenterCrop(224)(im);return np.asarray(im,dtype=np.uint8)

def cache(meta):
 import anndata as ad
 (E/'crops').mkdir(parents=True,exist_ok=True)
 for row in meta.itertuples():
  dest=E/'crops'/(row.sample+'.npy')
  if dest.exists():continue
  a=ad.read_h5ad(row.rna,backed='r');ids=a.obs_names.astype(str).to_numpy();xy=np.asarray(a.obsm['spatial']);a.file.close();assert len(ids)==row.spots
  with h5py.File(row.features) as f:assert np.array_equal(f['spot_ids'].asstr()[:],ids)
  slide=openslide.OpenSlide(row.image);size=round(128.0/row.mpp);arr=np.stack([crop224(slide,c,size) for c in xy]);slide.close()
  tmp=dest.with_suffix('.tmp.npy');np.save(tmp,arr);tmp.rename(dest);print('CACHED',row.sample,arr.shape,flush=True)

def check(meta):
 # 冻结 DenseNet 在缓存图上的特征须等于 T9-2 提取特征（同一预处理）
 model,_=load_backbone('densenet121','cpu');row=meta[meta['sample'].eq('SPA149')].iloc[0];x=np.load(E/'crops/SPA149.npy',mmap_mode='r')[:32]
 t=(torch.from_numpy(np.array(x)).permute(0,3,1,2).float()/255-MEAN)/STD
 with torch.inference_mode():v=model(t).numpy()
 ref=h5py.File('/Volumes/INTENSO/SpatialMosaic_data/02_derived/project_large_artifacts/t9/backbones/densenet121/features/SPA149.h5')['features'][:32];d=float(np.abs(v-ref).max());print('CHECK max abs diff',d,flush=True);assert d<1e-4

def tensors(crops,ix,device):return ((torch.from_numpy(crops[np.asarray(ix)]).to(device).permute(0,3,1,2).float()/255)-MEAN.to(device))/STD.to(device)
def predict(model,crops,device,batch=64):
 out=[]
 model.eval()
 with torch.inference_mode():
  for s in range(0,len(crops),batch):
   x=tensors(crops,range(s,min(len(crops),s+batch)),device);acc=0
   for k in range(4):
    for flip in (False,True):
     z=torch.rot90(x,k,(2,3));z=torch.flip(z,(3,)) if flip else z;acc=acc+model(z)
   out.append((acc/8).float().cpu().numpy())
 return np.vstack(out)

def load_rows(rows):
 crops=[];ys=[];meta=[]
 for row in rows.itertuples():
  with np.load(row.targets,allow_pickle=True) as t:
   with h5py.File(row.features) as f:assert np.array_equal(f['spot_ids'].asstr()[:],t['spot_ids'])
   c=np.load(E/'crops'/(row.sample+'.npy'));assert len(c)==len(t['scores']);crops.append(c);ys.append(t['scores']);meta.append(pd.DataFrame({'cohort':row.cohort,'patient_key':row.patient_key,'patient':row.patient,'sample':row.sample,'split':row.split},index=range(len(c))))
 # 缓存整体读入内存（外盘随机读慢；He2020 约 2.7 GB、Meylan 约 5.7 GB）
 return np.concatenate(crops),np.vstack(ys),pd.concat(meta,ignore_index=True)

def val_score(y,pred,meta,programs):
 rows=[]
 for (key,sample),ix in meta.groupby(['patient_key','sample']).groups.items():
  for j,p in enumerate(programs):rows.append({'patient_key':key,'program':p,'spearman':corr(y[ix,j],pred[ix,j])})
 m=pd.DataFrame(rows).groupby(['patient_key','program']).spearman.mean().reset_index().groupby('program').spearman.mean();return [float(m[p]) for p in programs]

def train(cohort,manifest,programs,device):
 out=O/cohort;out.mkdir(parents=True,exist_ok=True);status=O/'status.json';torch.manual_seed(CFG['seed']);rng=np.random.default_rng(CFG['seed'])
 ctr,ytr,_=load_rows(manifest[manifest.cohort.eq(cohort)&manifest.split.eq('train')]);cva,yva,mva=load_rows(manifest[manifest.cohort.eq(cohort)&manifest.split.eq('val')])
 mu=ytr.mean(0);sd=ytr.std(0);ztr=torch.from_numpy((ytr-mu)/sd).float()  # --norm：按训练集标准化目标
 net=torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1);net.classifier=torch.nn.Linear(1024,len(programs));net=net.to(device)
 opt=torch.optim.SGD(net.parameters(),lr=CFG['lr'],momentum=CFG['momentum']);log=[];best=(-np.inf,None);start=1;last=E/f'{cohort}_last.pt'
 if last.exists():
  # 断点续训：模型、优化器动量、两套随机数状态、日志与最佳 epoch 一并恢复
  ck=torch.load(last,map_location=device,weights_only=False);net.load_state_dict(ck['net']);opt.load_state_dict(ck['opt']);rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'].cpu());log=ck['log'];best=tuple(ck['best']);start=ck['epoch']+1;print('RESUME',cohort,'from epoch',start,flush=True)
 for epoch in range(start,CFG['epochs']+1):
  net.train();order=rng.permutation(len(ctr));t0=time.time();losses=[]
  for b,s in enumerate(range(0,len(order),CFG['batch'])):
   if b%(SHARE['every'] if SHARE['on'] else CFG['pause_every'])==0:wait_gpu(status,f'{cohort} epoch {epoch}');time.sleep(CFG['pause_seconds'])
   ix=order[s:s+CFG['batch']];x=tensors(ctr,ix,device);k=int(rng.integers(4));x=torch.rot90(x,k,(2,3))
   if rng.random()<.5:x=torch.flip(x,(3,))
   loss=torch.nn.functional.mse_loss(net(x),ztr[ix].to(device));opt.zero_grad();loss.backward();opt.step();losses.append(float(loss))
  wait_gpu(status,f'{cohort} val {epoch}');v=val_score(yva,predict(net,cva,device),mva,programs);score=float(np.nanmean(v))
  log.append({'cohort':cohort,'epoch':epoch,'train_mse':float(np.mean(losses)),'val_mean':score,**dict(zip(programs,v)),'seconds':time.time()-t0});pd.DataFrame(log).to_csv(out/'epoch_log.csv',index=False)
  if score>best[0]:best=(score,epoch);torch.save(net.state_dict(),E/f'{cohort}_best.pt')
  torch.save({'net':net.state_dict(),'opt':opt.state_dict(),'rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'epoch':epoch,'log':log,'best':list(best)},last.with_suffix('.tmp'));last.with_suffix('.tmp').rename(last)
  status.write_text(json.dumps({'date':stamp(),'phase':'training','cohort':cohort,'epoch':epoch,'val_mean':score,'best_epoch':best[1],'sec_epoch':time.time()-t0}));print('EPOCH',cohort,epoch,round(score,4),'best',best[1],f'{time.time()-t0:.0f}s',flush=True)
 net.load_state_dict(torch.load(E/f'{cohort}_best.pt',map_location=device));return net,best

def main():
 ap=argparse.ArgumentParser();ap.add_argument('step',choices=['cache','check','train']);ap.add_argument('--cohorts',nargs='*',default=['He2020','Meylan_ccRCC']);ap.add_argument('--share-gpu',action='store_true');a=ap.parse_args();SHARE['on']=a.share_gpu
 cfg=json.loads((R/'config/t4_frozen.json').read_text());programs=cfg['programs'];manifest=pd.read_csv(R/cfg['input_manifest']);torch.set_num_threads(2);O.mkdir(parents=True,exist_ok=True)
 groups={'He2020':['He2020','HER2ST'],'Meylan_ccRCC':['Meylan_ccRCC']};meta=manifest[manifest.cohort.isin(sum([groups[c] for c in a.cohorts],[]))].sort_values(['cohort','patient_key','sample'])
 if a.step=='cache':cache(meta);return
 if a.step=='check':check(manifest);return
 device='mps';assert torch.backends.mps.is_available();(O/'config.json').write_text(json.dumps(CFG,indent=2));sections=[];chosen=[]
 for cohort in a.cohorts:
  net,best=train(cohort,manifest,programs,device);chosen.append({'cohort':cohort,'best_epoch':best[1],'val_mean':best[0]})
  evals=manifest[manifest.split.isin(['test','external'])&(manifest.cohort.isin(['He2020','HER2ST']) if cohort=='He2020' else manifest.cohort.eq(cohort))]
  for row in evals.itertuples():
   c,y,_=load_rows(manifest[manifest['sample'].eq(row.sample)]);wait_gpu(O/'status.json',f'test {row.sample}');pred=predict(net,c,device);assert np.isfinite(pred).all()
   evaluation='HER2ST_external' if row.cohort=='HER2ST' else row.cohort+'_test'
   for j,p in enumerate(programs):sections.append({'backbone':'stnet_finetune','evaluation':evaluation,'cohort':row.cohort,'patient_key':row.patient_key,'patient':row.patient,'sample':row.sample,'program':p,'spots':row.spots,'spearman':corr(y[:,j],pred[:,j])})
  print('TESTED',cohort,best,flush=True)
 sections=pd.DataFrame(sections);patients=sections.groupby(['backbone','evaluation','cohort','patient_key','patient','program'],as_index=False).agg(spearman=('spearman','mean'),sections=('sample','size'),spots=('spots','sum'))
 summary=patients.groupby(['backbone','evaluation','cohort','program'],as_index=False).agg(patients=('patient_key','size'),patients_finite=('spearman','count'),median_spearman=('spearman','median'),mean_spearman=('spearman','mean'))
 pd.DataFrame(chosen).to_csv(O/'selected_epoch.csv',index=False);sections.to_csv(O/'heldout_section_spearman.csv',index=False);patients.to_csv(O/'heldout_patient_spearman.csv',index=False);summary.to_csv(O/'heldout_spearman_summary.csv',index=False);print(summary.to_string(),flush=True)
if __name__=='__main__':main()
