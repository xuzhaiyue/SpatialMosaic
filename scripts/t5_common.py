"""Shared fixed T5 methods; no expression-model refitting."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',MPLBACKEND='Agg')
from pathlib import Path
import json,itertools,hashlib,datetime
import numpy as np,pandas as pd
from scipy import sparse
from scipy.stats import rankdata
R=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[1]));C=json.loads((R/'config/t5.json').read_text());O=R/'results/t5';W=R/'docs/worklog/2026-10-27_T5';E=Path(C['output_root']);E.mkdir(parents=True,exist_ok=True);P=C['programs'];S=C['short'];PAIRS=list(itertools.combinations(range(4),2));MEANS=[s+'__mean' for s in S];SPACE=[s+'__'+k for k in ['moran','high_fraction','entropy'] for s in S]+[S[a]+'__'+S[b]+'__coloc' for a,b in PAIRS];FEATURES=MEANS+SPACE

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.datetime.now().astimezone().isoformat()
def status(file,**kw):
 p=W/file;t=p.with_suffix('.tmp');t.write_text(json.dumps({'time':stamp(),**kw},indent=2));t.replace(p)
def graph(coords,step):
 xy=np.rint(coords).astype('int64');lookup={tuple(v):i for i,v in enumerate(xy)};edges=[]
 for i,(x,y) in enumerate(xy):
  for dx,dy in [(step,0),(0,step),(step,step),(step,-step)]:
   j=lookup.get((x+dx,y+dy))
   if j is not None:edges.append((i,j))
 return np.asarray(edges,dtype='int64').reshape(-1,2)
def adjacency(n,edges):
 if not len(edges):return sparse.csr_matrix((n,n))
 return sparse.csr_matrix((np.ones(len(edges)*2),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
def moran(values,edges):
 v=np.asarray(values,dtype='float64');z=v-v.mean(0);den=(z*z).sum(0)
 if not len(edges):return np.full(v.shape[1],np.nan)
 num=(z[edges[:,0]]*z[edges[:,1]]).sum(0);return np.divide(len(v)*num,len(edges)*den,out=np.full(v.shape[1],np.nan),where=den>0)
def coloc(values):
 ranks=rankdata(values,axis=0,method='average');z=ranks-ranks.mean(0);den=np.sqrt((z*z).sum(0));m=np.divide(z.T@z,den[:,None]*den[None,:],out=np.full((4,4),np.nan),where=(den[:,None]*den[None,:])>0);return [float(m[a,b]) for a,b in PAIRS]
def local_entropy(codes,A):
 neighborhood=A+sparse.eye(A.shape[0],format='csr');degree=np.asarray(neighborhood.sum(1)).ravel();counts=np.column_stack([neighborhood@(codes==k).astype(float) for k in range(10)]);q=counts/degree[:,None];return float((-np.where(q>0,q*np.log2(np.maximum(q,1e-300)),0).sum(1)).mean())
def ucell(values,bg,members):
 # Rows are samples, columns are the fixed background genes.
 v=np.asarray(values);rank=np.minimum(len(bg)+1-rankdata(v,axis=1,method='average'),C['max_rank']);lookup={g:i for i,g in enumerate(bg)};scores=[]
 for p in P:
  ix=[lookup[g] for g in members[p]];n=len(ix);scores.append(1-(rank[:,ix].sum(1)-n*(n+1)/2)/(C['max_rank']*n-n*(n+1)/2))
 return np.column_stack(scores)

def cdr_table():
 import zipfile,xml.etree.ElementTree as ET,re
 ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};path=Path(C['reference_root'])/'TCGA-CDR-SupplementalTableS1.xlsx'
 with zipfile.ZipFile(path) as z:
  strings=[''.join(v.itertext()) for v in ET.fromstring(z.read('xl/sharedStrings.xml'))];wb=ET.fromstring(z.read('xl/workbook.xml'));index=[v.attrib['name'] for v in wb.find('m:sheets',ns)].index('TCGA-CDR')+1;rows=ET.fromstring(z.read(f'xl/worksheets/sheet{index}.xml')).find('m:sheetData',ns);values=[]
  for row in rows:
   record={}
   for cell in row:
    letters=re.match('[A-Z]+',cell.attrib['r']).group();ix=0
    for char in letters:ix=ix*26+ord(char)-64
    v=cell.find('m:v',ns);value=v.text if v is not None else None
    if cell.attrib.get('t')=='s' and value is not None:value=strings[int(value)]
    record[ix-1]=value
   values.append(record)
  data=pd.DataFrame(values);data.columns=data.iloc[0];return data.iloc[1:].reset_index(drop=True)
