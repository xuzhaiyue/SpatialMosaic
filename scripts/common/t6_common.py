"""Shared T6 paths and unchanged T5 spatial methods."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',BLIS_NUM_THREADS='2',MPLBACKEND='Agg')
from pathlib import Path
import json,datetime,hashlib,itertools
import numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata
from t5_common import graph,adjacency,moran,local_entropy,coloc
R=Path(os.environ.get('PROJECT_ROOT',Path(__file__).resolve().parents[2]));C=json.loads((R/'config/t6.json').read_text());O=R/'results/t6';W=R/'docs/worklog/2026-10-27_T6';E=Path(C['output_root']);E.mkdir(parents=True,exist_ok=True);P=C['programs'];S=C['short'];PAIRS=list(itertools.combinations(range(4),2));MEANS=[s+'__mean' for s in S];SPACE=[s+'__'+k for k in ['moran','high_fraction','entropy'] for s in S]+[S[a]+'__'+S[b]+'__coloc' for a,b in PAIRS];FEATURES=MEANS+SPACE

def stamp():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(file,**kw):
 p=W/file;t=p.with_suffix('.tmp');t.write_text(json.dumps({'time':stamp(),**kw},indent=2));t.replace(p)
def corr(a,b):return float(spearmanr(a,b).statistic) if len(a)>=3 and np.ptp(a)>0 and np.ptp(b)>0 else np.nan
