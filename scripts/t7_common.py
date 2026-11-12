"""T7 paths and fixed T6 spatial descriptor definitions."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',BLIS_NUM_THREADS='2',MPLBACKEND='Agg')
from pathlib import Path
import json,datetime,hashlib
import numpy as np,pandas as pd
from t6_common import P,S,PAIRS,MEANS,SPACE,FEATURES,graph,adjacency,moran,local_entropy,coloc,corr
R=Path(__file__).resolve().parents[1];C=json.loads((R/'config/t7.json').read_text());O=R/'results/t7';W=R/'docs/worklog/2026-10-27_T7';E=Path(C['output_root']);E.mkdir(parents=True,exist_ok=True);E6=E.parent/'t6';E5=E.parent/'t5'
def stamp():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(**kw):(W/'run_status.json').write_text(json.dumps({'time':stamp(),**kw},indent=2))
