"""Provenance hashes of the analysis plan and configuration (checked against the authors' recorded files)."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];PLAN=R/'PLAN.md'
def plan_bytes():
 """The authors' analysis plan is not distributed with the code; without it the provenance checks are skipped."""
 return PLAN.read_bytes() if PLAN.exists() else None
def allowed_hashes():
 if not PLAN.exists():return None,None
 path=R/'docs/worklog/2026-10-26_T4/display_A1/amendment.json'
 if not path.exists():
  plan=hashlib.sha256((R/'PLAN.md').read_bytes()).hexdigest();assert plan==(R/'docs/worklog/2026-10-26_T3b/FROZEN_PLAN.sha256').read_text().split()[0];return [plan],[hashlib.sha256((R/'config/t4_frozen.json').read_bytes()).hexdigest()]
 a=json.loads(path.read_text());raw=(R/'PLAN.md').read_bytes();cfg=json.loads((R/'config/t4_frozen.json').read_text());core={k:v for k,v in cfg.items() if k!='figures'}
 assert hashlib.sha256(raw[:a['base_plan_bytes']]).hexdigest()==a['base_plan_sha256']
 assert hashlib.sha256(raw).hexdigest()==a['amended_plan_sha256']
 assert hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':')).encode()).hexdigest()==a['numerical_config_sha256']
 assert hashlib.sha256((R/'config/t4_frozen.json').read_bytes()).hexdigest()==a['amended_config_sha256']
 return a['allowed_feature_plan_sha256'],a['allowed_feature_config_sha256']
