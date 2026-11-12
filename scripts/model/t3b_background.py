"""Define the shared uniquely mapped protein-coding gene universe (no targets used)."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
from pathlib import Path
import json,gzip,hashlib,collections
import pandas as pd,anndata as ad
R=Path(__file__).resolve().parents[2];O=R/'results/t3b';W=R/'docs/worklog/2026-10-26_T3b'
map_path=R/'references/2026-10-23_STPath_PathPrism/STPath/code/utils_data/symbol2ensembl.json';mapping=json.loads(map_path.read_text())
def digest(seq):return hashlib.sha256(('\n'.join(sorted(seq))+'\n').encode()).hexdigest()
def unique_ids(names,raw=None):
 ids=[v.split('.')[0] if str(v).startswith('ENSG') else mapping.get(str(n)) for n,v in zip(names,raw if raw is not None else names)]
 freq=collections.Counter(ids);return {g for g in ids if g and g.startswith('ENSG') and freq[g]==1}
records=[];common=None
for row in pd.read_csv(R/'results/t2/ST_X_counts_and_depth.csv').itertuples():
 a=ad.read_h5ad(row.source_path,backed='r');names=a.var_names.astype(str).tolist();raw=a.var['gene_ids'].astype(str).tolist() if 'gene_ids' in a.var else names;ids=unique_ids(names,raw);a.file.close()
 records.append({'kind':row.cohort,'sample':row.sample,'source':row.source_path,'n_unique_ids':len(ids),'gene_universe_sha256':digest(ids)})
 common=ids if common is None else common&ids
rna=pd.read_csv(R/'docs/worklog/2026-10-26_T3/download_manifest.csv');coding=None;gene_names=collections.defaultdict(set)
for row in rna[rna.purpose.eq('CPTAC_primary_RNA')].itertuples():
 ids=set();allids=[]
 with open(row.destination_file) as f:
  model=next(f).strip();header=next(f).strip().split('\t')
  for line in f:
   gene,name,typ=line.rstrip('\n').split('\t',3)[:3]
   if gene.startswith('ENSG'):
    gid=gene.split('.')[0];allids.append(gid)
    if typ=='protein_coding':ids.add(gid);gene_names[gid].add(name)
 freq=collections.Counter(allids);ids={g for g in ids if freq[g]==1};coding=ids if coding is None else coding&ids
 records.append({'kind':row.cohort,'sample':row.file,'source':row.destination_file,'source_sha256':row.sha256,'gene_model':model,'n_unique_ids':len(ids),'gene_universe_sha256':digest(ids)})
for row in pd.read_csv(R/'docs/worklog/2026-10-26_T1c_download/download_manifest.csv').to_dict('records'):
 # Locate the Xena expression matrices.
 p=Path(row.get('downloaded_path',row.get('destination_file',row.get('verified_path',row.get('path','')))))
 if not p.name.endswith('HiSeqV2.gz'):continue
 with gzip.open(p,'rt') as f:
  next(f);names=[line.split('\t',1)[0] for line in f]
 ids=unique_ids(names);common &= ids
 records.append({'kind':'Xena','sample':p.name,'source':str(p),'n_unique_ids':len(ids),'gene_universe_sha256':digest(ids)})
assert sum(x['kind']=='Xena' for x in records)==2,'Both TCGA gene universes required'
bg=sorted(common&coding);assert len(bg)>1500
symbols={g:';'.join(sorted(gene_names[g])) for g in bg};pd.DataFrame({'ensembl_id':bg,'gencode_symbols':[symbols[g] for g in bg]}).to_csv(O/'protein_coding_background.csv',index=False)
(O/'protein_coding_background.txt').write_text('\n'.join(bg)+'\n');pd.DataFrame(records).to_csv(O/'background_sources.csv',index=False)
gmt=R/'references/gene_sets/msigdb_2026.2.Hs/h.all.v2026.2.Hs.symbols.gmt';programs=['E2F_TARGETS','INTERFERON_GAMMA_RESPONSE','HYPOXIA','EPITHELIAL_MESENCHYMAL_TRANSITION'];alias={'FASTKD4':'TBRG4','RIGI':'DDX58','TMT1B':'METTL7B','HACL2':'ILVBL','NUCLEOLIN':'NCL','PRP4K':'PRPF4B'};members={};rows=[]
for line in gmt.read_text().splitlines():
 parts=line.split('\t');program=parts[0].replace('HALLMARK_','')
 if program not in programs:continue
 keep=[]
 for original in parts[2:]:
  symbol=alias.get(original,original);gid=mapping.get(symbol);present=gid in bg
  rows.append({'program':program,'official_symbol':original,'lookup_symbol':symbol,'ensembl_id':gid,'in_background':present})
  if present:keep.append(gid)
 members[program]=sorted(set(keep))
pd.DataFrame(rows).to_csv(O/'program_member_mapping.csv',index=False);(O/'program_members.json').write_text(json.dumps(members,indent=2))
provenance={'background_genes':len(bg),'background_sha256':digest(bg),'definition':'Intersection of unique human ENSG across all ST gene tables and both Xena gene tables, and protein_coding ENSG common to all approved GENCODE v36 STAR files; no expression filtering','program_members':{k:len(v) for k,v in members.items()},'gmt_sha256':hashlib.sha256(gmt.read_bytes()).hexdigest(),'mapping_sha256':hashlib.sha256(map_path.read_bytes()).hexdigest(),'input_sources':len(records),'date':'2026-10-26'}
(O/'background_provenance.json').write_text(json.dumps(provenance,indent=2));print(json.dumps(provenance,indent=2),flush=True)
