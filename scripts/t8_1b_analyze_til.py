"""Only requested BRCA TIL pattern/density associations; no clinical outcomes or retraining."""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2',MPLBACKEND='Agg')
from pathlib import Path
import pandas as pd,numpy as np,json,csv,re,zipfile,xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1];O=R/'results/t8_1b';W=R/'docs/worklog/2026-10-28_T8-1b';D=Path('/Volumes/INTENSO/SpatialMosaic_data/03_references/saltz_2018');F=O/'figures';score='E2F__IFNg__coloc'
# Read only identity, image TIL and author pattern fields from the integrated workbook.
ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};rows=[];allowed={'A','B','M','O','AD','AE','AF'}
with zipfile.ZipFile(D/'TILMap_TableS1.xlsx') as z:
 strings=[''.join(v.itertext()) for v in ET.fromstring(z.read('xl/sharedStrings.xml'))];data=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).find('m:sheetData',ns)
 for row in data:
  rec={}
  for c in row:
   col=re.match('[A-Z]+',c.attrib['r']).group()
   if col not in allowed:continue
   v=c.find('m:v',ns);value=v.text if v is not None else ''
   if c.attrib.get('t')=='s' and value!='':value=strings[int(value)]
   rec[col]=value
  rows.append(rec)
header=rows[0];table=pd.DataFrame(rows[1:]).rename(columns=header);table=table[table.Study.eq('BRCA')].copy();table['til_percentage']=pd.to_numeric(table.til_percentage,errors='coerce');table['Number of TIL Patches']=pd.to_numeric(table['Number of TIL Patches'],errors='coerce');assert not table.Slide.duplicated().any();table.to_csv(O/'Saltz_TableS1_BRCA_image_fields.csv',index=False)
# The published summary has a malformed leading quote; normalize only a derived reader.
parsed=[];repaired=0;bad=[]
with (D/'TILmap_Summary_20171013.tsv').open() as f:
 names=next(csv.reader([next(f)]));seen={};unique=[]
 for name in names:seen[name]=seen.get(name,0)+1;unique.append(name if seen[name]==1 else name+'__'+str(seen[name]))
 for lineno,line in enumerate(f,2):
  vals=next(csv.reader([line]))
  if len(vals)==len(names)-1 and re.match(r'^"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4},"TCGA-',line):
   line=re.sub(r'^"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}),"',r'"\1","',line,count=1);vals=next(csv.reader([line]));repaired+=1
  if len(vals)!=len(names):bad.append({'line':lineno,'columns':len(vals)});continue
  parsed.append(dict(zip(unique,vals)))
assert not bad,bad[:5];summary=pd.DataFrame(parsed);brca_ids=set(pd.read_csv(O/'TCGA_BRCA_DX1_tertiles_pre_TIL.csv').patient_id);summary=summary[summary.ParticipantBarcode.isin(brca_ids)].copy();summary['til_percentage']=pd.to_numeric(summary.til_percentage,errors='coerce');summary.to_csv(O/'Saltz_summary_BRCA_normalized.csv',index=False);verify=summary[['Slides','til_percentage']].merge(table[['Slide','til_percentage']],left_on='Slides',right_on='Slide',suffixes=('_summary','_S1'));verify['difference']=verify.til_percentage_summary-verify.til_percentage_S1;verify.to_csv(O/'density_source_field_comparison.csv',index=False);(O/'format_reading_record.json').write_text(json.dumps({'summary_actual_delimiter':',','summary_header_columns':len(names),'rows_repaired_missing_leading_quote':repaired,'bad_rows':bad,'source_not_modified':True,'density_primary':'Original TableS1 image til_percentage; not molecular Lymphocytes','overlapping_BRCA_rows':len(verify),'max_density_difference_between_publication_tables':float(verify.difference.abs().max())},indent=2))
pattern=pd.read_csv(D/'TIL-Pattern-Labels.csv',keep_default_na=False);assert not pattern.Slides.duplicated().any();mapping={'Brisk Diffuse':'brisk diffuse','Brisk Band-like':'band-like','Non-Brisk Multifocal':'non-brisk','Non-Brisk Focal':'non-brisk','None':'none','Indeterminate':'indeterminate','':'missing'};unknown=set(pattern.PatternLabels)-set(mapping);assert not unknown,unknown;pattern['til_pattern4']=pattern.PatternLabels.map(mapping)
base=pd.read_csv(O/'TCGA_BRCA_DX1_tertiles_pre_TIL.csv');full=pd.read_csv(R/'results/t7/wsi_spatial_by_slide.csv');full=full[full.cohort.eq('TCGA-BRCA')];full['slide_barcode']=full.slide.str.split('.').str[0];counts=full.groupby('slide_barcode').size();base['feature_files_with_same_slide_barcode']=base.slide_barcode.map(counts);matched=base.merge(pattern[['Slides','PatternLabels','til_pattern4','TIL-On-Ink-Artifacts']],left_on='slide_barcode',right_on='Slides',how='left',validate='one_to_one').merge(table[['Slide','til_percentage','Number of TIL Patches','Global_Pattern']],left_on='slide_barcode',right_on='Slide',how='left',validate='one_to_one');matched.to_csv(O/'TCGA_BRCA_Saltz_matched.csv',index=False)
order=['High','Middle','Low'];cats=['brisk diffuse','band-like','non-brisk','none'];valid=matched[matched.til_pattern4.isin(cats)];cross=pd.crosstab(valid.coloc_group,valid.til_pattern4).reindex(index=order,columns=cats,fill_value=0);cross.to_csv(O/'coloc_tertiles_by_TIL_pattern_counts.csv');pct=cross.div(cross.sum(1),axis=0)*100;pct.to_csv(O/'coloc_tertiles_by_TIL_pattern_percent.csv');status=matched.assign(reference_status=matched.til_pattern4.fillna('unmatched')).groupby(['coloc_group','reference_status']).size().unstack(fill_value=0).reindex(order);status.to_csv(O/'TIL_pattern_match_status.csv');fig,axes=plt.subplots(1,2,figsize=(12,4));im=axes[0].imshow(pct,cmap='Blues',vmin=0,vmax=100,aspect='auto');axes[0].set_xticks(range(4),cats,rotation=20,ha='right');axes[0].set_yticks(range(3),order);axes[0].set_title(f'Author TIL patterns | matched n={len(valid)}')
for i in range(3):
 for j in range(4):axes[0].text(j,i,f'{cross.iloc[i,j]}\n{pct.iloc[i,j]:.1f}%',ha='center',va='center',fontsize=10)
fig.colorbar(im,ax=axes[0],label='Within-coloc-group (%)');bottom=np.zeros(3)
for cat in cats:axes[1].bar(order,pct[cat],bottom=bottom,label=cat);bottom+=pct[cat].to_numpy()
axes[1].set(ylabel='Within-group %',ylim=(0,100),title='Non-brisk focal + multifocal combined');axes[1].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2);fig.suptitle('T7 cross-fitted E2F–IFNg colocation tertiles vs Saltz 2018 TIL pattern');fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(F/('TIL_pattern_crosstab.'+ext),dpi=170,bbox_inches='tight')
plt.close(fig);density=matched[matched.til_percentage.notna() & np.isfinite(matched[score])].copy();rho,p=spearmanr(density[score],density.til_percentage);density.to_csv(O/'coloc_vs_TIL_density_patients.csv',index=False);result=dict(n=int(len(density)),rho=float(rho),p=float(p),density_definition='TableS1 til_percentage: image TIL-positive patches / tissue patches (%)',score_definition='Within-slide E2F vs IFNg Spearman on existing T7 predicted patches',group_thresholds=json.loads((O/'pre_TIL_group_thresholds.json').read_text()),pattern_matched_n=int(len(valid)),indeterminate_n=int(matched.til_pattern4.eq('indeterminate').sum()),pattern_unmatched_n=int(matched.til_pattern4.isna().sum()),density_unmatched_n=int(matched.til_percentage.isna().sum()),ink_flag_n=int(matched['TIL-On-Ink-Artifacts'].eq('Y').sum()),ambiguous_scan_version_n=int(matched.feature_files_with_same_slide_barcode.gt(1).sum()));(O/'TIL_comparison_summary.json').write_text(json.dumps(result,indent=2));fig,ax=plt.subplots(figsize=(7,5))
for group,color in [('Low','#426E9C'),('Middle','#A4A8AA'),('High','#AF4E34')]:
 g=density[density.coloc_group.eq(group)];ax.scatter(g[score],g.til_percentage,s=14,alpha=.5,color=color,label=f'{group}, n={len(g)}')
ax.set(xlabel='T7 within-slide E2F–IFNg predicted Spearman',ylabel='Saltz image TIL-positive tissue patches (%)',title=f'BRCA | n={len(density)} | Spearman rho={rho:.3f}, p={p:.3g}');ax.legend(fontsize=9);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
for ext in ['png','pdf','svg']:fig.savefig(F/('coloc_vs_TIL_density.'+ext),dpi=170,bbox_inches='tight')
plt.close(fig);(W/'TIL_analysis_complete.json').write_text(json.dumps({'complete':True,**result},indent=2));print(json.dumps(result,indent=2));print(cross.to_string());print('TIL ANALYSIS COMPLETE',flush=True)
