from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from r2_analysis_33174 import ROOT, OUT

F = OUT / 'figures'
F.mkdir(parents=True, exist_ok=True)
S = json.loads((OUT / 'analysis/R2_analysis_summary.json').read_text())
OLD = ROOT / 'outputs/PONE-D-26-33174_revision_20260919'
BENCH = ROOT / 'server_results/petct_blood_benchmark'
BLUE, RED, GREEN, INK = '#2166ac', '#b2182b', '#34775b', '#222222'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9,
                     'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'legend.fontsize': 8, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
TEXT = {}


def tidy(ax, title):
    ax.set_title(title, loc='left', fontweight='bold', pad=9)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def save(fig, number):
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_text() and text.get_fontsize()<8:text.set_fontsize(8)
    fig.canvas.draw()
    TEXT[str(number)] = [item.get_text() for item in fig.findobj(matplotlib.text.Text) if item.get_text()]
    bounds=[]
    renderer=fig.canvas.get_renderer()
    ticklabels={label for axis in fig.axes for label in axis.get_xticklabels()+axis.get_yticklabels()}
    for item in fig.findobj(matplotlib.text.Text):
        if item not in ticklabels and item.get_text() and item.get_visible():
            b=item.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                bounds.append(item.get_text())
    if bounds:raise RuntimeError(f'Figure {number} has out-of-frame text: {bounds}')
    fig.savefig(F / f'Fig{number}.png', dpi=300)
    fig.savefig(F / f'Fig{number}.pdf')
    fig.savefig(F / f'Fig{number}.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def workflow():
    fig, ax = plt.subplots(figsize=(7.5, 7.4))
    fig.subplots_adjust(left=.03, right=.97, top=.98, bottom=.02)
    ax.set(xlim=(0, 100), ylim=(0, 100)); ax.axis('off')
    def box(x, y, w, h, text, color=BLUE):
        ax.add_patch(Rectangle((x, y), w, h, facecolor='white', edgecolor=color, lw=1.2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=8, linespacing=1.5)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1,color=INK))
    box(1, 86, 98, 13, 'PLOS-2024 cohort: 255 pretreatment stage III NSCLC patients\nAdenocarcinoma: 110 | Squamous carcinoma: 145\nOutcome: histological subtype; positive class = squamous carcinoma')
    box(1, 66, 30, 15, 'Clinical (7 variables)\nSex, age, BMI, numeric smoking\nexposure, T / N / stage codes')
    box(35, 66, 30, 15, 'Blood / inflammation (12)\nWBC, NEU, LYM, EOS, BAS,\nPLT, CEA, LDH, ALB, Ca,\nNLR and dNLR')
    box(69, 66, 30, 15, 'PET metabolic (5)\nTLG, MTV, SUVmean,\nSUVmax and SUVmin')
    for x in [16,50,84]:
        arrow((50,86),(x,81)); arrow((x,66),(50,61))
    box(1, 43, 98, 18, 'Development: 178 patients | Previously examined holdout: 77 patients\nDevelopment-only grid search; fully nested 5-fold CV refits all steps\nMedian imputation > variance filtering > standardization > SelectKBest > ridge\nPrimary model: sex, CEA, SUVmean, SUVmax, SUVmin (k = 5)')
    arrow((50,43),(50,39))
    box(1, 24, 47, 15, 'Comparator analyses\nMMFT / other ML models\nAdditional 3,850 radiomics features\nExploratory rank-refit kept separate')
    box(52, 24, 47, 15, 'Evaluation\nAUC, calibration, decision curves\nFully nested CV and split stability\nOutput: P(squamous histology)')
    arrow((25,24),(25,19)); arrow((75,24),(75,19))
    box(1,1,47,18,'TCGA primary-tumor analysis\nLUAD: 528 | LUSC: 501\nSubtype markers and gene sets\nBiological plausibility',GREEN)
    box(52,1,47,18,'Independent TCIA analyses\nClinical associations: 207 patients\nCT-semantic histology: 187 patients\nRidge versus MMFT; no PET/blood validation',GREEN)
    save(fig,1)


def performance():
    ridge = pd.read_csv(BENCH / 'best_model_test_predictions.csv')
    mmft = pd.read_csv(BENCH / 'mmft_rank_refit_predictions.csv')
    y, p = ridge.histology.to_numpy(), ridge.predicted_probability.to_numpy()
    fig, axes = plt.subplots(2,2,figsize=(7.5,6.9), layout='constrained')
    a,b,c,d = axes.flat
    for name, frame, col, style in [('Primary ridge',ridge,BLUE,'-'),('Rank-refit (exploratory)',mmft,RED,'--')]:
        fpr,tpr,_=roc_curve(frame.histology,frame.predicted_probability)
        a.plot(fpr,tpr,color=col,ls=style,label=f'{name}: {roc_auc_score(frame.histology,frame.predicted_probability):.3f}')
    a.plot([0,1],[0,1],':',color='#777777'); a.legend(loc='lower right')
    a.set(xlabel='1 - specificity',ylabel='Sensitivity',xlim=(0,1),ylim=(0,1));tidy(a,'A  Previously examined holdout')
    cm=confusion_matrix(y,p>=.5,labels=[0,1]); b.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): b.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=12,color='white' if cm[i,j]>cm.max()/2 else INK)
    b.set(xticks=[0,1],yticks=[0,1],xticklabels=['Adeno','Squamous'],yticklabels=['Adeno','Squamous'],xlabel='Predicted subtype',ylabel='Observed subtype');tidy(b,'B  Primary ridge at threshold 0.50')
    for name,label,col,x in [('Adenocarcinoma',0,BLUE,0),('Squamous',1,RED,1)]:
        values=p[y==label]; c.boxplot(values,positions=[x],widths=.55,patch_artist=True,boxprops={'facecolor':col,'alpha':.25},medianprops={'color':INK})
        c.scatter(x+np.random.default_rng(7).uniform(-.17,.17,len(values)),values,s=10,color=col,alpha=.7)
    c.axhline(.5,color='#777777',ls=':');c.set(xticks=[0,1],xticklabels=['Adenocarcinoma','Squamous'],ylabel='Predicted P(squamous)',ylim=(0,1));tidy(c,'C  Primary-model probabilities')
    frac,means=calibration_curve(y,p,n_bins=5,strategy='quantile')
    d.plot([0,1],[0,1],':',color='#777777');d.plot(means,frac,'o-',color=BLUE)
    d.text(.04,.95,'Brier = 0.166\nSlope = 1.51\nHL p = 0.086',va='top',transform=d.transAxes)
    d.set(xlabel='Predicted P(squamous)',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1));tidy(d,'D  Primary-model calibration')
    save(fig,2)


def modality():
    fig,axes=plt.subplots(2,2,figsize=(7.5,7.2),layout='constrained')
    a,b,c,d=axes.flat
    labels=['Clinical','+ Blood','+ PET metabolic','+ Radiomics (3,850)']
    for i,row in enumerate(S['blocks']):
        a.errorbar(row['heldout_auc'],3-i,xerr=[[row['heldout_auc']-row['ci_low']],[row['ci_high']-row['heldout_auc']]],fmt='o',color=BLUE,capsize=3)
        a.text(.98,3-i,f"{row['heldout_auc']:.3f}",ha='right',va='center')
    a.set(yticks=range(4),yticklabels=labels[::-1],xlim=(.45,1),xlabel='Holdout AUC (95% bootstrap CI)');tidy(a,'A  Development-only block tuning')
    modeldata=[('Primary ridge',.854),('MMFT ensemble',.830),('Rank-refit\n(exploratory)',.868)]
    b.scatter([x[1] for x in modeldata],[2,1,0],s=45,color=[BLUE,GREEN,RED])
    for i,(_,v) in enumerate(modeldata):b.text(.99,2-i,f'{v:.3f}',va='center',ha='right')
    b.axvline(.854,color=BLUE,ls=':',lw=1);b.set(yticks=[2,1,0],yticklabels=[x[0] for x in modeldata],xlim=(.65,1),xlabel='Previously examined holdout AUC');tidy(b,'B  Secondary deep-learning comparisons')
    row=S['blocks'][2]; c.scatter(range(1,6),row['nested_folds'],s=40,color=BLUE);c.axhline(row['nested_mean'],color=BLUE)
    c.axhspan(row['nested_mean']-row['nested_sd'],row['nested_mean']+row['nested_sd'],alpha=.1,color=BLUE)
    c.set(xticks=range(1,6),xlabel='Outer fold',ylabel='AUC',ylim=(.6,.95));tidy(c,'C  Fully nested primary-block CV')
    c.text(.04,.05,f"Mean {row['nested_mean']:.3f} (SD {row['nested_sd']:.3f})",transform=c.transAxes)
    dc=pd.read_csv(ROOT/'outputs/supplementary_experiments/S3_decision_curve_primary.csv')
    d.plot(dc.threshold,dc.net_benefit_model,color=BLUE,label='Primary ridge');d.plot(dc.threshold,dc.net_benefit_all,ls='--',color='#888888',label='Classify all squamous');d.axhline(0,color=INK,ls=':')
    d.set(xlim=(0,.6),ylim=(-.1,.65),xlabel='Threshold P(squamous)',ylabel='Net benefit');d.legend();tidy(d,'D  Primary-model decision curve')
    save(fig,3)


def biology():
    spec=importlib.util.spec_from_file_location('r2_tcga',ROOT/'scripts/create_tcga_external_bioinfo_figures.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    df=pd.read_csv(ROOT/'public_data/tcga_xena_lung_external/TCGA_LUAD_LUSC_external_bioinfo_merged.csv')
    counts=df.cohort.value_counts().to_dict()
    assert counts=={'LUAD':528,'LUSC':501},counts
    diff=pd.read_csv(ROOT/'public_data/tcga_xena_lung_external/TCGA_LUAD_LUSC_selected_gene_differential.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.5,7.3),layout='constrained');a,b,c,d=axes.flat
    a.scatter(diff.log2FC_LUSC_vs_LUAD,diff.neglog10FDR,c=np.where(diff.log2FC_LUSC_vs_LUAD>0,RED,BLUE),s=17,alpha=.8)
    for gene in ['CEACAM5','TP63','SLC2A1','NKX2-1']:
        r=diff.loc[diff.gene==gene].iloc[0];a.annotate(gene,(r.log2FC_LUSC_vs_LUAD,r.neglog10FDR),xytext=(3,5),textcoords='offset points',fontsize=8)
    a.axvline(0,color='#888888',ls=':');a.set(xlabel='Mean log2(TPM+1): LUSC - LUAD',ylabel='-log10(FDR)');tidy(a,'A  Selected-marker differential expression')
    genes=['CEACAM5','SLC2A1','HK2','LDHA','TP63','KRT5']
    values=df.groupby('cohort')[genes].mean().loc[['LUAD','LUSC']].to_numpy()
    image=b.imshow(values,aspect='auto',cmap='viridis');b.set(xticks=range(len(genes)),xticklabels=genes,yticks=[0,1],yticklabels=['LUAD','LUSC']);b.tick_params(axis='x',rotation=55);fig.colorbar(image,ax=b,label='Mean log2(TPM+1)',shrink=.65);tidy(b,'B  Model-linked marker expression')
    names=list(module.PATHWAYS)
    scores=df.groupby('cohort')[names].mean().loc[['LUAD','LUSC']].T
    vmax=abs(scores.to_numpy()).max(); image=c.imshow(scores,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    c.set(xticks=[0,1],xticklabels=['LUAD','LUSC'],yticks=range(len(names)),yticklabels=names)
    fig.colorbar(image,ax=c,label='Mean selected-gene z-score',shrink=.7);tidy(c,'C  Selected-gene proxy scores')
    module.plot_km(d,df,'');tidy(d,'D  Survival by exploratory biological proxy')
    save(fig,5)


def optimization():
    base=ROOT/'outputs/model_optimization'
    summary=json.loads((base/'model_optimization_summary.json').read_text())
    rep=np.loadtxt(base/'repeated_holdout_aucs.csv',delimiter=',',skiprows=1)
    lc=pd.read_csv(base/'learning_curve.csv'); sub=pd.read_csv(base/'subgroup_by_stage.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.5,6.9),layout='constrained');a,b,c,d=axes.flat
    a.hist(rep,bins=14,color=BLUE,edgecolor='white',alpha=.8);a.axvline(.854,color=RED,label='Original holdout: 0.854');a.axvline(rep.mean(),color=BLUE,ls='--',label=f'Fixed-config mean: {rep.mean():.3f}')
    a.set(xlabel='Test AUC',ylabel='Number of splits');a.legend();tidy(a,'A  Fixed-configuration split sensitivity')
    labels=['Primary ridge','Elastic-net','+ Radiomics signature','Ridge + XGBoost']
    vals=[summary['primary_reference']['test_auc'],summary['elastic_net']['test_auc'],summary['radiomics_signature']['primary_plus_signature_test_auc_like_for_like'],summary['stacking_ensemble']['stack_test_auc']]
    b.scatter(vals,[3,2,1,0],color=[BLUE,GREEN,RED,INK],s=40)
    for i,v in enumerate(vals): b.text(v+.01,3-i,f'{v:.3f}',va='center')
    b.set(yticks=[3,2,1,0],yticklabels=labels,xlabel='Previously examined holdout AUC',xlim=(.65,1));tidy(b,'B  Exploratory optimization attempts')
    c.errorbar(lc.train_fraction*100,lc.mean_auc,yerr=lc.std_auc,fmt='o-',capsize=3,color=BLUE)
    c.set(xlabel='Development-set fraction (%)',ylabel='AUC (mean +/- SD)',ylim=(.6,.95));tidy(c,'C  Training-subsample learning curve')
    ys=np.arange(len(sub))[::-1]
    for yi,(_,r) in zip(ys,sub.iterrows()):
        if pd.notna(r.auc):d.errorbar(r.auc,yi,xerr=[[r.auc-r.ci_low],[r.ci_high-r.auc]],fmt='o',color=GREEN,capsize=3)
    labs=[f"Code {str(r.subgroup).split()[1] if len(str(r.subgroup).split())>1 else r.subgroup} (n={int(r.n)})" for _,r in sub.iterrows()]
    d.set(yticks=ys,yticklabels=labs,xlabel='AUC (95% bootstrap CI)',xlim=(.4,1));tidy(d,'D  Raw stage-code subgroups')
    save(fig,6)


def cross_cohort():
    fig,axes=plt.subplots(1,2,figsize=(7.5,4.6),layout='constrained');a,b=axes
    labels=['Ridge (primary)','Rank-refit (exploratory)','MMFT search (10 seeds)','TabPFN','Biology-token search']
    vals=[.854,.868,.828,.852,.834]
    a.scatter(vals,range(4,-1,-1),color=[INK,RED,BLUE,GREEN,BLUE],s=40)
    for i,v in enumerate(vals):a.text(.99,4-i,f'{v:.3f}',va='center',ha='right')
    a.axvline(.854,color=INK,ls=':');a.set(yticks=range(4,-1,-1),yticklabels=labels,xlim=(.74,1),xlabel='Previously examined holdout AUC')
    tidy(a,'A  No comparator significantly exceeded ridge')
    b.errorbar([.749,.713],[1,0],xerr=[.105,.108],fmt='o',color=BLUE,capsize=5)
    b.set(yticks=[1,0],yticklabels=['Ridge','MMFT'],xlim=(.4,1),ylim=(-.6,1.6),xlabel='Nested-CV AUC (mean +/- SD)')
    for y,v in [(1,.749),(0,.713)]:b.text(v,y+.22,f'{v:.3f}',ha='center')
    tidy(b,'B  TCIA CT semantics (n = 187)')
    save(fig,9)


def supplement():
    fig,axes=plt.subplots(1,2,figsize=(7.5,4.6),layout='constrained');a,b=axes
    labels=['Clinical','+ Blood','+ PET (k=8)','+ Radiomics (3,850)']
    for i,row in enumerate(S['blocks']):
        a.errorbar(row['heldout_auc'],3-i,xerr=[[row['heldout_auc']-row['ci_low']],[row['ci_high']-row['heldout_auc']]],fmt='o',color=BLUE,capsize=3)
    a.set(yticks=[3,2,1,0],yticklabels=labels,xlim=(.45,1),xlabel='AUC (95% bootstrap CI)');tidy(a,'A  Revised modality comparisons')
    labels=['Primary ridge','FT-Transformer','MMFT (no rank loss)','MMFT ensemble','Rank-refit (exploratory)'];values=[.854,.815,.821,.830,.868]
    b.scatter(values,range(4,-1,-1),color=[INK,BLUE,BLUE,GREEN,RED],s=40)
    for i,v in enumerate(values):b.text(.99,4-i,f'{v:.3f}',ha='right',va='center')
    b.set(yticks=range(4,-1,-1),yticklabels=labels,xlim=(.65,1),xlabel='Previously examined holdout AUC');tidy(b,'B  Exploratory model comparisons')
    save(fig,'S2')


def main():
    chosen={'1':workflow,'2':performance,'3':modality,'5':biology,'6':optimization,'9':cross_cohort,'S2':supplement}
    functions=list(chosen.values()) if len(sys.argv)==1 else [chosen[key] for key in sys.argv[1:]]
    oldtext=OUT/'analysis/R2_figure_text.json'
    if oldtext.exists():TEXT.update(json.loads(oldtext.read_text(encoding='utf-8')))
    for function in functions:
        function();print('Rendered',function.__name__,flush=True)
    (OUT/'analysis/R2_figure_text.json').write_text(json.dumps(TEXT,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=[]
    for p in sorted(F.glob('*.tif')):
        with Image.open(p) as image:
            audit.append({'file':p.name,'pixels':image.size,'dpi':[float(x) for x in image.info.get('dpi',[])],'mode':image.mode,'bytes':p.stat().st_size})
    (OUT/'analysis/R2_figure_metadata.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')


if __name__=='__main__':main()
