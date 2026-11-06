"""Generate paper tables/figures and auditable error cases from completed runs only."""
import json
import re
import textwrap
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from trialmatch.data import ROOT, read_split, LABELS
from trialmatch.metrics import compute

def tex(s):
    chars={'&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$',
           '{':r'\{','}':r'\}','\\':'/','~':r'\textasciitilde{}','^':r'\textasciicircum{}',
           '\u2013':'-','\u2014':'-','\u2011':'-','\u2019':"'",'\u2018':"'",'\u201c':'"','\u201d':'"',
           '\u2265':r'\(\geq\)','\u2264':r'\(\leq\)','\u00d7':r'\(\times\)',
           '<':r'\(<\)','>':r'\(>\)',
           '\u00b0':r'\(^{\circ}\)','\u00a0':' '}
    return ''.join(chars.get(c,c) for c in ' '.join(str(s).split()))

def load(name): return json.loads((ROOT/'results'/name).read_text(encoding='utf-8'))

def main():
    names=[f'{split}_{model}' for split in ['validation','test'] for model in ['base','adapted']]
    metrics={n:load(n+'_metrics.json') for n in names}
    bench=load('benchmarks.json')
    cfg=load('training_config.json')
    selected=json.loads((ROOT/'models/adapter/config.json').read_text())
    logs=[json.loads(s) for s in (ROOT/'results/training.jsonl').read_text().splitlines()]
    pred=[json.loads(s) for s in (ROOT/'results/test_adapted_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    figdir=ROOT/'report/figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(3.5,2.1),layout='constrained')
    steps=[r for r in logs if 'step' in r]
    axes[0].plot([r['step']+(r['epoch']-1)*cfg['train_count'] for r in steps],[r['train_loss'] for r in steps],color='#16644e',lw=1.3)
    axes[0].set(xlabel='Examples processed',ylabel='Response CE',title='Training loss')
    axes[0].tick_params(labelsize=7)
    epochs=[r for r in logs if 'validation_loss' in r]
    axes[1].plot([r['epoch'] for r in epochs],[r['validation_loss'] for r in epochs],marker='o',color='#936323')
    axes[1].axvline(selected['selected_epoch'],color='#718577',ls='--',lw=.8)
    axes[1].set(xlabel='Epoch',ylabel='Validation CE',title='Validation loss',xticks=[1,2,3])
    fig.savefig(figdir/'training.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7,2.5),layout='constrained')
    short_labels={'included':'INC','not included':'NI','excluded':'EXC','not excluded':'NE',
                  'not enough information':'INFO','not applicable':'NA'}
    for ax,group in zip(axes,['overall','inclusion','exclusion']):
        m=metrics['test_adapted'][group];matrix=np.array(m['confusion_matrix'])
        ax.imshow(matrix,cmap='Greens')
        labels=[short_labels[s] for s in m['confusion_labels']]
        ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,title=group.capitalize(),xlabel='Predicted',ylabel='Expert')
        ax.tick_params(labelsize=7)
        for (i,j),value in np.ndenumerate(matrix):
            ax.text(j,i,str(value),ha='center',va='center',fontsize=7,color='white' if value>matrix.max()/2 else '#143d2b')
    fig.savefig(figdir/'confusion.png',dpi=240);plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.5,2.2),layout='constrained')
    for mode,color in [('uncached','#a77a3b'),('cached','#16644e'),('speculative','#7488a7')]:
        records=[r for r in bench['records'] if r['kind']=='decoding' and r['mode']==mode]
        ax.plot([f"{r['input_tokens']}/{r['output_tokens']}" for r in records],[r['median_s'] for r in records],marker='o',label=mode,color=color)
    ax.set(xlabel='Input / output tokens',ylabel='Median latency (s)',title='CPU generation latency')
    ax.legend(frameon=False,ncol=1,fontsize=7)
    fig.savefig(figdir/'latency.png',dpi=220);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,1.7),layout='constrained');ax.axis('off')
    boxes=[(.01,'Numbered patient\n+ criterion/type'),(.27,'135M decoder\n+ late-block LoRA'),(.53,'Label + evidence\nbitmap grammar'),(.79,'Source inspection\n+ review / export')]
    for x,label in boxes:
        ax.text(x+.1,.58,label,ha='center',va='center',fontsize=9,bbox={'boxstyle':'round,pad=.65','facecolor':'#edf4eb','edgecolor':'#8eab93'})
    for x in [.21,.47,.73]: ax.annotate('',xy=(x+.05,.58),xytext=(x,.58),arrowprops={'arrowstyle':'->','color':'#385c46'})
    ax.text(.5,.02,'Greedy reference / explicit KV cache / verified speculative path',ha='center',fontsize=9,color='#385c46')
    fig.savefig(figdir/'workflow.png',dpi=220);plt.close(fig)

    s=[]
    s.append(r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{workflow.png}\caption{Screening workflow. All decoding modes share the target model; the evidence parser preserves source sentence IDs.}\label{fig:workflow}\end{figure*}')
    s.append(r'Figure~\ref{fig:workflow} summarizes the application path. The full experiments use '+str(cfg['train_count'])+' training annotations; development smoke runs are archived separately and are not reported as final results.')
    test_log=(ROOT/'results/tests.txt').read_text(encoding='utf-8',errors='replace')
    passed=re.search(r'(\d+) passed',test_log)
    if not passed:raise ValueError('A successful final correctness-test record is required.')
    s.append(f'The final automated suite reports {passed.group(1)} passing checks. These cover attention agreement, cached/uncached/speculative token equality, cache growth, full-acceptance and rejection branches, early stopping, structured masks, evidence empty-set conventions, data isolation, response-loss equivalence, frozen base weights, truncation and application input/export handling.')
    s.append(r'\begin{table}[t]\centering\small\caption{Selected training configuration and observed preprocessing.}\label{tab:training}\begin{tabular}{lr}\toprule Setting & Value\\\midrule')
    s += [f'Target parameters & {cfg["total_parameters"]:,}'+r'\\',f'Trainable parameters & {cfg["trainable_parameters"]:,}'+r'\\',f'Selected epoch & {selected["selected_epoch"]}'+r'\\',f'Best validation loss & {selected["validation_loss"]:.4f}'+r'\\',f'Three-epoch elapsed time & {epochs[-1]["elapsed_s"]/60:.2f} min'+r'\\']
    trunc=load('truncation.json')
    for split in ['train','validation']:
        s.append(f'{split.capitalize()} truncated records & {trunc[split]["truncated_records"]}'+r'\\')
    test_trunc=sum(bool(r['prediction']['preprocessing']['omitted_sentence_ids']) for r in pred)
    s.append(f'Test truncated records & {test_trunc}'+r'\\\bottomrule\end{tabular}\end{table}')
    s.append(r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{training.png}\caption{Measured training and validation response loss. The dashed line marks the selected epoch.}\label{fig:training}\end{figure}')
    first,last=epochs[0]['validation_loss'],epochs[-1]['validation_loss']
    s.append(f'Table~\\ref{{tab:training}} records the selected adapter. Validation loss changed from {first:.4f} after epoch one to {last:.4f} after epoch three (Fig.~\\ref{{fig:training}}). Selection uses the minimum validation loss, not the final test score. Training loss windows are descriptive and are not an unbiased epoch-average curve.')
    for split,title in [('validation','Validation comparison'),('test','Held-out comparison')]:
        s.append(r'\begin{table}[t]\centering\small\caption{'+title+r'. Acc=accuracy; MF=macro-F1; EF=evidence micro-F1.}\label{tab:'+split+r'}\begin{tabular}{llrrr}\toprule Model & Criteria & Acc & MF & EF\\\midrule')
        for model in ['base','adapted']:
            for group in ['overall','inclusion','exclusion']:
                r=metrics[f'{split}_{model}'][group]
                s.append(f'{model} & {group} & {r["accuracy"]:.3f} & {r["macro_f1"]:.3f} & {r["evidence"]["micro_f1"]:.3f}'+r'\\')
        s.append(r'\bottomrule\end{tabular}\end{table}')
    b,a=metrics['test_base']['overall'],metrics['test_adapted']['overall']
    s.append(f'Tables~\\ref{{tab:validation}} and~\\ref{{tab:test}} compare identical examples and grammar settings before and after adaptation. Held-out accuracy changes from {b["accuracy"]:.3f} to {a["accuracy"]:.3f}, while macro-F1 changes from {b["macro_f1"]:.3f} to {a["macro_f1"]:.3f}. These measures must be interpreted together: frequent-label success can raise accuracy while rare classes remain poorly recognized. The results do not establish clinical reliability.')
    training_rows=read_split('train')
    majority={kind:Counter(r['label'] for r in training_rows if r['criterion_type']==kind).most_common(1)[0][0]
              for kind in ('inclusion','exclusion')}
    majority_rows=[dict(r,prediction={'label':majority[r['criterion_type']],'evidence':[]}) for r in pred]
    reference=compute(majority_rows)
    (ROOT/'results/train_majority_reference.json').write_text(json.dumps({'labels_from_training_only':majority,'metrics':reference},indent=2))
    s.append(f'A diagnostic majority-label reference chooses the most frequent training label separately for each criterion type, with no test-derived choice. It obtains held-out accuracy {reference["overall"]["accuracy"]:.3f} and macro-F1 {reference["overall"]["macro_f1"]:.3f}. This simple reference contextualizes the adapted model: improvement over an unadapted decoder does not by itself demonstrate strong clinical discrimination. The reference performs no evidence retrieval.')
    s.append(f'Evidence micro-F1 changes from {b["evidence"]["micro_f1"]:.3f} before adaptation to {a["evidence"]["micro_f1"]:.3f} afterwards. A label-accuracy gain therefore must not be described as a general evidence-quality gain. Many zero evidence flags and imbalanced label frequencies are plausible contributors, but this study does not establish their causal effect through an ablation.')
    distributions={kind:dict(Counter(r['prediction']['label'] for r in pred if r['criterion_type']==kind))
                   for kind in ('inclusion','exclusion')}
    empty_predictions=sum(not r['prediction']['evidence'] for r in pred)
    dist_text='; '.join(kind+': '+', '.join(f'{label} ({count})' for label,count in labels.items())
                       for kind,labels in distributions.items())
    s.append('The actual prediction distribution is '+tex(dist_text)+'. '+
             f'No sentence was cited in {empty_predictions} of {len(pred)} test records. '+
             'Thus syntactic preservation of all six labels must not be confused with successful recognition of all six classes. The concentration on frequent outcomes and weak evidence selection are substantive failures of the adapted matcher. Verified inference mechanisms do not establish clinical reasoning quality.')
    s.append(r'\begin{table}[t]\centering\scriptsize\caption{Adapted model: held-out per-class classification. All six labels remain distinct.}\label{tab:classes}\begin{tabular}{lrrrr}\toprule Label & Prec. & Recall & F1 & N\\\midrule')
    for label in LABELS.values():
        r=a['per_class'][label]
        s.append(f'{tex(label)} & {r["precision"]:.3f} & {r["recall"]:.3f} & {r["f1-score"]:.3f} & {int(r["support"])}'+r'\\')
    s.append(r'\bottomrule\end{tabular}\end{table}')
    s.append(r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{confusion.png}\caption{Adapted held-out confusion matrices. Rows are expert labels and columns are predictions. INC=included; NI=not included; EXC=excluded; NE=not excluded; INFO=not enough information; NA=not applicable.}\label{fig:confusion}\end{figure*}')
    s.append(r'Table~\ref{tab:classes} and Fig.~\ref{fig:confusion} expose minority-class errors that overall accuracy obscures. Zero-support classes remain in the specified macro denominator. Per-type per-class results and both base/adapted confusion matrices are also saved in the machine-readable metric files.')
    s.append(r'\begin{table}[t]\centering\scriptsize\caption{Adapted held-out evidence retrieval. Both-empty pairs score one per record, but contribute no pooled counts.}\label{tab:evidence}\begin{tabular}{lrrrrrr}\toprule & \multicolumn{3}{c}{Micro} & \multicolumn{3}{c}{Macro}\\Group & P & R & F1 & P & R & F1\\\midrule')
    for group in ['overall','inclusion','exclusion']:
        e=metrics['test_adapted'][group]['evidence']
        s.append(group+' & '+' & '.join(f'{e[k]:.3f}' for k in ['micro_precision','micro_recall','micro_f1','macro_precision','macro_recall','macro_f1'])+r'\\')
    s.append(r'\bottomrule\end{tabular}\end{table}')
    s.append(f'Table~\\ref{{tab:evidence}} includes {a["evidence"]["both_empty_count"]} test records where both evidence sets are empty. Differences between micro and macro scores reflect both citation-set sizes and the explicit empty-set convention. Evidence scores assess localization; they do not prove that cited sentences logically entail the model label.')
    s.append(r'\begin{table*}[t]\centering\small\caption{Adapted CPU decoding measurements. Median of three repetitions after one warmup; peak RSS is a fresh-process high-water mark including loaded model(s).}\label{tab:decode}\begin{tabular}{rrlrrrrr}\toprule Input & Output & Path & Prefill (s) & Decode (s) & Total (s) & Tokens/s & Peak RSS (MiB)\\\midrule')
    for r in bench['records']:
        if r['kind']=='decoding':
            s.append(f'{r["input_tokens"]} & {r["output_tokens"]} & {r["mode"]} & {r["median_prefill_s"]:.3f} & {r["median_decode_s"]:.3f} & {r["median_s"]:.3f} & {r["median_tokens_per_s"]:.2f} & {r["peak_process_rss_mib"]:.1f}'+r'\\')
    s.append(r'\bottomrule\end{tabular}\end{table*}')
    s.append(r'\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{latency.png}\caption{Measured end-to-end generation latency, including prefill.}\label{fig:latency}\end{figure}')
    equality=all(r.get('agrees_with_uncached',True) for r in bench['records'])
    s.append(f'Token-level agreement across all three final length pairs is {"verified" if equality else "NOT verified"}. Table~\\ref{{tab:decode}} and Fig.~\\ref{{fig:latency}} separate prefill from incremental decoding. Caching avoids repeated prefix computation but retains KV memory. Process RSS includes fixed framework and weight costs, so it is not an isolated cache-size measure.')
    cache_speedups=[r['speedup_vs_uncached'] for r in bench['records'] if r['kind']=='decoding' and r['mode']=='cached']
    s.append(f'Across the measured length pairs, cached generation is {min(cache_speedups):.2f}--{max(cache_speedups):.2f} times as fast as the uncached reference by median end-to-end latency. These ratios apply to this target, CPU, prompt family and output budgets; they are not universal model speedups.')
    specs=[r for r in bench['records'] if r['kind']=='decoding' and r['mode']=='speculative']
    s.append(r'\begin{table}[t]\centering\scriptsize\caption{Speculative acceptance and speed relative to ordinary cached greedy. Speedup below one means slower.}\label{tab:spec}\begin{tabular}{rrrr}\toprule Input/output & Accepted/proposed & Acceptance & Speedup\\\midrule')
    for r in specs:
        run=r['runs'][0]
        s.append(f'{r["input_tokens"]}/{r["output_tokens"]} & {run["accepted"]}/{run["proposed"]} & {run["acceptance_rate"]:.3f} & {r["speedup_vs_cached"]:.3f}'+r'\\')
    s.append(r'\bottomrule\end{tabular}\end{table}')
    s.append(r'Table~\ref{tab:spec} reports proposal counts from the first deterministic run and the median latency ratio. Low acceptance requires repeated draft forwards and target corrections. A smaller model is therefore insufficient by itself to guarantee acceleration. The pruned draft is a practical memory experiment with limited calibration, and the measured ratios bound claims about its usefulness.')
    s.append(r'\begin{table}[t]\centering\scriptsize\caption{Educational attention measurements (batch 1, heads 3, width 32). Error is against the dense reference.}\label{tab:attention}\begin{tabular}{rlrrr}\toprule Length & Path & Time (ms) & Max error & RSS MiB\\\midrule')
    for r in bench['records']:
        if r['kind']=='attention':
            s.append(f'{r["shape"][-2]} & {r["mode"]} & {r["median_s"]*1000:.3f} & {r["max_abs_error"]:.1e} & {r["peak_process_rss_mib"]:.1f}'+r'\\')
    s.append(r'\bottomrule\end{tabular}\end{table}')
    att=[r for r in bench['records'] if r['kind']=='attention' and r['mode']=='tiled']
    s.append(f'Table~\\ref{{tab:attention}} shows a maximum tiled numerical error of {max(r["max_abs_error"] for r in att):.2e}; the largest mean error is {max(r["mean_abs_error"] for r in att):.2e}. The logical score tile stays at most {max(r["tile_score_mib"] for r in att):.4f} MiB, whereas the dense score tensor reaches {max(r["dense_score_mib"] for r in att):.2f} MiB. These buffer estimates exclude other intermediates and are not claimed as peak process-memory savings. Python loops and small tensor dispatch can outweigh tiling benefits on CPU.')
    largest_tile=max(att,key=lambda r:r['shape'][-2])
    largest_ref=next(r for r in bench['records'] if r['kind']=='attention' and r['mode']=='reference' and r['shape']==largest_tile['shape'])
    s.append(f'At sequence length {largest_tile["shape"][-2]}, the tiled implementation takes {largest_tile["median_s"]*1000:.3f} ms versus {largest_ref["median_s"]*1000:.3f} ms for dense attention. This directly measures the educational implementation, not a fused FlashAttention kernel.')

    source={r['annotation_id']:r for r in read_split('test')}
    errors=[r for r in pred if r['gold_label']!=r['prediction']['label'] or set(r['gold_evidence'])!=set(r['prediction']['evidence'])]
    def tags(r):
        p=r['prediction'];g=set(r['gold_evidence']);e=set(p['evidence']);t=[]
        if r['gold_label']=='not enough information' and p['label']!='not enough information':t.append('unsupported certainty')
        if p['label']=='not enough information' and r['gold_label']!='not enough information':t.append('over-abstention')
        if 'not applicable' in (r['gold_label'],p['label']) and r['gold_label']!=p['label']:t.append('applicability confusion')
        if r['gold_label']=='excluded' and p['label']!='excluded':t.append('missed exclusion')
        if r['gold_label']=='not included' and p['label']!='not included':t.append('missed inclusion conflict')
        if e-g:t.append('extra citation')
        if g-e:t.append('missed citation')
        if not e and g:t.append('empty predicted evidence')
        if r['gold_label']!=p['label'] and not t:t.append('label confusion')
        return t
    selected_errors=[];covered=set()
    for r in errors:
        if set(tags(r))-covered:
            selected_errors.append(r);covered.update(tags(r))
    for diversify in (True,False):
        for r in errors:
            if len(selected_errors)>=8:break
            used_patients={x['patient_id'] for x in selected_errors}
            if r not in selected_errors and (not diversify or r['patient_id'] not in used_patients):
                selected_errors.append(r)
    selected_errors=selected_errors[:8]
    if len(selected_errors)<8:raise ValueError('Fewer than eight distinct errors; report the observed count honestly.')
    (ROOT/'results/error_cases.json').write_text(json.dumps([dict(r,source=source[r['annotation_id']],error_tags=tags(r)) for r in selected_errors],indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# Held-out error analysis','Eight distinct measured errors, selected after final evaluation. No model or prompt changes follow this analysis.','']
    s.append(r'\begin{table*}[t]\centering\scriptsize\caption{Eight distinct adapted-model held-out error cases. Full notes, IDs, gold and predicted evidence are preserved in the error-analysis file.}\label{tab:errors}\begin{tabular}{rp{4.4cm}p{2.1cm}p{2.1cm}p{4.2cm}}\toprule ID & Criterion (abbreviated) & Expert label & Prediction & Observed error\\\midrule')
    for r in selected_errors:
        row=source[r['annotation_id']];p=r['prediction'];tag=', '.join(tags(r))
        text=row['criterion_text'];short=text[:130]+('...' if len(text)>130 else '')
        s.append(f'{r["annotation_id"]} & {tex(short)} & {tex(r["gold_label"])} & {tex(p["label"])} & {tex(tag)}'+r'\\')
        md += [f'## Annotation {r["annotation_id"]} / {r["patient_id"]}',f'- Type: {r["criterion_type"]}',f'- Criterion: {text}',f'- Expert label: {r["gold_label"]}; model label: {p["label"]}',f'- Expert evidence: {r["gold_evidence"]}; model evidence: {p["evidence"]}',f'- Observed error: {tag}',f'- Missing gold citations: {sorted(set(r["gold_evidence"])-set(p["evidence"]))}; unsupported extra citations: {sorted(set(p["evidence"])-set(r["gold_evidence"]))}', '',row['note'],'']
    s.append(r'\bottomrule\end{tabular}\end{table*}')
    s.append(r'Table~\ref{tab:errors} distinguishes label errors from sentence-localization errors. Unsupported certainty is defined relative to an expert insufficient-information label, not an independently established medical diagnosis. Extra/missed citations compare source-ID sets. The cases show why a valid citation ID and a grammatical output cannot establish factual support. The full error log retains contradictory expert/model outcomes and source text for individual review.')
    by_id={r['annotation_id']:r for r in pred}
    if 161 in by_id and by_id[161]['gold_label']=='not included' and by_id[161]['prediction']['label']=='included':
        s.append('Annotation 161 is an unsupported positive result: the criterion specifies elective carotid endarterectomy, whereas cited sentence 0 describes cardiac catheterization. The citation is a valid source location but does not state the required procedure. This demonstrates the gap between source-ID validity and criterion entailment.')
    if 169 in by_id and by_id[169]['gold_label']=='not enough information' and by_id[169]['prediction']['label']=='not excluded':
        s.append('Annotation 169 asks about a pre-operative renal-filtration threshold. The supplied note contains no such measurement. The expert label is insufficient information, but the model returns not excluded with no citation. The app flags the absent evidence for review; the predicted negative outcome must not be read as proof that the threshold is not met.')
    if 185 in by_id and by_id[185]['gold_label']=='included' and by_id[185]['prediction']['label']=='not enough information':
        s.append('Annotation 185 combines under-recognition and failed localization. The experts associate the inclusion criterion with sentences 2 and 3, which report observations and vital signs; the model instead abstains and retrieves neither sentence. The error concerns the benchmark expert judgment, not an independent clinical diagnosis made by this system.')
    stress=load('stress_cases.json')['records']
    s.append(r'\begin{table}[t]\centering\scriptsize\caption{Supplementary synthetic probes. These are not benchmark labels and are excluded from accuracy/F1.}\label{tab:stress}\begin{tabular}{p{2.5cm}p{3.1cm}l}\toprule Probe & Model output & Cited IDs\\\midrule')
    stress_names={'contradictory_age':'Conflicting ages','missing_renal_information':'Missing kidney data','explicit_negation':'Explicit negation'}
    for r in stress:
        s.append(f'{stress_names[r["case"]["case_id"]]} & {tex(r["prediction"]["label"])} & {tex(r["prediction"]["evidence"])}'+r'\\')
    s.append(r'\bottomrule\end{tabular}\end{table}')
    s.append(r'Table~\ref{tab:stress} tests three deliberately constructed situations after protocol freeze. The conflicting-age record contains both 52 and 72 years without resolving the discrepancy; citing a single age cannot resolve that contradiction. The missing-kidney-information case provides no renal measurement; a definite category would not establish that missing fact. The negation case distinguishes a condition mention from its presence. These probes illustrate behaviors for human review, not additional ground-truth benchmark performance or a basis for retuning.')
    screenshot=figdir/'application.png'
    if not screenshot.exists():screenshot=figdir/'application.jpg'
    if screenshot.exists():
        s.append(r'\begin{figure*}[t]\centering\includegraphics[width=\textwidth]{'+screenshot.name+r'}\caption{Actual local browser screening interface. Source citations and model predictions remain visible for review.}\label{fig:app}\end{figure*}')
        s.append(r'Figure~\ref{fig:app} shows the running application. Its inference tab computes new measurements, and exported reports retain the source record and model provenance.')
    (ROOT/'report/generated_results.tex').write_text('\n\n'.join(s),encoding='utf-8')
    (ROOT/'results/error_analysis.md').write_text('\n'.join(md),encoding='utf-8')
    (ROOT/'results/summary.json').write_text(json.dumps({'selected_epoch':selected['selected_epoch'],
                'test_base':b,'test_adapted':a,'token_equality':equality},indent=2))
    card=f'''# Criterion CPU LoRA adapter

Base: HuggingFaceTB/SmolLM2-135M-Instruct, revision 12fd25f77366fa6b3b4b768ec3050bf629380bac (Apache-2.0).

Author/submission: Muhammad Hashim Rahim, i230028, FAST NUCES, Generative AI. OpenAI Codex assisted implementation and report preparation.

Adaptation: 92,160 trainable rank-eight query/value adapter parameters in the last six blocks; alpha 16; three fixed epochs; float32 CPU; AdamW learning rate 0.001; batch one, accumulation eight; seed 42. Selected epoch: {selected['selected_epoch']}, using validation response loss {selected['validation_loss']:.4f}.

Data: ncbi/TrialGPT-Criterion-Annotations, public-domain dataset. Fixed patient-grouped 37/8/8 split, 603/206/206 criterion rows. GPT-generated fields never enter model features. The complete patient-ID split is in data/split_manifest.json.

Held-out accuracy: {a['accuracy']:.4f}; six-class macro-F1: {a['macro_f1']:.4f}; evidence micro-F1: {a['evidence']['micro_f1']:.4f}. Consult the report and full JSON metrics for minority-class errors, evidence conventions and limitations. No clinical reliability claim is made.

Observed prediction distribution: {json.dumps(distributions)}. No sentence was cited in {empty_predictions}/{len(pred)} test records. In particular, the model's concentration on frequent labels and weak exclusion evidence retrieval are major measured failures, even though adaptation improves accuracy over the unadapted decoder.

Output: one constrained label code followed by one evidence bit per patient sentence, parsed into label and original sentence IDs. Use trialmatch.pipeline.predict rather than loading this small adapter file as a complete language model.

The adapter file contains only LoRA matrices. Download public base weights with scripts/download.py and load the adapter through trialmatch.model.load. The uncalibrated six-layer speculative draft is constructed from base weights and does not use this adapter.

Educational screening prototype. Not a clinical enrollment decision. The general-purpose small model can produce incorrect labels and irrelevant/missing citations. Template explanations do not prove clinical entailment. All outputs require human review.
'''
    (ROOT/'models/adapter/MODEL_CARD.md').write_text(card,encoding='utf-8')
    print('Measured tables, figures, summary and eight error cases generated.')

if __name__=='__main__':main()
