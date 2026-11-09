"""Create a measured GPU companion report after local results verification."""
import argparse
import json
import statistics
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    folder = args.results
    read = lambda name: json.loads((folder / name).read_text())
    verification = read('local_verification.json')
    protocol, selection = read('protocol.json'), read('selection.json')
    history = read('validation_history.json')
    base = read('base_validation_metrics.json')['overall']
    benchmarks = read('gpu_benchmarks.json')
    selected = selection['validation']
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9.5
    styles['BodyText'].leading = 12
    styles['BodyText'].allowWidows = 0
    styles['BodyText'].allowOrphans = 0
    styles['Heading1'].keepWithNext = True
    styles['Heading2'].keepWithNext = True
    story = []
    def p(text, style='BodyText'):
        story.append(Paragraph(text, styles[style]))
        if style == 'BodyText': story.append(Spacer(1, 4))
    def table(rows, widths):
        rows = [[Paragraph(str(v), styles['BodyText']) for v in row] for row in rows]
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5edf6')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#bcc5cf')),
                              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                              ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.append(t); story.append(Spacer(1, 10))
    p('Criterion: Kaggle GPU Study', 'Title')
    p('Muhammad Hashim Rahim | i230028<br/>FAST NUCES | Generative AI')
    p('Exploratory extension to the original CPU assignment', 'Heading2')
    p('This companion records a private Kaggle GPU experiment. The original CPU study, its adapter, '
      'and its held-out test results remain unchanged. Because those test results were already known '
      'before this extension, this study is exploratory. Training and checkpoint selection use only '
      'the original training and validation patients; there is no new blind-test performance claim.')
    p('Protocol and selection', 'Heading2')
    p(f"Hardware: {protocol['gpu']} ({protocol['gpu_memory_gib']:.1f} GiB visible to one device). "
      f"PyTorch {protocol['torch']}, CUDA {protocol['cuda']}. Kaggle allocated T4 x2; the study uses device 0, "
      'without multi-GPU training. Model: SmolLM2-360M-Instruct, with rank-16, alpha-32 LoRA on '
      'query/value projections in every block. The original 135M checkpoint is a separate model.')
    p('Data: 603 training criteria from 37 patients and 206 validation criteria from 8 disjoint patients. '
      'No test annotations or prior test predictions were uploaded. Both predetermined objectives use '
      'three epochs, learning rate 0.0005, gradient accumulation 8 and maximum context 1024. '
      'Training uses AMP with float32 master weights; inference uses float32. Seed is 42.')
    p('Uniform loss averages label and evidence-token cross entropy. Balanced loss gives half the '
      'objective to the class-weighted label and half to the evidence bitmap, weighting positive '
      'evidence using training-only statistics. Selection maximizes the mean of generated validation '
      'classification macro-F1 and evidence micro-F1; ties retain the earlier checkpoint.')
    p('Generated validation results', 'Heading2')
    rows = [['Model / objective', 'Epoch', 'Accuracy', 'Macro-F1', 'Evidence F1', 'Selection']]
    rows.append(['360M base', '-', f"{base['accuracy']:.4f}", f"{base['macro_f1']:.4f}", f"{base['evidence']['micro_f1']:.4f}", '-'])
    for r in history:
        rows.append([r['objective'], r['epoch'], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}",
                     f"{r['evidence_micro_f1']:.4f}", f"{r['selection_score']:.4f}"])
    table(rows, [122, 52, 72, 72, 88, 72])
    p(f"Selected: <b>{selected['objective']}, epoch {selected['epoch']}</b>. Its validation accuracy is "
      f"{selected['accuracy']:.4f}, classification macro-F1 {selected['macro_f1']:.4f}, and evidence "
      f"micro-F1 {selected['evidence_micro_f1']:.4f}. These are checkpoint-selection scores on a small "
      'validation set, not estimates from an untouched test cohort.')
    p(f"Against the same 360M base, accuracy rose by {(selected['accuracy']-base['accuracy'])*100:.2f} "
      f"percentage points, macro-F1 by {selected['macro_f1']-base['macro_f1']:.4f}, and evidence F1 by "
      f"{selected['evidence_micro_f1']-base['evidence']['micro_f1']:.4f}. Uniform epoch 2 performed worse "
      'than epoch 1 despite lower training loss, and the balanced objective did not win selection. '
      'The selected checkpoint still has zero recall for not included (3 examples), excluded '
      '(2 examples), and not applicable (12 examples). It correctly identifies 18 of 53 '
      'insufficient-information cases. Higher overall accuracy does not establish reliable rare-label reasoning.')
    p('GPU inference and attention checks', 'Heading1')
    p('Each configuration receives one warmup and three measured repeats. CUDA is synchronized '
      'around the decoding timers; latency includes prefill. The speculative draft is the full '
      'pretrained SmolLM2-135M-Instruct. Vocabulary and special-token IDs must match the target. '
      'Every measured token sequence must exactly match uncached greedy decoding.')
    rows = [['Input / output tokens', 'Uncached (s)', 'Cached (s)', 'Speculative (s)', 'Cache speedup']]
    decoding = [r for r in benchmarks['records'] if r['kind'] == 'decoding']
    for length in (32, 128, 256):
        rs = {r['mode']: r for r in decoding if r['input_tokens'] == length}
        u, c, s = [rs[k]['median_s'] for k in ('uncached','cached','speculative')]
        rows.append([f"{length} / {rs['cached']['output_tokens']}", f'{u:.4f}', f'{c:.4f}', f'{s:.4f}', f'{u/c:.2f}x'])
    table(rows, [130, 85, 85, 85, 83])
    p('Token equality passed for all three input lengths across uncached, cached and speculative '
      'decoding, including all repeats. Whether speculative decoding is faster must be judged from '
      'the measured latency; a smaller draft alone does not guarantee a speedup.')
    rows = [['Input / mode', 'Prefill (ms)', 'Decode (ms)', 'Tokens/s', 'Peak alloc. MiB']]
    for r in decoding:
        runs = r['runs']
        rows.append([f"{r['input_tokens']} / {r['mode']}",
                     f"{statistics.median(x['prefill_s'] for x in runs)*1000:.2f}",
                     f"{statistics.median(x['decode_s'] for x in runs)*1000:.2f}",
                     f"{statistics.median(x['tokens_per_s'] for x in runs):.1f}",
                     f"{r['peak_gpu_allocated_mib']:.1f}"])
    table(rows, [138, 82, 82, 76, 90])
    p('Prefill and decode columns are medians across the three runs; throughput is total emitted '
      'tokens divided by total latency in each run. Their separate medians need not sum to median '
      'total latency. The uncached path repeatedly recomputes the full prefix during decoding. '
      'Speculative peak allocation includes both target and draft models.')
    rows = [['Input tokens', 'Proposed', 'Accepted', 'Acceptance', 'Speedup vs cached']]
    for r in decoding:
        if r['mode'] != 'speculative': continue
        proposed = sum(x['proposed'] for x in r['runs'])
        accepted = sum(x['accepted'] for x in r['runs'])
        cached = next(x for x in decoding if x['mode']=='cached' and x['input_tokens']==r['input_tokens'])
        rows.append([r['input_tokens'], proposed, accepted, f'{accepted/max(1,proposed):.3f}',
                     f"{cached['median_s']/r['median_s']:.2f}x"])
    table(rows, [90, 85, 85, 95, 113])
    p('Acceptance counts pool all three measured repeats and exclude warmup. All proposals matched '
      'in these benchmarks, yet speculation was slower (0.74-0.81x cached speed). Extra serial draft '
      'forwards and cache maintenance outweighed block-verification savings on these short outputs. '
      'Cache reuse gave little benefit for the two shorter inputs, rising to 1.43x at 256 input tokens.')
    p('Tiled attention and reproducibility', 'Heading1')
    rows = [['Queries / keys', 'Causal', 'Reference (ms)', 'Tiled (ms)', 'Max abs. error']]
    attention = [r for r in benchmarks['records'] if r['kind'] == 'attention']
    for r in attention:
        if r['mode'] != 'reference': continue
        t = next(t for t in attention if t['mode']=='tiled' and t['query_length']==r['query_length'] and t['key_length']==r['key_length'])
        rows.append([f"{r['query_length']} / {r['key_length']}", str(r['causal']), f"{r['median_s']*1000:.3f}",
                     f"{t['median_s']*1000:.3f}", f"{r['max_abs_error']:.2e}"])
    table(rows, [105, 60, 108, 105, 90])
    p('The tiled online-softmax implementation is a correctness and memory demonstration in PyTorch, '
      'not a fused FlashAttention kernel. Checks cover square causal attention, an offset causal '
      'case (19 queries / 28 keys), and a noncausal case. Agreement uses atol=2e-6 and rtol=2e-5.')
    p('Tiling agreed numerically but was slower than the dense reference in every tested shape. '
      'Many small Python-dispatched GPU operations dominate this educational implementation; '
      'a fused attention kernel is a different performance proposition.')
    p('Memory and reproducibility', 'Heading2')
    p('The accompanying JSON records reset peak CUDA tensor allocation and reservation for each '
      'configuration, including warmup and active models. These shared-process measurements are '
      'not total board memory or host RSS. The draft is loaded only after cached/uncached comparisons. '
      'Pinned model revisions, source/data hashes, all epoch predictions, training logs, adapter '
      'weights and raw repeated timings accompany this report.')
    p('Local verification recomputed all seven validation metric files from 206 predictions each, '
      'checked them against the original validation annotations, confirmed checkpoint selection, '
      'and rechecked saved decoding equality and attention errors. The 360M adapter requires its '
      'matching base model and loader; it must not replace the original 135M adapter.')
    p('Limitations', 'Heading2')
    p('Two objectives and six checkpoints are compared on one small validation cohort. Selection '
      'can overfit those patients; no confidence intervals or broad clinical generalization are '
      'claimed. This is a research demonstration requiring human review, not a clinical eligibility '
      'decision system. Source project: '
      '<link href="https://github.com/50022005/GenerativeAi_2">github.com/50022005/GenerativeAi_2</link>.')
    p('AI-use disclosure: OpenAI Codex assisted with this extension\'s implementation, debugging, '
      'execution checks and report preparation. The student must review and understand the code '
      'and independently explain the results during the required defense.')
    p('Sources and run', 'Heading2')
    p('<link href="https://www.kaggle.com/code/hashimrahim/criterion-gpu-validation-study">Private Kaggle notebook</link> '
      '(version 1); <link href="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct">360M target model card</link>; '
      '<link href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct">135M draft model card</link>. '
      'Original data provenance and clinical-task references are documented in the main report.')
    p('Downloaded archive SHA-256: ' + verification['archive_sha256'], 'Code')
    output = ROOT / 'output/pdf/Criterion_Kaggle_Study.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#526071'))
        canvas.drawString(42, 24, 'Criterion | Exploratory GPU study | i230028')
        canvas.drawRightString(570, 24, str(doc.page))
    SimpleDocTemplate(str(output), pagesize=(612,792), rightMargin=42, leftMargin=42,
                      topMargin=38, bottomMargin=42).build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == '__main__': main()
