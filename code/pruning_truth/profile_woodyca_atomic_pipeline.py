from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import inspect
import time

import torch

import profile_atomic_pruning_pipeline as continuous
import profile_pruning_components as runtime
import profile_pruning_structure as structure
import logic_models


CHECKPOINT = runtime.ROOT / (
    '04_results/cloud_handoffs/WoodyDA-WoodyCAR_Ablation_Results_20260807_full/'
    'WoodyDA-WoodyCAR_Ablation_Results_20260807/runs/'
    'branchseg_csnet_v2_channel_only_channel_only_roi_20260807_s42/checkpoints/best_primary.pth'
)
CHECKPOINT_SHA256 = 'b7016e1d7deeb4dd0a1010298fc4c68a1e71fd55943a5ec59db3787180b54f1d'
OLD = runtime.ROOT / '04_results/pruning_decision/runtime_profiling_v1'
COMPONENTS = ('roi', 'woody', 'tape_support', 'bud', *continuous.STRUCTURAL_STAGES,
              'graphsage_ensemble')


def load_woodyca(manager):
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = logic_models.CSNet(in_channels=3, n_classes=1,
                              use_channel_attention=True, use_spatial_attention=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    manager.branch_model = model.to(manager.device).eval()
    return manager.branch_model


class ComponentTimers:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.rows = {}

    def wrap(self, name, call, model=None):
        def measured(*args, **kwargs):
            if name in self.rows:
                raise RuntimeError(f'Unexpected repeated stage: {name}')
            if model is not None:
                result, timing = runtime.ForwardTimer(model).measure(lambda: call(*args, **kwargs))
            else:
                runtime.sync()
                started = time.perf_counter()
                result = call(*args, **kwargs)
                runtime.sync()
                timing = {'total_seconds': time.perf_counter() - started}
            self.rows[name] = timing
            return result
        return measured

    @contextmanager
    def enabled(self):
        self.rows = {}
        with ExitStack() as stack:
            targets = [
                (runtime.baseline, 'apply_roi_filter', 'roi', self.pipeline.roi),
                (runtime.baseline, 'run_branch_segmentation', 'woody', self.pipeline.woody),
                (runtime.routing, 'prepare_processed_router_mask', 'tape_support', self.pipeline.tape),
                (runtime.logic_bud, 'run_bud_global_detection', 'bud', self.pipeline.bud),
                (self.pipeline, 'predict_sage', 'graphsage_ensemble', None),
            ]
            for owner, attribute, stage, model in targets:
                stack.enter_context(runtime.replace_callable(
                    owner, attribute, self.wrap(stage, getattr(owner, attribute), model)))
            original = structure.make_call

            def timed_factory(stage, *args, **kwargs):
                return self.wrap(stage, original(stage, *args, **kwargs))

            stack.enter_context(runtime.replace_callable(structure, 'make_call', timed_factory))
            yield
        if set(self.rows) != set(COMPONENTS):
            raise RuntimeError(f'Incomplete component observations: {set(self.rows)}')


def checked_result(reference_signature, reference_scores, result):
    observed = continuous.output_signature(result)
    checks = {stage: {'exact_equal': expected == observed[stage],
                      'reference_sha256': expected, 'observed_sha256': observed[stage]}
              for stage, expected in reference_signature.items()}
    checks['graphsage_ensemble'] = continuous.score_comparison(reference_scores, result['graphsage_ensemble'])
    return checks


def verify_checks(checks):
    if not all(row['exact_equal'] for stage, row in checks.items() if stage != 'graphsage_ensemble'):
        raise RuntimeError('A non-GraphSAGE output differs from the uninstrumented reference')
    graph = checks['graphsage_ensemble']
    if not graph['candidate_set_equal'] or not graph['candidate_order_equal']:
        raise RuntimeError('GraphSAGE candidate set or order differs from the reference')


def main():
    parser = argparse.ArgumentParser(description='Existing atomic pipeline with verified paper-facing WoodyCA-Net')
    parser.add_argument('--output', type=runtime.Path,
                        default=runtime.ROOT / '04_results/pruning_decision/runtime_profiling_woodyca_20260903')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if min(args.warmup, args.repeats) < 1:
        parser.error('warmup and repeats must be positive')
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('Use a new output directory; existing measurements are read-only')
    if runtime.baseline.sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError('WoodyCA-Net checkpoint hash mismatch')
    frozen = runtime.freeze_samples(OLD, 10, 20260903)
    sources = {}
    for name in ('run_manifest.json', 'structural_run_manifest.json'):
        sources.update(runtime.read_json(OLD / name)['source_sha256'])
    for relative, expected in sources.items():
        if runtime.baseline.sha256_file(runtime.ROOT / relative) != expected:
            raise RuntimeError(f'Existing inference source changed: {relative}')
    for path in [runtime.Path(__file__), runtime.Path(continuous.__file__),
                 runtime.Path(continuous.sage.__file__), runtime.Path(continuous.sage.original.__file__),
                 runtime.Path(inspect.getfile(logic_models.CSNet))]:
        sources[str(path.relative_to(runtime.ROOT))] = runtime.baseline.sha256_file(path)
    previous = runtime.read_json(OLD / 'atomic_e2e_same_process/run_manifest.json')
    weights = dict(previous['weights'])
    weights['branch'] = {'path': str(CHECKPOINT.relative_to(runtime.ROOT)), 'sha256': CHECKPOINT_SHA256}
    for item in weights.values():
        if runtime.baseline.sha256_file(runtime.ROOT / item['path']) != item['sha256']:
            raise RuntimeError(f"Weight mismatch: {item['path']}")
    rule_paths = [
        runtime.ROOT / '04_results/mask_topology_routing/gt_mask_thickness_rulebook_20260731/gt_mask_thickness_rulebook.json',
        structure.load_binary_v3_config().rulebook_path,
    ]
    rule_hashes = {str(path.relative_to(runtime.ROOT)): runtime.baseline.sha256_file(path) for path in rule_paths}
    runtime.baseline.ensure_dir(output)
    runtime.save_json(output / 'sample_manifest.json', frozen)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    manifest = {
        'status': 'preregistered_before_measurement',
        'model': 'WoodyCA-Net / csnet_v2_channel_only; strict checkpoint loading',
        'authorized_change': 'load verified channel-only checkpoint; original inference and postprocessing functions unchanged',
        'scope': 'atomic-only joint HGB and five-seed GraphSAGE output; no region aggregation/ranking',
        'ranker_scope': 'saved fixed-split atomic rankers; inference latency, not cross-validation training cost',
        'component_protocol': 'separate instrumented continuous calls; stage intervals around unchanged functions; not used as end-to-end latency',
        'end_to_end_protocol': 'separate calls without component/forward instrumentation; one outer CUDA-synchronized wall-clock timer',
        'excluded': ['model loading', 'image decoding', 'saving', 'profiling output checks'],
        'preserved': ['second ROI within bud entry', 'original bud drawing', 'V3.1 structural audit'],
        'bud_component_boundary': 'original ROI-conditioned patch detector, excluding repeated ROI and rendering',
        'forward_boundary': 'detector forward includes native decoding/postprocessing; before/between/after forward are mixed processing boundaries',
        'warmup_full_calls_per_mode': args.warmup, 'repeats_per_image_per_mode': args.repeats,
        'reference': 'one uninstrumented same-process full call per image, compared against all timed calls',
        'n_images': 10, 'n_trees': 10, 'sample_manifest_sha256': runtime.baseline.sha256_file(OLD / 'sample_manifest.json'),
        'source_sha256': sources, 'weights': weights, 'rulebook_sha256': rule_hashes,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn_benchmark': torch.backends.cudnn.benchmark, 'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'torch_cpu_threads': torch.get_num_threads(), 'opencv_threads': runtime.cv2.getNumThreads(),
        'hardware': runtime.read_json(OLD / 'hardware_snapshot.json'),
        'graphsage_score_gate': {'atol': 1e-7, 'rtol': 0, 'order_exact_required': True,
                                'policy': 'retain flag for numerical differences; never change gate or predictions'},
        'per_tree_multiview_measured': False, 'manuscript_updated': False,
    }
    runtime.save_json(output / 'run_manifest.json', manifest)
    component_rows = {name: [] for name in COMPONENTS}
    e2e_rows, audits = [], []
    try:
        with runtime.replace_callable(runtime.baseline.ModelManager, 'load_branch_model', load_woodyca):
            pipeline = continuous.AtomicPipeline(frozen, output)
        runtime.save_json(output / 'model_loading_runtime.json', pipeline.loading)
        timers = ComponentTimers(pipeline)
        first = frozen['samples'][0]
        rgb = runtime.baseline.load_image_rgb(runtime.ROOT / first['image_path'])
        for mode in ('components', 'end_to_end'):
            for index in range(args.warmup):
                if mode == 'components':
                    with timers.enabled():
                        warm = pipeline(rgb, first)
                else:
                    warm = pipeline(rgb, first)
                runtime.sync()
                del warm
                print(f'WARMUP {mode} {index + 1}/{args.warmup}', flush=True)
        for sample_index, sample in enumerate(frozen['samples']):
            sample_id = sample['sample_id']
            rgb = runtime.baseline.load_image_rgb(runtime.ROOT / sample['image_path'])
            reference = pipeline(rgb, sample)
            runtime.sync()
            expected = continuous.output_signature(reference)
            scores = reference['graphsage_ensemble'].copy()
            reference_dir = runtime.baseline.ensure_dir(output / 'uninstrumented_reference')
            runtime.save_json(reference_dir / f'{sample_id}_signature.json', expected)
            for stage in ('hgb_ranking', 'graphsage_ensemble'):
                reference[stage].to_csv(reference_dir / f'{sample_id}_{stage}.csv', index=False)
            del reference
            print(f'REFERENCE {sample_index + 1}/10 {sample_id}', flush=True)
            for mode in ('components', 'end_to_end'):
                for repeat in range(args.repeats):
                    if mode == 'components':
                        with timers.enabled():
                            result = pipeline(rgb, sample)
                    else:
                        runtime.sync()
                        started = time.perf_counter()
                        result = pipeline(rgb, sample)
                        runtime.sync()
                        seconds = time.perf_counter() - started
                    checks = checked_result(expected, scores, result)
                    audits.append({'sample_id': sample_id, 'mode': mode, 'repeat': repeat + 1, 'stages': checks})
                    runtime.save_json(output / 'equivalence_audit.json', audits)
                    verify_checks(checks)
                    base = {'sample_id': sample_id, 'tree_id': sample['tree_id'], 'repeat': repeat + 1}
                    if mode == 'components':
                        for stage in COMPONENTS:
                            component_rows[stage].append({**base, **timers.rows[stage]})
                            runtime.write_rows(output / 'components' / stage / 'per_image_runtime.csv', component_rows[stage])
                    else:
                        e2e_rows.append({**base, 'total_seconds': seconds, 'n_candidates': len(result['hgb_ranking'])})
                        runtime.write_rows(output / 'end_to_end/per_image_runtime.csv', e2e_rows)
                    ranking_dir = runtime.baseline.ensure_dir(output / 'ranked_candidates' / mode)
                    for stage in ('hgb_ranking', 'graphsage_ensemble'):
                        result[stage].to_csv(ranking_dir / f'{sample_id}_{stage}_repeat{repeat + 1}.csv', index=False)
                    del result
                    elapsed = f'{seconds:.4f}s' if mode == 'end_to_end' else 'stages recorded'
                    print(f'TIMED {mode} {sample_index + 1}/10 {sample_id} {repeat + 1}/{args.repeats} {elapsed}', flush=True)
        source_ok = {p: runtime.baseline.sha256_file(runtime.ROOT / p) == h for p, h in sources.items()}
        weights_ok = {key: runtime.baseline.sha256_file(runtime.ROOT / item['path']) == item['sha256'] for key, item in weights.items()}
        images_ok = {s['sample_id']: runtime.baseline.sha256_file(runtime.ROOT / s['image_path']) == s['image_sha256'] for s in frozen['samples']}
        rules_ok = {p: runtime.baseline.sha256_file(runtime.ROOT / p) == h for p, h in rule_hashes.items()}
        if not all(all(group.values()) for group in (source_ok, weights_ok, images_ok, rules_ok)):
            raise RuntimeError('Source, checkpoint, image or rulebook changed during profiling')
        summaries = {stage: runtime.summarize(rows) for stage, rows in component_rows.items()}
        for stage, summary in summaries.items():
            runtime.save_json(output / 'components' / stage / 'summary.json', summary)
        end_to_end = runtime.summarize(e2e_rows)
        runtime.save_json(output / 'end_to_end/summary.json', end_to_end)
        graph_checks = [item['stages']['graphsage_ensemble'] for item in audits]
        report = {
            'status': 'passed' if all(c['preset_score_gate_passed'] for c in graph_checks) else 'measured_with_graphsage_numerical_equivalence_flag',
            'end_to_end': end_to_end, 'components': summaries,
            'non_graphsage_outputs_exact': True, 'graphsage_candidate_order_exact': True,
            'graphsage_max_abs_score_delta': max(c['max_abs_score_delta'] for c in graph_checks),
            'graphsage_bitwise_exact': all(c['exact_equal'] for c in graph_checks),
            'sources_unchanged': source_ok, 'weights_unchanged': weights_ok,
            'images_unchanged': images_ok, 'rulebooks_unchanged': rules_ok,
            'manuscript_updated': False, 'per_tree_multiview_measured': False,
        }
        runtime.save_json(output / 'summary.json', report)
        print(f'COMPLETE {end_to_end}', flush=True)
    except Exception as error:
        runtime.save_json(output / 'failure.json', {'error': repr(error), 'completed_e2e_calls': len(e2e_rows),
                                                   'completed_component_calls': len(component_rows['roi'])})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
