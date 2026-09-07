from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

import profile_pruning_components as runtime
import profile_pruning_structure as structure
import profile_pruning_graphsage as sage


STRUCTURAL_STAGES = tuple(stage for stage in structure.STAGES if stage != 'region_features')


class AtomicPipeline:
    def __init__(self, frozen, output):
        self.defaults = frozen['defaults']
        self.output = output
        self.loading = []
        self.manager = runtime.baseline.ModelManager()
        self.manager.device = 'cuda:0'
        self.roi = self.load('roi', self.manager.load_roi_model)
        self.woody = self.load('woody', self.manager.load_branch_model)
        self.bud = self.load('bud', self.manager.load_bud_global_model)
        self.tape = self.load('tape', lambda: runtime.routing._get_tape_model('cuda:0'))
        self.config = structure.load_binary_v3_config()
        rulebook = runtime.ROOT / '04_results/mask_topology_routing/gt_mask_thickness_rulebook_20260731/gt_mask_thickness_rulebook.json'
        runtime.routing._choose_trunk_path_on_skeleton = structure.make_router_selector(rulebook, self.config.rulebook_path)
        closeout = runtime.ROOT / '04_results/pruning_decision/pruning_decision_closeout_20260809'
        self.hgb_path = closeout / 'baselines_v1/models/hgb_full.joblib'
        self.hgb = self.load('atomic_hgb', lambda: joblib.load(self.hgb_path))
        model_root = closeout / 'gnn_sage_branch_group_v1'
        self.seeds = runtime.read_json(model_root / 'summary.json')['seeds']
        self.models, self.checkpoints, self.paths = [], [], []
        for seed in self.seeds:
            path = model_root / 'checkpoints' / f'sage_full_seed{seed}.pt'

            def load_sage():
                checkpoint = torch.load(path, map_location='cpu')
                config = checkpoint['config']
                if config['architecture'] != 'sage' or config['graph_context'] != 'branch_group':
                    raise ValueError('Unexpected saved GraphSAGE configuration')
                model = sage.original.SegmentGNN(len(checkpoint['feature_columns']), config['hidden'],
                                                 config['dropout'], config['architecture'])
                model.load_state_dict(checkpoint['model_state'])
                return model.to('cuda:0').eval(), checkpoint

            model, checkpoint = self.load(f'atomic_graphsage_seed{seed}', load_sage)
            self.models.append(model)
            self.checkpoints.append(checkpoint)
            self.paths.append(path)
        columns = self.checkpoints[0]['feature_columns']
        if any(checkpoint['feature_columns'] != columns for checkpoint in self.checkpoints):
            raise ValueError('Saved GraphSAGE schemas differ')
        self.indices = [structure.features.FEATURE_COLUMNS.index(column) for column in columns]

    def load(self, name, call):
        runtime.sync()
        started = time.perf_counter()
        model = call()
        runtime.sync()
        self.loading.append({'model': name, 'seconds': time.perf_counter() - started})
        print(f'LOADED {name}', flush=True)
        return model

    def predict_sage(self, sample, feature_output):
        rows, _, tensor = feature_output
        metadata = pd.DataFrame(rows)
        virtual_root = self.output / 'in_memory_graph_interface'
        expected = virtual_root / 'graphs' / sample['sample_id'] / 'graph.pt'

        def load_from_memory(*args, **kwargs):
            if Path(args[0]).resolve() != expected.resolve():
                raise ValueError('Unexpected disk request in the in-memory graph interface')
            return tensor

        with runtime.replace_callable(torch, 'load', load_from_memory):
            graph, candidate_rows = sage.original.load_split(
                virtual_root, metadata, 'inference', self.indices, 'branch_group')
        frames = []
        for model, checkpoint in zip(self.models, self.checkpoints):
            normalized = dict(graph)
            normalized['x'] = (graph['x'] - checkpoint['mean']) / checkpoint['std']
            normalized = {key: value.to('cuda:0') for key, value in normalized.items()}
            frames.append(sage.original.predict(model, normalized, candidate_rows))
        prediction = pd.concat(frames, ignore_index=True)
        prediction = prediction.groupby(sage.GROUP_KEYS, as_index=False).score.mean()
        if not np.isfinite(prediction.score.to_numpy()).all():
            raise RuntimeError('Non-finite GraphSAGE score')
        return prediction[['sample_id', 'segment_id', 'score']].sort_values('score', ascending=False)

    def __call__(self, rgb, sample):
        inputs = {'rgb': rgb}
        inputs['roi'] = runtime.baseline.apply_roi_filter(self.roi, rgb)
        inputs['woody'] = runtime.baseline.run_branch_segmentation(self.woody, inputs['roi'][1], device='cuda:0')
        buds = runtime.baseline.run_bud_detection_pipeline(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), bud_model=self.bud, roi_model=self.roi,
            use_roi_filter=True, score_thr=float(self.defaults['bud_score_threshold']), device='cuda:0')
        inputs['bud'] = tuple(buds[key] for key in ('boxes', 'scores', 'labels', 'masks_info'))
        inputs['tape_support'] = runtime.routing.prepare_processed_router_mask(inputs['woody'], rgb)
        for stage in STRUCTURAL_STAGES:
            inputs[stage] = structure.make_call(stage, inputs, self.defaults, self.config, sample, self.hgb)()
        inputs['graphsage_ensemble'] = self.predict_sage(sample, inputs['structural_features'])
        return inputs


def score_comparison(reference, observed):
    order = reference.segment_id.tolist() == observed.segment_id.tolist()
    aligned = reference.merge(observed, on=['sample_id', 'segment_id'], suffixes=('_reference', '_observed'), validate='one_to_one')
    same_candidates = len(aligned) == len(reference) == len(observed)
    delta = float(np.max(np.abs(aligned.score_reference.to_numpy() - aligned.score_observed.to_numpy()))) if len(aligned) else None
    exact = runtime.fingerprint(structure.canonical(reference)) == runtime.fingerprint(structure.canonical(observed))
    return {'candidate_set_equal': same_candidates, 'candidate_order_equal': order,
            'exact_equal': exact, 'max_abs_score_delta': delta,
            'preset_score_gate_passed': bool(same_candidates and order and delta is not None and delta <= 1e-7)}


def audit_cached_outputs(output, sample_id, result):
    comparisons = {}
    for stage in (*runtime.STAGES, *STRUCTURAL_STAGES):
        reference = runtime.load_cache(output, sample_id, stage)
        expected = runtime.fingerprint(structure.canonical(reference))
        observed = runtime.fingerprint(structure.canonical(result[stage]))
        comparisons[stage] = {'reference_sha256': expected, 'observed_sha256': observed, 'exact_equal': expected == observed}
    reference = pd.read_csv(output / 'graphsage_variance_diagnostic/ranked_candidates' / f'{sample_id}_graphsage_ensemble.csv', float_precision='round_trip')
    comparisons['graphsage_ensemble'] = score_comparison(reference, result['graphsage_ensemble'])
    return comparisons


def output_signature(result):
    return {stage: runtime.fingerprint(structure.canonical(result[stage]))
            for stage in (*runtime.STAGES, *STRUCTURAL_STAGES)}


def main():
    parser = argparse.ArgumentParser(description='Continuous existing RGB-to-atomic-ranking pipeline; no region inference')
    parser.add_argument('--output', type=Path, default=runtime.ROOT / '04_results/pruning_decision/runtime_profiling_v1')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--run-name', default='atomic_e2e')
    args = parser.parse_args()
    if min(args.repeats, args.warmup) < 1:
        parser.error('repeats and warmup must be positive')
    output = args.output.resolve()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        parser.error('run-name must be a single directory name')
    destination = output / args.run_name
    if destination.exists():
        raise RuntimeError('E2E directory already exists; preserve it and explicitly choose a fresh run directory')
    runtime.baseline.ensure_dir(destination)
    frozen = runtime.freeze_samples(output, 10, 20260903)
    sources = {}
    for name in ('run_manifest.json', 'structural_run_manifest.json'):
        for relative, expected in runtime.read_json(output / name)['source_sha256'].items():
            observed = runtime.baseline.sha256_file(runtime.ROOT / relative)
            if observed != expected:
                raise RuntimeError(f'Frozen source changed: {relative}')
            sources[relative] = observed
    for path in (Path(__file__), Path(sage.__file__), Path(sage.original.__file__)):
        sources[str(path.relative_to(runtime.ROOT))] = runtime.baseline.sha256_file(path)
    weights = dict(runtime.read_json(output / 'run_manifest.json')['weights'])
    for value in weights.values():
        if runtime.baseline.sha256_file(runtime.ROOT / value['path']) != value['sha256']:
            raise RuntimeError(f"Checkpoint changed: {value['path']}")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    manifest = {
        'status': 'preregistered_before_measurement', 'scope': 'atomic only; regions excluded by author',
        'timing_boundary': 'single outer synchronized wall clock around a continuous call from already-loaded RGB to both HGB and five-seed GraphSAGE ranked candidates',
        'ranking_mode': 'shared perception/structure/features, then HGB and saved five-seed GraphSAGE sequentially; joint-output latency, not standalone GraphSAGE latency',
        'excluded': ['model loading', 'image loading', 'result persistence', 'hashing and equivalence audits', 'region features and ranking'],
        'preserved': ['second ROI invocation inside original bud pipeline', 'bud annotation rendering inside original bud entry', 'existing V3.1 quality audit', 'original functions, settings and weights'],
        'model_scope': 'historical dual-attention woody checkpoint and fixed-split atomic rankers; not paper channel-only WoodyCA-Net or nested-CV latency',
        'warmup_full_pipeline': args.warmup, 'repeats_per_image': args.repeats,
        'n_images': len(frozen['samples']), 'sample_manifest_sha256': runtime.baseline.sha256_file(output / 'sample_manifest.json'),
        'score_gate': {'atol': 1e-7, 'rtol': 0, 'order_required_exact': True,
                       'policy': 'retain strict flag; known GraphSAGE numerical variance may fail this flag without changing order; do not silently raise tolerance'},
        'reference_protocol': 'one complete uninstrumented continuous call per image in the same process, then timed repetitions; historical separate-process caches audited separately, not used as timer equivalence reference',
        'source_sha256': sources, 'weights': weights, 'gpu': torch.cuda.get_device_name(0),
        'torch': torch.__version__, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'torch_cpu_threads': torch.get_num_threads(), 'opencv_threads': cv2.getNumThreads(),
        'per_tree_status': 'not measured; one selected core view per tree is not a complete multi-view tree run',
    }
    runtime.save_json(destination / 'run_manifest.json', manifest)
    rows, audits = [], []
    try:
        pipeline = AtomicPipeline(frozen, destination)
        runtime.save_json(destination / 'model_loading_runtime.json', pipeline.loading)
        for path in [pipeline.hgb_path, *pipeline.paths]:
            manifest['weights'][path.stem] = {'path': str(path.relative_to(runtime.ROOT)), 'sha256': runtime.baseline.sha256_file(path)}
        runtime.save_json(destination / 'run_manifest.json', manifest)
        first = frozen['samples'][0]
        rgb = runtime.baseline.load_image_rgb(runtime.ROOT / first['image_path'])
        for index in range(args.warmup):
            warm = pipeline(rgb, first)
            runtime.sync()
            if index == args.warmup - 1:
                check = audit_cached_outputs(output, first['sample_id'], warm)
                runtime.save_json(destination / 'warmup_historical_cache_comparison.json', check)
            del warm
            print(f'WARMUP complete pipeline {index + 1}/{args.warmup}', flush=True)
        for sample in frozen['samples']:
            sample_id = sample['sample_id']
            rgb = runtime.baseline.load_image_rgb(runtime.ROOT / sample['image_path'])
            reference = pipeline(rgb, sample)
            runtime.sync()
            expected = output_signature(reference)
            reference_scores = reference['graphsage_ensemble'].copy()
            reference_dir = runtime.baseline.ensure_dir(destination / 'uninstrumented_reference')
            runtime.save_json(reference_dir / f'{sample_id}_signature.json', expected)
            runtime.save_json(reference_dir / f'{sample_id}_historical_cache_comparison.json', audit_cached_outputs(output, sample_id, reference))
            for stage in ('hgb_ranking', 'graphsage_ensemble'):
                reference[stage].to_csv(reference_dir / f'{sample_id}_{stage}.csv', index=False)
            del reference
            print(f'REFERENCE complete pipeline {sample_id}', flush=True)
            previous = None
            for repeat in range(args.repeats):
                torch.cuda.reset_peak_memory_stats()
                runtime.sync()
                start = time.perf_counter()
                result = pipeline(rgb, sample)
                runtime.sync()
                seconds = time.perf_counter() - start
                observed = output_signature(result)
                check = {stage: {'reference_sha256': expected[stage], 'observed_sha256': observed[stage],
                                 'exact_equal': expected[stage] == observed[stage]} for stage in expected}
                check['graphsage_ensemble'] = score_comparison(reference_scores, result['graphsage_ensemble'])
                if previous is not None:
                    check['graphsage_repeated_e2e'] = score_comparison(previous, result['graphsage_ensemble'])
                previous = result['graphsage_ensemble'].copy()
                audits.append({'sample_id': sample_id, 'repeat': repeat + 1, 'stages': check})
                runtime.save_json(destination / 'equivalence_audit.json', audits)
                if not check['structural_features']['exact_equal'] or not check['hgb_ranking']['exact_equal'] or not check['graphsage_ensemble']['candidate_order_equal']:
                    raise RuntimeError(f'Integrated output mismatch: {sample_id}; timer did not alter algorithm settings')
                rows.append({'sample_id': sample_id, 'tree_id': sample['tree_id'], 'repeat': repeat + 1,
                             'total_seconds': seconds, 'n_candidates': len(result['hgb_ranking']),
                             'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
                runtime.write_rows(destination / 'per_image_runtime.csv', rows)
                ranked_dir = runtime.baseline.ensure_dir(destination / 'ranked_candidates')
                for stage in ('hgb_ranking', 'graphsage_ensemble'):
                    result[stage].to_csv(ranked_dir / f'{sample_id}_{stage}_repeat{repeat + 1}.csv', index=False)
                del result
                print(f'TIMED E2E {sample_id} {repeat + 1}/{args.repeats} {seconds:.4f}s', flush=True)
        final_sources = {relative: runtime.baseline.sha256_file(runtime.ROOT / relative) == expected for relative, expected in sources.items()}
        final_weights = {name: runtime.baseline.sha256_file(runtime.ROOT / value['path']) == value['sha256'] for name, value in manifest['weights'].items()}
        report = runtime.summarize(rows)
        all_scores_pass = all(item['stages']['graphsage_ensemble']['preset_score_gate_passed'] for item in audits)
        report.update(status='passed' if all_scores_pass else 'measured_with_graphsage_numerical_equivalence_flag',
                      continuous_wall_clock=True, atomic_only=True, joint_hgb_graphsage_output=True,
                      structural_features_exact=True, hgb_output_exact=True, graphsage_candidate_order_exact=True,
                      graphsage_bitwise_exact=all(item['stages']['graphsage_ensemble']['exact_equal'] for item in audits),
                      graphsage_preset_score_gate_passed=all_scores_pass,
                      graphsage_max_abs_score_delta=max(item['stages']['graphsage_ensemble']['max_abs_score_delta'] for item in audits),
                      source_unchanged=final_sources, weights_unchanged=final_weights,
                      manuscript_updated=False, per_tree_multiview_measured=False)
        if not all(final_sources.values()) or not all(final_weights.values()):
            raise RuntimeError('Sources or weights changed during profiling')
        runtime.save_json(destination / 'summary.json', report)
        print(f"COMPLETE E2E {report['total_seconds']}", flush=True)
    except Exception as error:
        runtime.save_json(destination / 'failure.json', {'error': repr(error), 'completed_timed_runs': len(rows)})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
