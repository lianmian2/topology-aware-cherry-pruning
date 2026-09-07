from __future__ import annotations

import argparse
import copy
import pickle
import time
from dataclasses import asdict, fields, is_dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
import joblib

import profile_pruning_components as runtime
import run_e2e_regression as original
from mask_topology_routing import utils as routing
from mask_topology_routing_binary_v3 import (
    audit_binary_topology_v3, load_binary_v3_config, refine_clipped_groups_v3,
)
from mask_topology_routing_binary_v3.thickness_routing import make_router_selector
import build_atomic_segment_graphs as atomic
from export_cleaned_pruning_gnn_dataset import effective_candidate_type
import prepare_pruning_decision_features as features
from build_region_level_dataset import build_region_frame


STAGES = ('routing_geometry', 'bud_directions', 'bud_flow_rerank', 'clip_refine',
          'attachment', 'directed_graph', 'group_features', 'atomic_structure',
          'structural_features', 'region_features', 'hgb_ranking')


def canonical(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, pd.DataFrame):
        return canonical(value.to_dict(orient='records'))
    if isinstance(value, nx.Graph):
        return {'directed': value.is_directed(), 'graph': canonical(value.graph),
                'nodes': canonical(sorted(value.nodes(data=True), key=lambda row: repr(row[0]))),
                'edges': canonical(sorted(value.edges(data=True), key=lambda row: repr(row[:2])))}
    if is_dataclass(value):
        return {field.name: canonical(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, set):
        return sorted((canonical(item) for item in value), key=repr)
    if isinstance(value, (str, int, float, bool, np.ndarray, np.generic)) or value is None:
        return value
    raise TypeError(f'Unverified fingerprint type: {type(value)}')


def attachment_rows(inputs):
    rows = []
    for item in inputs['attachment']:
        row = asdict(item)
        row['bud_label'] = int(inputs['bud'][2][item.bud_index])
        row['bud_score'] = float(inputs['bud'][1][item.bud_index])
        rows.append(row)
    return rows


def make_call(stage, inputs, defaults, config, sample, ranker=None):
    def call():
        if stage == 'routing_geometry':
            processed, tape = inputs['tape_support']
            initial_config = {**original.ROUTER_CONFIG, 'enable_junction_pairing': False,
                              'enable_bud_density_prior': False, 'enable_bud_direction_flow': False,
                              'enable_bud_root_split': False}
            return original.build_prediction_result(
                combined_mask=inputs['woody'], image_rgb=inputs['rgb'],
                processed_mask=processed, protected_tape_mask=tape, **initial_config)
        if stage == 'bud_directions':
            _, scores, _, masks = inputs['bud']
            return (original.extract_bud_orientations(masks),
                    original.extract_directed_bud_orientations(
                        masks, inputs['routing_geometry'].skeleton_map, scores,
                        min_confidence=original.BUD_DIRECTION_MIN_CONFIDENCE))
        if stage == 'bud_flow_rerank':
            orientations, directions = inputs['bud_directions']
            return original.rerank_variant('full_flow', inputs['routing_geometry'],
                                           inputs['tape_support'][1], {'boxes': inputs['bud'][0]},
                                           directions, orientations)
        if stage == 'clip_refine':
            prediction = inputs['bud_flow_rerank']
            clip = defaults['mask_clip']
            clipped, clip_stats = original.clip_annotation_groups(
                prediction.annotation_groups, prediction.mask,
                max_exit_distance=float(clip['max_exit_distance']), max_gap=float(clip['max_gap']),
                snap_radius=int(clip['snap_radius']))
            groups, refinement = refine_clipped_groups_v3(clipped, inputs['tape_support'][0], config)
            audit = audit_binary_topology_v3(groups, inputs['tape_support'][0], config,
                                            baseline_groups=clipped, include_global_checks=True)
            return groups, clip_stats, refinement, audit
        if stage == 'attachment':
            prediction = inputs['bud_flow_rerank']
            return original.attach_buds_to_skeleton(
                inputs['bud'][0], inputs['bud'][3], prediction.skeleton_map, prediction.mask,
                annotation_groups=inputs['clip_refine'][0])
        if stage == 'directed_graph':
            return original._build_directed_topology(inputs['clip_refine'][0])
        if stage == 'group_features':
            return original.extract_features(sample['sample_id'], inputs['clip_refine'][0],
                                             inputs['attachment'], inputs['bud'][2], inputs['rgb'].shape[:2])
        if stage == 'atomic_structure':
            graph, group_sets = atomic.build_graph(inputs['clip_refine'][0])
            roots = atomic.choose_roots(graph, group_sets)
            topological = atomic.trace_topological_segments(graph, roots)
            segments, types, bud_ids, unattached = atomic.build_atomic_segments(
                graph, roots, topological, attachment_rows(inputs))
            audit = atomic.validate_atomic_partition(topological, segments)
            if not audit['valid']:
                raise RuntimeError(f'Atomic partition invalid: {audit}')
            for segment in segments:
                candidate = effective_candidate_type(segment)
                segment['candidate_type'] = candidate
                segment['is_candidate'] = int(candidate is not None)
                segment['label_mask'] = segment['is_candidate']
            return {'sample_id': sample['sample_id'], 'segment_nodes': segments,
                    'geometry_audit': audit, 'landmark_types': types,
                    'landmark_bud_ids': bud_ids, 'unattached_buds': unattached}
        if stage == 'structural_features':
            branch_features = {
                str(row['branch_id']): {key: float(value) for key, value in row.items()
                                       if key not in {'sample_id', 'branch_id'} and value is not None}
                for row in inputs['group_features']}
            directions = [asdict(item) for item in inputs['bud_directions'][1]]
            return features.process_graph_payload(inputs['atomic_structure'], inputs['rgb'].shape[:2],
                                                  attachment_rows(inputs), directions, branch_features,
                                                  inputs['roi'][0], 'inference')
        if stage == 'region_features':
            frame = pd.DataFrame(inputs['structural_features'][0])
            frame = frame.loc[frame.label_mask == 1].reset_index(drop=True)
            return build_region_frame(frame, features.FEATURE_COLUMNS)
        if stage == 'hgb_ranking':
            frame = pd.DataFrame(inputs['structural_features'][0])
            candidates = frame.loc[frame.label_mask == 1].copy()
            candidates['score'] = ranker['model'].predict_proba(candidates[ranker['feature_columns']])[:, 1]
            return candidates[['sample_id', 'segment_id', 'score']].sort_values('score', ascending=False)
        raise ValueError(stage)

    return call


def main():
    parser = argparse.ArgumentParser(description='Existing V3.1 CPU components, unchanged functions and parameters')
    parser.add_argument('--output', type=runtime.Path, default=runtime.ROOT / '04_results/pruning_decision/runtime_profiling_v1')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    output = args.output.resolve()
    frozen = runtime.read_json(output / 'sample_manifest.json')
    for stage in runtime.STAGES:
        report = runtime.read_json(output / stage / 'summary.json')
        if report['status'] != 'passed' or report['n_images'] != len(frozen['samples']):
            raise RuntimeError(f'Perception gate not complete: {stage}')
    if not runtime.read_json(output / 'feature_interface_equivalence.json')['all_equal']:
        raise RuntimeError('Feature interface equality gate not passed')
    config = load_binary_v3_config()
    rulebook = runtime.ROOT / '04_results/mask_topology_routing/gt_mask_thickness_rulebook_20260731/gt_mask_thickness_rulebook.json'
    routing._choose_trunk_path_on_skeleton = make_router_selector(rulebook, config.rulebook_path)
    runtime.save_json(output / 'structural_run_manifest.json', {
        'pipeline': 'unchanged V3.1 stable thickness + bud flow + v3 refinement; no hierarchy partition',
        'rulebook_sha256': original.sha256_file(rulebook),
        'binary_config': {**asdict(config), 'rulebook_path': str(config.rulebook_path)},
        'router_config': original.ROUTER_CONFIG, 'repeats': args.repeats, 'warmup': args.warmup,
        'source_sha256': {str(path.relative_to(runtime.ROOT)): original.sha256_file(path)
                          for path in [runtime.Path(__file__), runtime.Path(features.__file__),
                                       runtime.Path(atomic.__file__), runtime.Path(original.__file__),
                                       *sorted((runtime.ROOT / '02_code/02_models/mask_topology_routing_binary_v3').glob('*.py')),
                                       *sorted((runtime.ROOT / '02_code/02_models/bud_skeleton_fusion').glob('*.py'))]},
        'end_to_end': False, 'input_copy_and_cache_io_excluded': True,
        'inference_candidate_scope': 'existing effective_candidate_type; no manual context promotion, no cut truth, is_cut_segment remains zero and is not a predictor',
        'ranking_scope': 'saved fixed-split atomic HGB only; no region/CV model checkpoint invented',
    })
    needed = {
        'routing_geometry': ('woody', 'tape_support'),
        'bud_directions': ('bud', 'routing_geometry'),
        'bud_flow_rerank': ('bud', 'routing_geometry', 'bud_directions', 'tape_support'),
        'clip_refine': ('bud_flow_rerank', 'tape_support'),
        'attachment': ('bud', 'bud_flow_rerank', 'clip_refine'),
        'directed_graph': ('clip_refine',),
        'group_features': ('clip_refine', 'attachment', 'bud'),
        'atomic_structure': ('clip_refine', 'attachment', 'bud'),
        'structural_features': ('atomic_structure', 'attachment', 'bud', 'bud_directions', 'group_features', 'roi'),
        'region_features': ('structural_features',),
        'hgb_ranking': ('structural_features',),
    }
    for stage in args.stages:
        destination = original.ensure_dir(output / stage)
        if (destination / 'summary.json').exists():
            print(f'SKIP completed {stage}', flush=True)
            continue
        rows, audits = [], []
        ranker = None
        if stage == 'hgb_ranking':
            model_path = runtime.ROOT / '04_results/pruning_decision/pruning_decision_closeout_20260809/baselines_v1/models/hgb_full.joblib'
            start = time.perf_counter()
            ranker = joblib.load(model_path)
            load_seconds = time.perf_counter() - start
            runtime.save_json(destination / 'model_loading.json', {
                'seconds': load_seconds, 'path': str(model_path.relative_to(runtime.ROOT)),
                'sha256': original.sha256_file(model_path), 'feature_columns': ranker['feature_columns']})
        try:
            for sample_index, sample in enumerate(frozen['samples']):
                inputs = {name: runtime.load_cache(output, sample['sample_id'], name) for name in needed[stage]}
                inputs['rgb'] = original.load_image_rgb(runtime.ROOT / sample['image_path'])

                def fresh_call():
                    return make_call(stage, copy.deepcopy(inputs), frozen['defaults'], config, sample, ranker)

                if sample_index == 0:
                    for index in range(args.warmup):
                        fresh_call()()
                        print(f'WARMUP {stage} {index + 1}/{args.warmup}', flush=True)
                reference = fresh_call()()
                expected = runtime.fingerprint(canonical(reference))
                for repetition in range(args.repeats):
                    call = fresh_call()
                    runtime.sync()
                    start = time.perf_counter()
                    result = call()
                    runtime.sync()
                    seconds = time.perf_counter() - start
                    observed = runtime.fingerprint(canonical(result))
                    audits.append({'sample_id': sample['sample_id'], 'repeat': repetition + 1,
                                   'reference_sha256': expected, 'profiled_sha256': observed,
                                   'exact_equal': observed == expected})
                    runtime.save_json(destination / 'equivalence_audit.json', audits)
                    if observed != expected:
                        raise RuntimeError(f"Equality gate failed: {stage}/{sample['sample_id']}")
                    rows.append({'sample_id': sample['sample_id'], 'tree_id': sample['tree_id'],
                                 'repeat': repetition + 1, 'total_seconds': seconds,
                                 'cpu_postprocessing_seconds': seconds})
                    runtime.write_rows(destination / 'per_image_runtime.csv', rows)
                    print(f"TIMED {stage} {sample['sample_id']} {repetition + 1}/{args.repeats} {seconds:.4f}s", flush=True)
                path = runtime.cache_path(output, sample['sample_id'], stage)
                original.ensure_dir(path.parent)
                with path.open('wb') as handle:
                    pickle.dump(reference, handle, protocol=pickle.HIGHEST_PROTOCOL)
            summary = runtime.summarize(rows)
            summary.update(status='passed', exact_equivalence_all=True)
            runtime.save_json(destination / 'summary.json', summary)
            print(f"COMPLETE {stage} mean={summary['total_seconds']['mean']:.4f}s", flush=True)
        except Exception as error:
            runtime.save_json(destination / 'failure.json', {'error': repr(error), 'completed_timed_runs': len(rows)})
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
