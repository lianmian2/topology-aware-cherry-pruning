from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

import run_e2e_regression as baseline
import logic_bud
import logic_roi
from mask_topology_routing import utils as routing


ROOT = baseline.PROJECT_ROOT
SOURCE = ROOT / '01_data/03_processed/pruning_gnn_v1/manifests/current_pipeline_manifest.json'
DATASET = ROOT / '04_results/pruning_decision/pruning_segment_cleaning_v31_stable_20260731/exports/gnn_dataset_20260809'
STAGES = ('roi', 'woody', 'tape_support', 'bud')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save_json(path, value):
    baseline.ensure_dir(path.parent)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def fingerprint(value):
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, (list, tuple)):
            digest.update(b'[')
            for child in item:
                visit(child)
            digest.update(b']')
        elif isinstance(item, dict):
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, np.generic):
            visit(item.item())
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def sync():
    torch.cuda.synchronize(0)


@contextmanager
def replace_callable(owner, name, replacement):
    own_attribute = name in vars(owner)
    previous = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield previous
    finally:
        if own_attribute:
            setattr(owner, name, previous)
        else:
            delattr(owner, name)


class ForwardTimer:
    def __init__(self, model):
        self.model = model
        self.events = []

    @contextmanager
    def enabled(self):
        original = self.model.forward

        def timed(*args, **kwargs):
            sync()
            start = time.perf_counter()
            result = original(*args, **kwargs)
            sync()
            self.events.append((start, time.perf_counter()))
            return result

        with replace_callable(self.model, 'forward', timed):
            yield

    def measure(self, call):
        self.events = []
        with self.enabled():
            sync()
            start = time.perf_counter()
            result = call()
            sync()
            end = time.perf_counter()
        forward = sum(stop - begin for begin, stop in self.events)
        first = self.events[0][0] if self.events else end
        last = self.events[-1][1] if self.events else end
        return result, {
            'total_seconds': end - start,
            'model_forward_seconds': forward,
            'before_first_forward_seconds': first - start,
            'between_forwards_seconds': max(0.0, last - first - forward),
            'after_last_forward_seconds': end - last,
            'forward_calls': len(self.events),
        }


def summarize(rows):
    by_image = {}
    for row in rows:
        by_image.setdefault(row['sample_id'], []).append(row)
    fields = [key for key in rows[0] if key.endswith('_seconds')] if rows else []
    result = {'n_images': len(by_image), 'n_trees': len({key.split('_before_')[0] for key in by_image}),
              'n_timed_runs': len(rows), 'aggregation': 'equal-weight image means; SD across image means'}
    for field in fields:
        values = [statistics.mean(float(row[field]) for row in group) for group in by_image.values()]
        result[field] = {'mean': statistics.mean(values),
                         'sd': statistics.stdev(values) if len(values) > 1 else None,
                         'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    return result


def write_rows(path, rows):
    baseline.ensure_dir(path.parent)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def freeze_samples(output, count, seed):
    destination = output / 'sample_manifest.json'
    if destination.exists():
        frozen = read_json(destination)
        if frozen['requested_images'] != count or frozen['seed'] != seed:
            raise ValueError('Existing frozen sample selection differs; use a new output directory.')
        for sample in frozen['samples']:
            if baseline.sha256_file(ROOT / sample['image_path']) != sample['image_sha256']:
                raise ValueError(f"Input changed: {sample['sample_id']}")
        return frozen
    test_trees = read_json(DATASET / 'splits/test_trees.json')['tree_ids']
    source = read_json(SOURCE)
    candidates = {}
    for sample in source['samples']:
        tree = sample['sample_id'].split('_before_')[0]
        if tree in test_trees and (DATASET / 'graphs' / sample['sample_id']).is_dir():
            candidates.setdefault(tree, []).append(sample)
    rng = random.Random(seed)
    trees = rng.sample(sorted(candidates), count)
    samples = []
    for tree in trees:
        sample = dict(rng.choice(sorted(candidates[tree], key=lambda row: row['sample_id'])))
        image = baseline.load_image_rgb(ROOT / sample['image_path'])
        sample.update(image_sha256=baseline.sha256_file(ROOT / sample['image_path']),
                      image_shape=list(image.shape), tree_id=tree)
        samples.append(sample)
    frozen = {'requested_images': count, 'seed': seed,
              'selection': 'seeded random trees from existing fixed test split, one random accepted core view per tree; no score or runtime selection',
              'source_manifest': str(SOURCE.relative_to(ROOT)),
              'source_sha256': baseline.sha256_file(SOURCE),
              'split_sha256': baseline.sha256_file(DATASET / 'splits/test_trees.json'),
              'defaults': source['defaults'], 'samples': samples}
    save_json(destination, frozen)
    return frozen


def cache_path(output, sample_id, stage):
    return output / 'component_outputs' / sample_id / f'{stage}.pkl'


def load_cache(output, sample_id, stage):
    with cache_path(output, sample_id, stage).open('rb') as handle:
        return pickle.load(handle)


def stage_call(stage, model, rgb, output, sample_id):
    if stage == 'roi':
        return lambda: baseline.apply_roi_filter(model, rgb)
    if stage == 'woody':
        _, filtered = load_cache(output, sample_id, 'roi')
        return lambda: baseline.run_branch_segmentation(model, filtered, device='cuda:0')
    if stage == 'tape_support':
        mask = load_cache(output, sample_id, 'woody')
        return lambda: routing.prepare_processed_router_mask(mask, rgb)
    _, filtered = load_cache(output, sample_id, 'roi')
    bgr = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
    return lambda: logic_bud.run_bud_global_detection(model, bgr)


def main():
    parser = argparse.ArgumentParser(description='Unmodified-entry component profiling; not end-to-end latency')
    parser.add_argument('--output', type=Path, default=ROOT / '04_results/pruning_decision/runtime_profiling_v1')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--images', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260903)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    if min(args.images, args.repeats, args.warmup) < 1:
        parser.error('images, repeats and warmup must be positive')
    output = baseline.ensure_dir(args.output.resolve())
    frozen = freeze_samples(output, args.images, args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Existing tape inference requires CUDA; CPU substitution is prohibited.')
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    manager = baseline.ModelManager()
    manager.device = 'cuda:0'
    source_paths = [Path(__file__), Path(baseline.__file__), Path(logic_bud.__file__),
                    Path(logic_roi.__file__), Path(routing.__file__),
                    ROOT / '07_graphical_interface/unified_system/logic_branch.py',
                    ROOT / '07_graphical_interface/unified_system/logic_models.py']
    provenance = {
        'purpose': 'component runtime feasibility; no end-to-end claim',
        'model_scope': 'unchanged deployed historical dual-attention branch checkpoint; not channel-only WoodyCA-Net runtime',
        'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'torch_cpu_threads': torch.get_num_threads(), 'opencv_threads': cv2.getNumThreads(),
        'bud_patch_batch': logic_bud.BATCH_SIZE,
        'git_commit': baseline.git_commit(),
        'source_sha256': {str(path.relative_to(ROOT)): baseline.sha256_file(path) for path in source_paths},
        'weights': {}, 'warmup_per_component': args.warmup, 'repeats_per_image': args.repeats,
        'timing_boundary': 'already-loaded stage input to returned stage output; excludes model loading, cache/image IO and fingerprinting',
        'breakdown_boundary': 'forward includes detector-native decoding/postprocessing for ROI/buds; before-first includes preprocessing/H2D; between-forwards mixes patch preprocessing, D2H and CPU filtering; after-last includes D2H/output postprocessing. These are measured boundaries, not pure neural-kernel latency.',
        'bud_component_boundary': 'ROI-conditioned BGR input; excludes ROI and drawing. Existing full pipeline repeats ROI inside bud entry; future E2E must retain and time it.',
        'end_to_end_status': 'not_implemented',
    }
    for key in ('roi', 'branch', 'bud'):
        relative = frozen['defaults'][f'{key}_weight']
        actual = baseline.sha256_file(ROOT / relative)
        if actual != frozen['defaults'][f'{key}_sha256']:
            raise ValueError(f'Frozen checkpoint mismatch: {relative}')
        provenance['weights'][key] = {'path': relative, 'sha256': actual}
    tape_weight = ROOT / '03_models/Tape_Segmentation_V2_outputs/best_model.pth'
    provenance['weights']['tape'] = {'path': str(tape_weight.relative_to(ROOT)), 'sha256': baseline.sha256_file(tape_weight)}
    save_json(output / 'run_manifest.json', provenance)
    loaders = {'roi': manager.load_roi_model, 'woody': manager.load_branch_model,
               'tape_support': lambda: routing._get_tape_model('cuda:0'), 'bud': manager.load_bud_global_model}
    load_path = output / 'model_loading_runtime.json'
    loading = read_json(load_path) if load_path.exists() else {}
    for stage in args.stages:
        stage_dir = baseline.ensure_dir(output / stage)
        if (stage_dir / 'summary.json').exists():
            print(f'SKIP completed {stage}', flush=True)
            continue
        print(f'LOAD {stage}', flush=True)
        sync()
        started = time.perf_counter()
        model = loaders[stage]()
        sync()
        loading[stage] = time.perf_counter() - started
        save_json(load_path, loading)
        timer = ForwardTimer(model)
        first = frozen['samples'][0]
        rgb = baseline.load_image_rgb(ROOT / first['image_path'])
        call = stage_call(stage, model, rgb, output, first['sample_id'])
        for index in range(args.warmup):
            call()
            sync()
            print(f'WARMUP {stage} {index + 1}/{args.warmup}', flush=True)
        rows, audits = [], []
        try:
            for sample in frozen['samples']:
                sample_id = sample['sample_id']
                rgb = baseline.load_image_rgb(ROOT / sample['image_path'])
                call = stage_call(stage, model, rgb, output, sample_id)
                reference = call()
                sync()
                expected = fingerprint(reference)
                for repetition in range(args.repeats):
                    result, measured = timer.measure(call)
                    observed = fingerprint(result)
                    audit = {'sample_id': sample_id, 'repeat': repetition + 1,
                             'reference_sha256': expected, 'profiled_sha256': observed,
                             'exact_equal': observed == expected}
                    audits.append(audit)
                    save_json(stage_dir / 'equivalence_audit.json', audits)
                    if observed != expected:
                        raise RuntimeError(f'Output equality gate failed: {stage}/{sample_id}; no algorithm or determinism settings changed to force agreement')
                    row = {'sample_id': sample_id, 'tree_id': sample['tree_id'],
                           'repeat': repetition + 1, **measured}
                    rows.append(row)
                    write_rows(stage_dir / 'per_image_runtime.csv', rows)
                    print(f"TIMED {stage} {sample_id} {repetition + 1}/{args.repeats} {measured['total_seconds']:.4f}s", flush=True)
                destination = cache_path(output, sample_id, stage)
                baseline.ensure_dir(destination.parent)
                with destination.open('wb') as handle:
                    pickle.dump(reference, handle, protocol=pickle.HIGHEST_PROTOCOL)
            report = summarize(rows)
            report.update(status='passed', exact_equivalence_all=True)
            save_json(stage_dir / 'summary.json', report)
            print(f"COMPLETE {stage} mean={report['total_seconds']['mean']:.4f}s", flush=True)
        except Exception as error:
            save_json(stage_dir / 'failure.json', {'error': repr(error), 'completed_timed_runs': len(rows)})
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
