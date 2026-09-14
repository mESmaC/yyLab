"""Standard-library-only hook/receipt checks: no Torch import or GPU execution."""
from __future__ import annotations
import ast
import builtins
import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AllocatorHooks(unittest.TestCase):
    def simulate(self, composed=False, first=False, final=False, fail_at=None):
        module = load('benchmark_allocator_bf16' if composed else 'benchmark_allocator')
        events, written = [], []
        device = SimpleNamespace(type='cuda', index=None)
        model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device=device)]))
        class Optimizer:
            def candidate_metadata(self):
                return {'name':'bf16_moments_large_slices_v1', 'expected_state_bytes':42}
        sprint = SimpleNamespace(SprintCPUOffloadAdamW=Optimizer)
        class Engine:
            def __init__(self):
                self.device = device
                self.optimizer = sprint.SprintCPUOffloadAdamW()
            def train_step(self, batches, *, step=0):
                events.append(('step', step))
                if step == fail_at:
                    raise ValueError('injected update failure')
                return object()
        sprint.SprintStepEngine = Engine
        harness = SimpleNamespace(PROVENANCE_SOURCE_PATHS=('production.py',))
        harness._decoder_samples = lambda model: events.append(('sample',None))
        harness._training_memory = lambda model, optimizer: {'optimizer_state':{'bytes':42}}
        harness.write_json = lambda path, value: written.append(value)
        def case(config, cfg, precision, batch, mode, device, inputs):
            engine = Engine()
            for step in range(cfg['warmup_steps']):
                engine.train_step([],step=step)
            harness._decoder_samples(model)
            events.append(('timer_start',None))
            for step in range(cfg['warmup_steps'],cfg['warmup_steps']+cfg['measure_steps']):
                engine.train_step([],step=step)
            events.append(('timer_end',None))
            harness._decoder_samples(model)
            memory = harness._training_memory(model, engine.optimizer)
            return {'status':'ok', 'runtime':{}, 'memory':memory}
        harness._measure_precision_case = case
        def benchmark(cfg, output):
            row = harness._measure_precision_case(None,cfg,'nvfp4_mxfp8_fast',16,'default',device,None)
            report = {'schema':'yylab-sprint-precision-benchmark-v1', 'status':row['status'],
                      'rows':[row], 'source_sha256':dict.fromkeys(harness.PROVENANCE_SOURCE_PATHS,'source')}
            harness.write_json(output / 'benchmark.json', report)
            return report
        harness.run_sprint_precision_benchmark = benchmark
        def fraction(index):
            assert type(index) is int, 'fraction query requires explicit integer device'
            return 1.0
        torch = SimpleNamespace(cuda=SimpleNamespace(
            memory_allocated=lambda device: 100,
            memory_reserved=lambda device: 200,
            synchronize=lambda device: events.append(('sync',None)),
            empty_cache=lambda: events.append(('release',None)),
            current_device=lambda: 0,
            get_per_process_memory_fraction=fraction,
            memory=SimpleNamespace(get_allocator_backend=lambda:'native')))
        # Execute the actual candidate context without importing its Torch module.
        candidate_source = ast.parse((BASE/'bf16_moments_candidate.py').read_text())
        function = next(node for node in candidate_source.body if isinstance(node,ast.FunctionDef) and node.name=='_candidate_runner')
        namespace={'contextmanager':contextmanager,'wraps':wraps,'NAME':'bf16_moments_large_slices_v1',
                   'NOTICE':'changed optimizer numerics','BF16MomentsAdamW':Optimizer}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'candidate_context_ast','exec'),namespace)
        candidate=SimpleNamespace(__file__=str(BASE/'bf16_moments_candidate.py'),NOTICE=namespace['NOTICE'],
                                  _candidate_runner=namespace['_candidate_runner'])
        original_import = builtins.__import__
        def importing(name, *args, **kwargs):
            if name=='torch': return torch
            if name=='yylab.train': return SimpleNamespace(benchmark_sprint_precision=harness,sprint=sprint)
            if name=='bf16_moments_candidate': return candidate
            return original_import(name,*args,**kwargs)
        policy={'config':'backend:native','release_after_warmup':final,'release_after_first_warmup_step':first}
        cfg={'experimental_allocator':policy,'warmup_steps':3,'measure_steps':2,'effective_batch':32}
        if composed:
            cfg.update(experimental_optimizer_candidate='bf16_moments_large_slices_v1',cpu_optimizer_kernel='compiled')
        originals = harness._measure_precision_case,harness._decoder_samples,harness.write_json,Engine.train_step,harness._training_memory
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{},clear=True), patch('builtins.__import__',side_effect=importing):
            report=module.run(cfg,Path(directory)/'new-output')
        self.assertEqual(originals,(harness._measure_precision_case,harness._decoder_samples,harness.write_json,Engine.train_step,harness._training_memory))
        self.assertEqual(harness.PROVENANCE_SOURCE_PATHS,('production.py',))
        row=report['rows'][0]; metadata=row['runtime']['experimental_allocator']
        self.assertEqual(row['allocator_experiment'],metadata)
        self.assertEqual(metadata['actual_per_process_memory_fraction'],1.0)
        if fail_at is not None:
            self.assertEqual(row['status'],'error')
            self.assertIn('injected update failure',row['error'])
            self.assertEqual(metadata['first_warmup_release_count'],int(first and fail_at>0))
            return
        self.assertEqual(row['status'],'ok')
        self.assertEqual(metadata['train_step_calls'],5)
        self.assertEqual(metadata['first_warmup_release_count'],int(first))
        self.assertEqual(metadata['release_count'],int(final))
        self.assertEqual(sum(event[0]=='release' for event in events),int(first or final))
        if first:
            self.assertLess(events.index(('step',0)),events.index(('release',None)))
            self.assertLess(events.index(('release',None)),events.index(('step',1)))
        if final:
            self.assertLess(events.index(('step',2)),events.index(('release',None)))
            self.assertLess(events.index(('release',None)),events.index(('timer_start',None)))
        if composed:
            self.assertEqual(row['optimizer_candidate']['name'],'bf16_moments_large_slices_v1')
            self.assertEqual(written[0]['optimizer_candidate']['name'],'bf16_moments_large_slices_v1')
            self.assertTrue(any(path.endswith('benchmark_allocator_bf16.py') for path in written[0]['source_sha256']))
            self.assertTrue(any(path.endswith('bf16_moments_candidate.py') for path in written[0]['source_sha256']))
            self.assertFalse(metadata['training_math_unchanged'])
            self.assertTrue(metadata['allocator_training_math_unchanged'])

    def test_base_first(self): self.simulate(first=True)
    def test_base_final(self): self.simulate(final=True)
    def test_base_control(self): self.simulate()
    def test_composed_first(self): self.simulate(composed=True,first=True)
    def test_composed_final(self): self.simulate(composed=True,final=True)
    def test_cleanup_after_failure(self): self.simulate(composed=True,first=True,fail_at=1)
    def test_failed_first_step_never_releases(self): self.simulate(composed=True,first=True,fail_at=0)


if __name__=='__main__':
    unittest.main(verbosity=2)
