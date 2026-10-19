import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/交互式安全推理_研究资料/论文研究_20260927/src')
sys.path.insert(0, '/root/交互式安全推理_研究资料/端到端范围检查_20261003/src')
from integrated_model import IntegratedPythia
from stateful_runner import StatefulRunner

path = Path('/root/pythia-160m')
print('loading model...', flush=True)
t0 = time.perf_counter()
model = IntegratedPythia(path, 16 + 32 + 4 + 2)
print('model_loaded_s', round(time.perf_counter()-t0, 2), 'n_params', len(model.params), flush=True)

runner = StatefulRunner(profile=False, field='FM128', fraction=24)
weights = [f'w{i}' for i in range(len(model.params))]
print('binding weights...', flush=True)
t0 = time.perf_counter()
share_s = sum(runner.bind(n, p) for n, p in zip(weights, model.params))
print('bind_s', round(share_s, 2), flush=True)

print('compiling native prefill...', flush=True)
t0 = time.perf_counter()
exe, compile_s = model.compile_integrated('native', 16, True)
print('compile_prefill_s', round(compile_s, 2), flush=True)

prefix = np.array([[29618, 247, 13814, 2086, 326, 9563, 512, 253, 2505, 4367, 762, 247, 9617, 285, 6548, 1755]], np.int64)
runner.bind('cache', model.empty_cache())
runner.bind('ids', prefix)
runner.bind('draft', np.zeros((1, 0), np.int64))
runner.bind('pos', np.array(0, np.int64), public=True)
inputs = ['ids', 'draft', 'pos', 'cache'] + weights
print('running prefill...', flush=True)
t0 = time.perf_counter()
actual, timing = runner.run(exe, inputs, {3: 'cache'}, [0, 1, 2])
print('run_prefill_s', round(time.perf_counter()-t0, 2), 'timing', timing, flush=True)
print('valid', bool(actual[0][0]), 'accepted', int(actual[1][0]), 'token', int(actual[2][0]), flush=True)
runner.close()
print('SMOKE_DONE', flush=True)
