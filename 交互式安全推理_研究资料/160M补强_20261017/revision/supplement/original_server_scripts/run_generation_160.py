"""Four-condition, paired full-generation experiment with integrated guards."""
import argparse
import hashlib
import json
import platform
import random
import time
import traceback
from pathlib import Path
from integrated_model import ROOT, BASE, IntegratedPythia
from common import PREVIOUS
from stateful_runner import StatefulRunner
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=int, choices=[14, 31, 160], required=True)
    ap.add_argument('--indices', default='4,10,16,22,28,34,40,46')
    ap.add_argument('--tokens', type=int, default=32)
    ap.add_argument('--gamma', type=int, default=4)
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = ROOT / 'results' / args.tag
    out.mkdir(parents=True, exist_ok=False)
    def write(name, obj):
        temp = out / (name + '.tmp')
        temp.write_text(json.dumps(obj, indent=2))
        temp.replace(out / name)
    def announce(event, **kwargs):
        print(json.dumps(dict(event=event, time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kwargs)), flush=True)
    source_files = [ROOT/'src/integrated_model.py', ROOT/'src/run_generation.py',
                    BASE/'src/cached_model.py', BASE/'src/stateful_runner.py', BASE/'src/lowering.py']
    write('config.json', dict(**vars(args), created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          protocol='ABY3', field='FM128', fraction=24, profile=False, order_seed=1003,
          bootstrap_seed=1004, platform=platform.platform(), validity_output='one bit per call',
          source_sha256={str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
          timing='request wall time includes reset, sharing, draft, MPC and reconstruction; excludes initial setup and warmup'))
    for p in source_files:
        (out / (p.stem + '_measured.py')).write_bytes(p.read_bytes())
    path = PREVIOUS/'model' if args.model == 14 else (BASE/'model31' if args.model == 31 else Path('/root/pythia-160m'))
    model = IntegratedPythia(path, 16 + args.tokens + args.gamma + 2)
    draft_model = AutoModelForCausalLM.from_pretrained(PREVIOUS/'draft', local_files_only=True, torch_dtype=torch.float32).eval()
    target_model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float32).eval()
    metadata = json.loads((PREVIOUS/'data/trace_metadata.json').read_text())
    indices = [int(x) for x in args.indices.split(',')]
    methods = ['native', 'packed', 'guarded_native', 'guarded_packed']
    runners, compiled, setup = {}, {}, {}
    weights = [f'w{i}' for i in range(len(model.params))]
    rows, warmups = [], []
    try:
        for base_method in ['native', 'packed']:
            runners[base_method] = StatefulRunner(profile=False, field='FM128', fraction=24)
            setup[base_method+'_weight_share_s'] = sum(runners[base_method].bind(n, p) for n, p in zip(weights, model.params))
        for method in methods:
            for phase, length in [('prefill', 16), ('decode', args.gamma+1)]:
                announce('compile', method=method, phase=phase)
                exe, elapsed = model.compile_integrated(method, length, phase == 'prefill')
                compiled[method, phase] = exe
                (out/f'{method}_{phase}.mlir').write_bytes(exe.code)
                setup[f'{method}_{phase}_compile_s'] = elapsed
        write('setup.json', setup)

        def request(method, index, warmup=False):
            runner = runners[method.removeprefix('guarded_')]
            prefix = metadata[index]['prefix_ids'][-16:]
            assert len(prefix) == 16
            start = time.perf_counter()
            share = runner.bind('cache', model.empty_cache())
            share += runner.bind('ids', np.array([prefix], np.int64))
            share += runner.bind('draft', np.zeros((1, 0), np.int64))
            share += runner.bind('pos', np.array(0, np.int64), public=True)
            inputs = ['ids', 'draft', 'pos', 'cache'] + weights
            actual, timing = runner.run(compiled[method, 'prefill'], inputs, {3:'cache'}, [0, 1, 2])
            valid = bool(actual[0][0])
            generated = [int(actual[2][0])] if valid else []
            stamps = [time.perf_counter()-start] if valid else []
            blocks = [dict(phase='prefill', position=0, valid=valid, accepted=int(actual[1][0]),
                           token=int(actual[2][0]), outputs=generated.copy(), **timing)]
            draft_s = 0.0
            while valid and len(generated) < args.tokens:
                pos = len(prefix)+len(generated)-1
                draft_start = time.perf_counter()
                with torch.inference_mode():
                    proposal = draft_model.generate(torch.tensor([prefix+generated]), max_new_tokens=args.gamma,
                               do_sample=False, pad_token_id=0, eos_token_id=None).tolist()[0][-args.gamma:]
                draft_s += time.perf_counter()-draft_start
                share += runner.bind('ids', np.array([[generated[-1]]+proposal], np.int64))
                share += runner.bind('draft', np.array([proposal], np.int64))
                share += runner.bind('pos', np.array(pos, np.int64), public=True)
                actual, timing = runner.run(compiled[method, 'decode'], inputs, {3:'cache'}, [0, 1, 2])
                valid, k, token = bool(actual[0][0]), int(actual[1][0]), int(actual[2][0])
                assert (valid and 0 <= k <= args.gamma) or (not valid and k == token == 0)
                emitted = (proposal[:k]+[token])[:args.tokens-len(generated)] if valid else []
                generated.extend(emitted)
                stamps.extend([time.perf_counter()-start]*len(emitted))
                blocks.append(dict(phase='decode', position=pos, valid=valid, accepted=k, token=token,
                                   draft=proposal, outputs=emitted, **timing))
            elapsed = time.perf_counter()-start
            return dict(method=method, model=args.model, index=index, question_id=metadata[index]['question_id'],
                        warmup=warmup, prefix_ids=prefix, token_ids=generated, completed=valid and len(generated)==args.tokens,
                        request_wall_s=elapsed, runtime_s=sum(b['runtime_s'] for b in blocks),
                        input_share_s=share, draft_s=draft_s, reconstruct_s=sum(b['reconstruct_s'] for b in blocks),
                        ttft_s=stamps[0] if stamps else None, token_timestamps_s=stamps,
                        decode_calls=len(blocks)-1, valid_calls=sum(b['valid'] for b in blocks), blocks=blocks)

        for method in methods:
            announce('warmup_start', method=method)
            row = request(method, indices[0], True)
            warmups.append(row)
            write('warmups.json', warmups)
            announce('warmup_end', method=method, seconds=row['request_wall_s'], completed=row['completed'])
            assert row['completed'], 'Warmup failed its range check; inspect recorded evidence'
        rng = random.Random(1003)
        order = indices.copy()
        rng.shuffle(order)
        for index in order:
            names = methods.copy()
            rng.shuffle(names)
            quartet = []
            for method in names:
                announce('request_start', index=index, method=method)
                row = request(method, index)
                rows.append(row); quartet.append(row)
                write('requests.json', rows)
                announce('request_end', index=index, method=method, seconds=row['request_wall_s'],
                         completed=row['completed'], calls=row['decode_calls'])
            same = all(r['token_ids'] == quartet[0]['token_ids'] for r in quartet)
            calls_same = len({r['decode_calls'] for r in quartet}) == 1
            with torch.inference_mode():
                reference = target_model.generate(torch.tensor([quartet[0]['prefix_ids']]), max_new_tokens=args.tokens,
                            do_sample=False, pad_token_id=0, eos_token_id=None).tolist()[0][-args.tokens:]
            for row in quartet:
                row.update(all_methods_equal=same, all_call_counts_equal=calls_same, torch_greedy_ids=reference,
                           torch_sequence_equal=row['token_ids']==reference,
                           torch_token_mismatches=sum(a!=b for a,b in zip(row['token_ids'],reference)))
            write('requests.json', rows)
            assert same and calls_same and all(r['completed'] for r in quartet), 'Quartet failed equivalence checks'
        (out/'COMPLETE').write_text('All planned requests completed.\n')
        announce('complete', requests=len(rows))
    except BaseException:
        (out/'FAILURE.txt').write_text(traceback.format_exc())
        raise
    finally:
        for runner in runners.values():
            runner.close()


if __name__ == '__main__':
    main()
