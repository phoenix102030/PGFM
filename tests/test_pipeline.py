"""Exercise both training stages with temporary synthetic data on CPU."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from pgfm.cli import evaluate, predict_physics, train, train_physics

ROOT = Path(__file__).resolve().parents[1]


def run_cli(module, arguments):
    with patch('sys.argv', [module.__name__, *map(str, arguments)]):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.main()


class PipelineTest(unittest.TestCase):
    def test_two_stage_training_export_and_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix='pgfm_test_') as directory:
            root = Path(directory)
            cache = root / 'data'
            cache.mkdir()
            cfg = yaml.safe_load((ROOT / 'configs/pgfm.yaml').read_text())
            cfg['data'].update(cache_dir=str(cache), split_day='2023-01-03', past_hours=24,
                               horizon_hours=2, bg_scalar_keys=['kp'], bg_map_keys=[])
            for split in ('train', 'val'):
                cfg['data'][f'future_physics_convgru_{split}_path'] = str(root / 'physics' / f'forecasts_{split}.npz')
            cfg['physics_forecaster'].update(outdir=str(root/'physics'), device='cpu',
                history_hours=24, horizon_hours=2, hidden_channels=[4], epochs=1,
                batch_size=8, eval_batch_size=8, num_workers=0, use_amp=False,
                max_train_samples=4, max_val_samples=4)
            cfg['train'].update(outdir=str(root/'flow'), device='cpu', epochs=1,
                batch_size=8, eval_batch_size=8, rollout_eval_hours=2,
                rollout_eval_every=1, rollout_eval_max_batches=1)
            cfg['model']['hidden'] = 8
            cfg['fm'].update(ode_steps=4, use_amp=False)
            cfg['eval'].update(days=0, start_hour=0, block_step_hours=24, eval_steps=4)
            rng = np.random.default_rng(42)
            shape = (96, 16, 16)
            for day in range(1, 5):
                date = f'2023-01-{day:02d}'
                seconds = np.datetime64(date, 's').astype(np.int64) + np.arange(96)*900
                arrays = dict(tec=(10 + rng.normal(size=shape)).astype('float32'),
                    time=seconds, utc_hour=np.arange(96, dtype='float32')/4,
                    lat=np.linspace(-87.5,87.5,16,dtype='float32'),
                    lon=np.linspace(0,337.5,16,dtype='float32'),
                    kp=rng.normal(size=96).astype('float32'))
                for key in cfg['data']['phys_term_keys']:
                    arrays[key] = (rng.normal(size=shape)*1e-4).astype('float32')
                for key in cfg['data']['phys_uv_keys']:
                    arrays[key] = rng.normal(size=shape).astype('float32')
                np.savez(cache/f'PHYS_{date}.npz', **arrays)
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump(cfg))
            run_cli(train_physics, ['--config', config])
            physics_ckpt = root/'physics/models/best.pt'
            self.assertTrue(physics_ckpt.exists())
            for split in ('train','val'):
                run_cli(predict_physics, ['--config', config, '--ckpt', physics_ckpt, '--split', split])
                rhs = np.load(root/'physics'/f'forecasts_{split}_future_rhs_rollout.npy')
                self.assertEqual(rhs.shape[1:], (2,1,4,16,16))
                self.assertTrue(np.isfinite(rhs).all())
            run_cli(train, ['--config', config])
            flow_ckpt = root/'flow/models/best.pt'
            self.assertTrue(flow_ckpt.exists())
            run_cli(evaluate, ['--config', config, '--ckpt', flow_ckpt, '--split', 'val', '--save-full-pred-truth'])
            metrics = json.loads((root/'flow/eval/metrics_val.json').read_text())
            self.assertTrue(np.isfinite(metrics['overall']['rmse']))
            self.assertEqual(len(metrics['per_lead']), 2)
            with np.load(root/'flow/eval/full_pred_truth_val.npz') as results:
                self.assertEqual(set(results.files), {'pred','truth'})
                self.assertEqual(results['pred'].shape[1:], (2,16,16))


if __name__ == '__main__':
    unittest.main()
