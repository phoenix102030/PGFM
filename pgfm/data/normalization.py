"""Single-pass, training-only normalization shared by both DDP stages."""
import hashlib, json, os
from pathlib import Path
import numpy as np
from .daily2d import FIELDS
from .flow import _load_phys_npz, list_phys_cached_days
from .split import configured_splits

def fingerprint(cfg):
    d = cfg['data']
    root = Path(d['cache_dir'])
    days = configured_splits(list_phys_cached_days(str(root)), d)['train']
    payload = {'root': str(root.resolve()), 'days': list(days), 'mapping': FIELDS, 'manifest': hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest(), 'terms': d['phys_term_keys'], 'uv': d['phys_uv_keys'], 'bg': d['bg_scalar_keys'] + d['bg_map_keys']}
    return (hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), days)

def load_normalization(cfg):
    path = cfg['data'].get('normalization_path')
    if not path:
        return None
    saved = json.loads(Path(path).read_text())
    key, _ = fingerprint(cfg)
    if saved['fingerprint'] != key:
        raise ValueError('Stale normalization artifact')
    return saved['stats']

def prepare(cfg):
    key, days = fingerprint(cfg)
    path = Path(cfg['data']['normalization_path'])
    if path.exists():
        load_normalization(cfg)
        return
    d = cfg['data']
    keys = ['tec'] + d['phys_term_keys'] + d['phys_uv_keys'] + d['bg_scalar_keys'] + d['bg_map_keys']
    accum = {k: [0, 0.0, 0.0] for k in keys + ['rhs']}
    for day in days:
        with _load_phys_npz(d['cache_dir'], day) as z:
            arrays = {k: np.asarray(z[k], dtype='float64') for k in keys}
            arrays['rhs'] = sum((arrays[k] for k in d['phys_term_keys']))
            for k, a in arrays.items():
                if not np.isfinite(a).all():
                    raise ValueError(f'Invalid {day}:{k}')
                n = a.size
                mu = float(a.mean())
                cnt, mean, m2 = accum[k]
                delta = mu - mean
                accum[k] = [cnt + n, mean + delta * n / (cnt + n), m2 + float(np.square(a - mu).sum()) + delta ** 2 * cnt * n / (cnt + n)]
    stats = {k: [m, max(float(np.sqrt(m2 / n)), 1e-08)] for k, (n, m, m2) in accum.items()}
    payload = {'fingerprint': key, 'scope': 'training source files only; equal grid weights, float64 online moments', 'days': list(days), 'stats': stats}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    assert json.loads(tmp.read_text()) == payload
    os.replace(tmp, path)
