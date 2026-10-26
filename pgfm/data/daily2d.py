"""Read daily NetCDF4 fields through the physics dataset interface."""
from pathlib import Path
from datetime import date, timedelta
import h5py
import numpy as np
FIELDS = {'tec': ('TEC_model', 1.0), 'kp': ('Kp', 1.0), 'f107d': ('f107d', 1.0), 'hpower': ('hpower', 1.0), 'ctpoten': ('ctpoten', 1.0), **{f'op_{n}_tecu_rate': (f'OP_{n.upper()}_COLUMN', 1e-12) for n in ('prod', 'loss', 'drift', 'wind')}, 'ui_exb_layer_mean': ('UI_ExB_HEIGHT_MEAN', 1.0), 'vi_exb_layer_mean': ('VI_ExB_HEIGHT_MEAN', 1.0)}

def daily_path(root, day):
    d = date.fromisoformat(day)
    return Path(root) / f'{d.year}_doy_{d.timetuple().tm_yday:03d}.nc'

def daily_days(root):
    return sorted(((date(int(p.name[:4]), 1, 1) + timedelta(days=int(p.stem[-3:]) - 1)).isoformat() for p in Path(root).glob('????_doy_???.nc')))

class Daily2D:

    def __init__(self, root, day):
        self.f = h5py.File(daily_path(root, day), 'r')
        self.files = [k for k, (src, _) in FIELDS.items() if src in self.f] + ['time', 'lat', 'lon', 'utc_hour']

    def __contains__(self, key):
        return key in self.files

    def shape(self, key):
        return self.f[FIELDS.get(key, (key, 1.0))[0]].shape

    def __getitem__(self, key):
        if key == 'utc_hour':
            return self.f['time'][:] % 86400 / 3600.0
        src, scale = FIELDS.get(key, (key, 1.0))
        a = self.f[src][:]
        if not np.isfinite(a).all():
            raise ValueError(f'Nonfinite source field {self.f.filename}:{src}; no zero filling permitted')
        if key.startswith('op_') and (not self.f['OP_COLUMN_VALID'][:].all()):
            raise ValueError('Invalid OP column; refusing unmasked input')
        if 'exb_layer_mean' in key and (not self.f['EXB_COLUMN_VALID'][:].all()):
            raise ValueError('Invalid ExB column; refusing unmasked input')
        return a * scale if scale != 1.0 else a

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, 'f'):
            self.f.close()
