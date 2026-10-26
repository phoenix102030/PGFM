"""Validate the daily adapter's physical units and held-out date buffers."""
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from pgfm.data.daily2d import Daily2D
from pgfm.data.split import configured_splits


class DataTests(unittest.TestCase):
    def test_column_units_and_invalid_mask(self):
        with tempfile.TemporaryDirectory(prefix='pgfm_adapter_') as directory:
            path = Path(directory) / '2023_doy_001.nc'
            with h5py.File(path, 'w') as f:
                f['OP_PROD_COLUMN'] = np.full((4, 2, 3), 2e12)
                f['OP_COLUMN_VALID'] = np.ones((4, 2, 3), dtype=bool)
                f['time'] = np.arange(4)*900 + 1672531200
            with Daily2D(directory, '2023-01-01') as fields:
                np.testing.assert_array_equal(fields['op_prod_tecu_rate'], np.full((4, 2, 3), 2.))
                np.testing.assert_array_equal(fields['utc_hour'], [0., .25, .5, .75])
            with h5py.File(path, 'r+') as f:
                f['OP_COLUMN_VALID'][0, 0, 0] = False
            with Daily2D(directory, '2023-01-01') as fields:
                with self.assertRaisesRegex(ValueError, 'Invalid OP column'):
                    fields['op_prod_tecu_rate']

    def test_monthly_source_buffers_are_disjoint(self):
        days = [(date(2023, 1, 1) + timedelta(days=i)).isoformat() for i in range(31)]
        splits = configured_splits(days, dict(past_hours=24, horizon_hours=24,
            monthly_origin_split={'val':[10,11], 'test':[20,21,22]}))
        self.assertEqual(splits['val'].origin_days, ['2023-01-10','2023-01-11'])
        self.assertEqual(set(splits['val']), {'2023-01-09','2023-01-10','2023-01-11','2023-01-12'})
        for a,b in [('train','val'),('train','test'),('val','test')]:
            self.assertFalse(set(splits[a]) & set(splits[b]))


if __name__ == '__main__':
    unittest.main()
