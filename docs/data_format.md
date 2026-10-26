# Input data format

No data are distributed. Readers consume already prepared daily grids; raw
simulation integration and scientific preprocessing are outside this release.

## Daily NPZ files

Place files named `PHYS_YYYY-MM-DD.npz` in `data.cache_dir`. Each day normally
contains 96 frames sampled every 900 seconds on a common latitude–longitude grid.
At least one contiguous history plus forecast window is required in each split;
a 24-hour history and 24-hour forecast need more than two complete daily files
for useful origin coverage.

| Field | Shape | Meaning |
| --- | --- | --- |
| `tec` | `(T, H, W)` | TEC in TECU |
| `time` | `(T,)` | UTC Unix seconds, or NumPy datetime64 |
| `utc_hour` | `(T,)` | Fractional UTC hour in [0, 24) |
| `lat`, `lon` | `(H,)`, `(W,)` | Grid coordinates in degrees |
| `op_prod_tecu_rate`, `op_loss_tecu_rate`, `op_drift_tecu_rate`, `op_wind_tecu_rate` | `(T, H, W)` | Signed column rates in TECU/s |
| `ui_exb_layer_mean`, `vi_exb_layer_mean` | `(T, H, W)` | Zonal and meridional ExB fields, consistent source units |
| configured scalar drivers | `(T,)` | Background time series |
| configured map drivers | `(T, H, W)` | Background maps |

The default scalar drivers are `kp`, `f107d`, `f107a`, `hpower`, `ctpoten`,
`bximf`, `byimf`, `bzimf`; map drivers are `tlbc`, `ulbc`, `vlbc`.
Missing required physics fields raise an error. The inherited NPZ path and model
replace nonfinite values with zero in several places; supply finite, physically
validated input rather than relying on that fallback as a missing-data method.
The model uses grid-weighted MSE and does not apply an observation mask.

Means and standard deviations are estimated from training files only. Scalar,
map and UV fields are standardized separately; process rates remain in TECU/s
in the flow model. Normalization metadata are stored with checkpoints. The
normalization helpers retain the original numerical epsilon conventions.

## Daily NetCDF4 files

`configs/pgfm_daily2d.yaml` reads `YYYY_doy_DDD.nc` through h5py. These files must
have HDF5-compatible NetCDF4 storage and the same time/grid ordering as above.
The adapter maps `TEC_model` to `tec`, `Kp` to `kp`, and
`OP_PROD_COLUMN`, `OP_LOSS_COLUMN`, `OP_DRIFT_COLUMN`, `OP_WIND_COLUMN` to the four
rate fields with scale `1e-12`. This conversion assumes the source column rates
are in the units for which multiplying by `1e-12` gives TECU/s; check source units.
`UI_ExB_HEIGHT_MEAN` and `VI_ExB_HEIGHT_MEAN` supply UV fields. `OP_COLUMN_VALID`
and `EXB_COLUMN_VALID` must be entirely valid. The adapter rejects nonfinite
source arrays. `time` must be UTC Unix seconds for this adapter.

The monthly split uses validation origin days 10–11 and test origin days 20–22,
with a full calendar day on each side reserved for history/future support.
All buffer dates must exist. Training uses the remaining dates and only windows
with continuous 900-second timestamps. These rules require a 24-hour history
and horizon. The NPZ example instead uses a single chronological train/validation
split and does not define a separate test set.

## Exported physics forecasts

The physics prediction command writes paired files:

- `forecasts_SPLIT.npz`: metadata including `anchor_time`.
- `forecasts_SPLIT_future_rhs_rollout.npy`: float32 RHS values with shape
  `(origins, horizon_hours, 1, 4, H, W)` in TECU/s.

Keep both together. PGFM matches forecasts to integer UTC origin times, not
array positions. Default hourly ConvGRU predictions are repeated across each
hour's four subintervals. Forecast metadata also include a JSON summary.

## Direct model inputs

The flow model takes standardized `x_tau` `(B,1,H,W)`, `tau` `(B,1,1,1)`,
`tec_hist` `(B,past_hours,H,W)`, `bg_scalar_hist` `(B,Cs,past_hours)` and
`bg_map_hist` `(B,Cm,past_hours,H,W)`. Supply empty tensors with a zero channel
dimension for absent driver groups. Physics inputs are `(B,4,4,H,W)` for
process terms and `(B,2,4,H,W)` for UV. Optional `future_rhs_hour` `(B,1,4,H,W)`
overrides those process inputs in ConvGRU mode. `return_aux=True` returns the
data velocity, physics velocity, lambda map and intermediate physics quantities.
