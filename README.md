# PGFM

Physics-guided flow matching for total electron content (TEC) forecasting.
This repository contains the Proposed model extracted from TEC_demo: a causal
ConvGRU physics forecaster followed by a conditional, one-hour piecewise flow
model with a learned spatial mixing weight. The one-hour model rolls forward
autoregressively for a 24-hour forecast.

Only the model, its data readers, training, prediction and evaluation are included.
Training data, checkpoints, cached forecasts, benchmark models and historical
figures are excluded. The source equations and checkpoint parameter names are
preserved. Default configurations select the best PGFM checkpoint by full
24-hour rollout validation MSE each epoch.

## Installation

Use Python 3.10 or later and install the package from this directory:

```bash
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml`. CPU execution is supported;
full-size training is intended for a suitable GPU. The local verification
environment uses PyTorch 2.10.0 and NumPy 2.4.2; older supported versions have
not been exhaustively tested.

## Training and evaluation

Run commands from the repository root. Edit `configs/pgfm.yaml`, especially
`data.cache_dir`, `data.split_day`, batch sizes and device settings. All relative
paths resolve against the working directory. The default input is a sequence of
prepared daily `PHYS_YYYY-MM-DD.npz` files, described in
[data format](docs/data_format.md).

First train the physics forecaster and export its predictions for both splits:

```bash
python -m pgfm.cli.train_physics --config configs/pgfm.yaml
python -m pgfm.cli.predict_physics --config configs/pgfm.yaml --ckpt outputs/pgfm/physics/models/best.pt --split train
python -m pgfm.cli.predict_physics --config configs/pgfm.yaml --ckpt outputs/pgfm/physics/models/best.pt --split val
```

Then train PGFM and evaluate the autoregressive TEC forecast:

```bash
python -m pgfm.cli.train --config configs/pgfm.yaml
python -m pgfm.cli.evaluate --config configs/pgfm.yaml --ckpt outputs/pgfm/models/best.pt --split val --save-full-pred-truth
```

Both stages save `models/best.pt`, `models/last.pt` and `history.json` under their
configured output directories. PGFM checkpoints include optimizer and AMP state.
To resume a PGFM run, add `--resume outputs/pgfm/models/last.pt` and increase the
configured total epoch count. Resume restores training state, but does not
restore the random-number-generator state for bitwise replay.

For multiple CUDA devices, use the corresponding training module with torchrun:

```bash
torchrun --standalone --nproc_per_node=2 -m pgfm.cli.train --config configs/pgfm.yaml
```

The alternative `configs/pgfm_daily2d.yaml` reads prepared NetCDF4 daily files,
uses four scalar drivers and no background maps, and defines buffered monthly
train/validation/test splits. Use it for **all** stages of a run; export physics
forecasts for `--split test` before TEC test evaluation. Its outputs are under
`outputs/pgfm_daily2d`. These are two data configurations of the same model.

`eval.mode: truth` means evaluation against observed TEC. It does not select
future physics: `data.future_physics_mode` and `eval.future_physics_mode` do that,
and both release configurations explicitly use `convgru`. Future physics `truth`
mode is retained as an explicit diagnostic option, but uses information unavailable
in a causal forecast.

## Model and lambda

```text
u = lambda_data * u_data + (1 - lambda_data) * u_phys
lambda_data = 1 / (1 + exp(clamp(lambda_head(features), -30, 30)))
```

`lambda_data` has shape `(batch, 1, latitude, longitude)` and changes with the
interpolated TEC state, flow time and historical context. It is learned jointly
with the data branch using Adam; it has no separate labels or optimizer.
The physics weight is `1 - lambda_data`. The loss is:

```text
MSE(u, u_target)
+ lambda_balance * mean((lambda_data - 0.5)^2)
+ lambda_phys * MSE(u, u_phys)
```

Both regularization coefficients default to `1e-4`. The ConvGRU is trained
separately to predict physical RHS residuals relative to a diurnal baseline.
Its exported predictions are fixed during PGFM training; gradients do not pass
back into ConvGRU. The U-Net data head and mixing head are trained jointly.

| Component | Implementation |
| --- | --- |
| Flow model and mixing head | `pgfm/models/flow.py` |
| Shared U-Net blocks | `pgfm/models/blocks.py` |
| ConvGRU physics forecaster | `pgfm/models/physics.py` |
| Flow targets, loss and integration | `pgfm/training.py` |
| Dataset construction and splits | `pgfm/data/` |
| Training and evaluation entry points | `pgfm/cli/` |

Evaluation reports grid-weighted overall and per-lead metrics in TECU, together
with relative-error metrics. Deterministic CRPS equals MAE. These metrics do not
use latitude area weighting. The release makes no numerical skill claim without
training data and checkpoints.

## Inference integration

This package supplies model code, not pretrained weights or a serving API.
Deployment requires compatible ConvGRU and PGFM checkpoints, their saved
normalization statistics, historical inputs and an application-specific inference
entry point. The CLI evaluation path reads future observations to score forecasts
and is not a standalone live forecasting interface.

`PiecewisePhysFlowMatchingUNet.forward` produces the fused velocity; the trainer's
`sample_one_hour` integrates it and `rollout_hours` produces autoregressive TEC
forecasts. These model-level methods do not require future TEC targets.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests generate temporary synthetic inputs and check mixing, gradients,
piecewise targets, physics scaling, checkpoint reloads and the complete two-stage
CPU workflow. They require no research data. Synthetic checks establish software
behavior, not forecasting performance.
