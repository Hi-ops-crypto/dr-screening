# DR Screening Prototype

A research prototype that triages retinal fundus photographs for diabetic
retinopathy (DR). An image first passes a quality gate; if usable, a model
assigns an ICDR grade from 0 to 4 and the app shows screening guidance with a
Grad-CAM heatmap. A separate calculator turns clinical risk factors into a
screening-priority tier.

> **This is a screening aid, not a medical diagnosis.** It is not a medical
> device, has not been clinically validated and must not be used for patient
> care.

## Status

| Stage | State |
|---|---|
| 1. Scaffold: tooling, config, logging | done |
| 2. Data pipeline | planned |
| 3. Grading model | planned |
| 4. Quality model | planned |
| 5. Grad-CAM | planned |
| 6. Inference | planned |
| 7. Risk calculator | planned |
| 8. Web app and deployment | planned |

Commands below for stages not yet built are the intended interface.

## What it does

1. **Quality gate.** A lightweight 3-class model (good / usable / reject)
   trained on EyeQ labels. A rejected image stops the pipeline with a retake
   message naming the reason.
2. **Grading.** EfficientNet-B0 fine-tuned on APTOS 2019. The backbone is set
   in config, so it can be swapped (for example for RETFound) without code
   changes.
3. **Explainability.** A Grad-CAM overlay showing which regions drove the
   predicted grade.
4. **Referral guidance.** Grades 2 to 4: refer to an ophthalmologist soon.
   Grade 1: repeat screening in 6 months. Grade 0: repeat in 12 months.
5. **Risk calculator.** Diabetes duration, HbA1c or fasting glucose, systolic
   BP, age and kidney disease give a high, medium or low priority tier with the
   contributing factors listed.
6. **Audit log.** Each submission is appended to a local CSV as timestamp,
   quality outcome and grade. Images are never stored.

## Architecture overview

```
image -> inference.validation -> inference.quality --(reject)--> retake message
                                        | (good / usable)
                                        v
                 data.preprocessing -> grading model -> explain.gradcam
                                        v
                               inference.referral -> app.ui
```

Training and serving share only `data.preprocessing` and `models.factory`.
`inference` never imports `training`, and `app` contains no model logic.
[ARCHITECTURE.md](ARCHITECTURE.md) explains why the boundaries are drawn there.

## Key decisions

- **Ben Graham preprocessing** subtracts a blurred copy of the image, which
  evens out lighting differences between cameras. The grading model sees
  normalised images. The quality model sees raw crops, because normalisation
  would hide the lighting and blur problems it exists to detect.
- **Class imbalance.** About half of APTOS is grade 0. Cross-entropy is
  weighted by inverse class frequency, computed on the training split only.
- **QWK, not accuracy.** Grades are ordinal: calling a grade 4 a grade 0 is far
  worse than calling it a 3, and quadratic weighted kappa penalises errors by
  distance. Per-class sensitivity matters most for grades 2 to 4, where a miss
  is a missed referral.
- **IDRiD as external validation.** It comes from a different clinic, camera
  and population, and is never used for training or model selection, so it
  measures how the model generalises.

## Setup

Requires Python 3.11 or newer.

```bash
git clone <repo> && cd dr-screening
make install    # creates .venv, installs pinned deps, editable package, pre-commit hooks
```

Dependencies: `pyproject.toml` lists direct dependencies with compatible
ranges; `requirements.txt` (runtime) and `requirements-dev.txt` (runtime plus
tooling) pin every package exactly. After changing `pyproject.toml`, run
`make lock` and commit the regenerated files.

Downloading data needs a Kaggle API token at `~/.kaggle/kaggle.json` and
acceptance of the APTOS 2019 competition rules on kaggle.com.

## Running each stage

| Stage | Command | Where |
|---|---|---|
| Fetch and normalise datasets | `make fetch-data` | local or Kaggle |
| Preprocess to 512x512 cache | `make prepare-data` | local or Kaggle |
| Class balance and sample grids | `make inspect` | local |
| Train grading model | `make train-grading` | Kaggle GPU via `notebooks/kaggle_training.ipynb` |
| Train quality model | `make train-quality` | Kaggle GPU (EyePACS images are attached there, not downloaded) |
| Train both | `make train` | |
| External validation on IDRiD | `make evaluate` | local CPU |
| Web app | `make serve` | local CPU |

Every script takes `--config <yaml>`, any number of `--override key=value`
(for example `--override train.epochs=5`) and `--log-level`. Training runs
write `runs/<timestamp>/` containing `run.json` (seed, git SHA, resolved
config), `metrics.jsonl` (one line per epoch), `best.pt` and `last.pt`.

## Tests and quality checks

```bash
make test      # pytest: unit and integration, no datasets or network needed
make lint      # ruff lint, ruff format check, mypy --strict
make format    # apply ruff fixes and formatting
```

Tests use synthetic fundus-like images and randomly initialised tiny models,
so they pass on a clean checkout. Pre-commit runs ruff and mypy on every commit.

## Deployment

`deploy/hf_space/` is pushed to a Hugging Face Space on the free CPU tier.
Model weights live in a Hugging Face Hub model repository, are downloaded at
startup at a pinned revision and are never committed to git.

## Known limitations

- Trained on clinic fundus-camera images. Phone-camera degradation
  augmentation is an approximation, not validation on phone captures.
- IDRiD has 516 images from one centre, so per-class metrics for grades 3 and
  4 rest on few examples.
- APTOS has known label noise and near-duplicate images, which can inflate
  internal validation scores. Treat IDRiD results as the honest estimate.
- Displayed confidence is a softmax probability, not a calibrated likelihood.
- The quality gate learned EyePACS image quality, which may not transfer to
  phone captures.
- Grad-CAM shows where the model looked, not where lesions are.
- The risk tier combines published risk associations; the tier cut-offs are a
  documented design choice, not a validated clinical score.
- On free Hugging Face Spaces the audit CSV is lost when the Space restarts.
