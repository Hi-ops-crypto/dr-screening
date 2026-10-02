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
| 2. Data pipeline | done |
| 3. Grading model | done (code; first Kaggle run pending) |
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
  evens out lighting differences between cameras. The original method leaves
  bright and dark halos where the retina meets the black border, and their
  shape differs by camera, which would let a model tell datasets apart by
  their borders. Our variant averages over retina pixels only, so the border
  leaves no halo. The grading model sees normalised images. The quality model
  sees raw crops, because normalisation would hide the lighting and blur
  problems it exists to detect.
- **Patient-grouped splits.** EyePACS images come in left/right pairs from the
  same patient; both eyes always go to the same split so validation never sees
  a patient the model trained on.
- **Augment before normalising.** The cache stores cropped, resized images.
  Phone-camera augmentation (framing, rotation, exposure, white balance, blur,
  vignetting) is applied to them before lighting normalisation, exactly where
  a real phone's degradation would enter the pipeline.
- **Class imbalance.** About half of APTOS is grade 0. Cross-entropy is
  weighted by inverse class frequency, computed on the training split only.
  Weighted loss was chosen over weighted sampling because it keeps every
  image in every epoch instead of repeating the few severe cases.
- **QWK, not accuracy.** Grades are ordinal: calling a grade 4 a grade 0 is far
  worse than calling it a 3, and quadratic weighted kappa penalises errors by
  distance. The best checkpoint is chosen by validation QWK. Reports also give
  per-grade sensitivity and specificity and, because a screening programme
  acts on "refer or not", sensitivity and specificity for referable DR
  (moderate or worse).
- **IDRiD as external validation.** It comes from a different clinic, camera
  and population, and is never used for training or model selection, so it
  measures how the model generalises. The training config refuses IDRiD as a
  source. External metrics carry 95% bootstrap confidence intervals because
  the set is small.
- **Checkpoints are self-describing.** Each stores the backbone, class names,
  preprocessing recipe and input statistics, so evaluation and the app feed
  the model exactly what it was trained on, whatever the current configs say.
- **Swappable backbone.** `model.backbone` takes any timm model name and
  `model.weights_path` a local weights file, so RETFound can be tried by editing
  `configs/train_grading.yaml` alone.

## Setup

Requires Python 3.12 or newer (Kaggle notebooks run 3.13).

```bash
git clone <repo> && cd dr-screening
make install    # creates .venv, installs pinned deps, editable package, pre-commit hooks
```

Dependencies: `pyproject.toml` lists direct dependencies with compatible
ranges; `requirements.txt` (runtime) and `requirements-dev.txt` (runtime plus
tooling) pin every package exactly. After changing `pyproject.toml`, run
`make lock` and commit the regenerated files.

### Data sources

| Dataset | Use | Source | Size |
|---|---|---|---|
| APTOS 2019 | Grading: train and validation | Kaggle competition `aptos2019-blindness-detection` | ~9.5 GB |
| IDRiD | Grading: external test only | Kaggle mirror of the official CC BY 4.0 release, pinned version | ~0.4 GB |
| EyeQ | Quality: train and validation | Labels from the EyeQ GitHub repo, EyePACS images from a pinned Kaggle mirror | ~7.8 GB |

APTOS needs a Kaggle API token at `~/.kaggle/kaggle.json` and acceptance of
the competition rules on kaggle.com. EyeQ's images are large; on a local
machine you can skip them with `--override sources.eyeq.enabled=false` and
train the quality model on Kaggle, where the dataset is attached rather than
downloaded. Datasets are subject to their own licences and are never
redistributed by this repository.

## Running each stage

| Stage | Command | Where |
|---|---|---|
| Download datasets, write manifests with splits | `make fetch-data` | local or Kaggle |
| Preprocess into the image cache | `make prepare-data` | local or Kaggle |
| Class balance and sample grids (raw and preprocessed) | `make inspect` | local |
| Train grading model | `make train-grading` | Kaggle GPU via `notebooks/kaggle_training.ipynb` |
| Train quality model | `make train-quality` | Kaggle GPU (EyePACS images are attached there, not downloaded) |
| Train both | `make train` | |
| External validation on IDRiD | `make evaluate CHECKPOINT=runs/<run>/best.pt` | local CPU |
| Web app | `make serve` | local CPU |

Data stages write to `data/`: `manifests/` (one CSV per dataset),
`processed/<dataset>/` (cached images, their manifest, a `failures.csv` of
images that could not be preprocessed, and the recipe that produced them) and
`reports/`. Re-running `make prepare-data` resumes; changing the
preprocessing recipe requires deleting the old cache, which the script tells
you.

Every script takes `--config <yaml>`, any number of `--override key=value`
(for example `--override optimisation.epochs=5`) and `--log-level`. Training
runs write `runs/<source>-<UTC timestamp>/` containing `run.json` (seed, git
commit and whether the tree had local changes, torch version, device, class
weights, resolved config), `metrics.jsonl` (one line per epoch), `best.pt`
(highest validation QWK) and `last.pt`. Evaluation writes
`eval-<dataset>/metrics.json` and `confusion_matrix.png` next to the
checkpoint it scored.

### Training on Kaggle

1. Push this repository to GitHub (public, or private with a Kaggle secret
   holding a token).
2. Create a Kaggle notebook from `notebooks/kaggle_training.ipynb`, attach the
   inputs listed in its first cell, enable the GPU and internet, and set
   `REPO_URL`.
3. *Run all*. Fetching, preprocessing, training (about 20 epochs, under an
   hour on a T4) and IDRiD evaluation run in order. *Save Version* keeps the
   `runs/` output, from which `best.pt` can be downloaded.

## Tests and quality checks

```bash
make test      # pytest: unit and integration, no datasets or network needed
make lint      # ruff lint, ruff format check, mypy --strict, import contracts
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
  4 rest on few examples; the confidence intervals show how wide they are.
- Swapping in RETFound is a config change for the architecture, but its
  published checkpoint may need its keys mapped to timm's layout first; this
  has not yet been tried.
- APTOS has known label noise and near-duplicate images, which can inflate
  internal validation scores. Treat IDRiD results as the honest estimate.
- Displayed confidence is a softmax probability, not a calibrated likelihood.
- The quality gate learned EyePACS image quality, which may not transfer to
  phone captures.
- Grad-CAM shows where the model looked, not where lesions are.
- The risk tier combines published risk associations; the tier cut-offs are a
  documented design choice, not a validated clinical score.
- On free Hugging Face Spaces the audit CSV is lost when the Space restarts.
