# DR Screening Prototype

A research prototype that triages retinal fundus photographs for diabetic
retinopathy (DR). An image first passes a quality gate; if usable, a model
assigns an ICDR grade from 0 to 4 and the app shows screening guidance with a
Grad-CAM heatmap. A separate calculator turns clinical risk factors into a
screening-priority tier.

**Live demo:** https://bhmeha--dr-screening-web.modal.run ·
**Weights:** https://huggingface.co/synchrono/dr-screening-models

> **This is a screening aid, not a medical diagnosis.** It is not a medical
> device, has not been clinically validated and must not be used for patient
> care.

## Status

| Stage | State |
|---|---|
| 1. Scaffold: tooling, config, logging | done |
| 2. Data pipeline | done |
| 3. Grading model | done; v4 trained on Kaggle and served by the app |
| 4. Quality model | done; trained on Kaggle |
| 5. Grad-CAM | done |
| 6. Inference | done |
| 7. Risk calculator | done |
| 8. Web app and deployment | done; live on Modal |

Trained weights are not in git. The app expects them in `models/` (see Deployment).

## Results

On IDRiD, 516 photos from a clinic none of the models trained on:

| Grading model | Exact grade | Within one grade | QWK (95% CI) | Referral sensitivity / specificity |
|---|---|---|---|---|
| v1: B0, APTOS only | 52% | 91% | 0.75 (0.71-0.79) | 89% / 79% |
| v2+v3: B0 and B3 on APTOS + EyePACS train, averaged | 64% | 93% | 0.83 (0.79-0.86) | 96% / 69% |
| **v4: B3 on APTOS + EyePACS train and test labels + DDR** (served) | **71%** | **94%** | **0.87 (0.84-0.89)** | **98% / 69%** |
| v4 + v5 (ConvNeXt-Small, same data), averaged; not served | 71% | 93% | 0.86 (0.83-0.88) | 99% / 61% |

v5 alone scored below v4 (validation QWK 0.818 against 0.834; IDRiD 0.79).
Averaging it with v4 raised validation QWK only to 0.841 at about four times
the CPU time per photo and did not help on IDRiD, so v4 stays the served model.

Referral uses the tuned threshold (95% sensitivity on validation). Site
adaptation to IDRiD (fine-tuning v4 on its 330 training photos) raised exact
grade on its 103 untouched test photos from 58% to 74%, QWK 0.83 (0.75-0.91);
`configs/app_site_idrid.yaml` serves that model. One photo takes about one
second on a laptop CPU.

## What it does

1. **Quality gate.** A lightweight 3-class model (good / usable / reject),
   MobileNetV3-Large trained on EyeQ labels. A rejected image stops the pipeline with a retake
   message naming the reason.
2. **Grading.** EfficientNet-B3 fine-tuned on about 41,000 photos from APTOS
   2019, EyePACS (Kaggle DR 2015 training and test labels) and DDR. Several
   checkpoints can be served as an averaged ensemble. The backbone is set in
   config, so it can be swapped (for example for RETFound) without code
   changes.
3. **Explainability.** A Grad-CAM overlay showing which regions drove the
   predicted grade.
4. **Referral guidance.** Refer to an ophthalmologist soon when the grade is
   2 to 4, or when the model's probability of grade 2 or worse reaches a
   threshold tuned for high sensitivity. Otherwise grade 1: repeat screening
   in 6 months; grade 0: repeat in 12 months.
5. **Risk calculator.** Diabetes duration, HbA1c or fasting glucose, systolic
   BP, age and kidney disease give a high, medium or low priority tier with the
   contributing factors listed. Bands, points and cut-offs live in
   `configs/risk.yaml` with their sources.
6. **Audit log.** Each submission is appended to a local CSV as timestamp,
   quality outcome and grade. Images are never stored.

## Architecture overview

```
image -> inference.validation -> quality model (inference.screener) --(reject)--> retake message
                                        | (good / usable)
                                        v
                 data.preprocessing -> grading model -> explain.gradcam
                                        v
                               inference.referral -> app.ui -> app.audit
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
- **Referral by probability threshold.** The first model under-graded
  images from a new clinic, so referring only on the predicted grade missed
  a quarter of referable cases on IDRiD. The app also refers when the summed
  probability of grades 2 to 4 reaches a threshold. `make tune-referral`
  chooses the highest threshold that reaches 95% sensitivity on the APTOS +
  EyePACS validation images, never on IDRiD, and the app refuses a threshold tuned for a
  different checkpoint.
- **Ensembles chosen on validation.** Several grading checkpoints can be
  served together; their softmax probabilities are averaged, optionally with
  each image's mirror image too (`flip_tta`). Candidates are compared by
  quadratic weighted kappa on the pooled APTOS + EyePACS validation images,
  never on IDRiD. The serving config ties the run ids, flip setting and
  referral threshold together, and the app refuses any other combination.
- **Site adaptation.** Clinics grade borderline cases differently, which caps
  a model trained elsewhere at about 60% exact-grade accuracy on a new site.
  `configs/finetune_site.yaml` fine-tunes the trained checkpoints on a few
  hundred labelled images from the target site. With `sources.idrid.adaptation`
  the 413 official IDRiD training images become train/validation and the 103
  official test images stay untouched: on those, v4's exact-grade accuracy
  rose from 58% to 74%, with QWK 0.83. Serve the
  adapted models with `make serve APP_CONFIG=configs/app_site_idrid.yaml`.
- **IDRiD as external validation.** It comes from a different clinic, camera
  and population, and is never used for training or model selection, so it
  measures how the model generalises. The training config refuses IDRiD as a
  source. External metrics carry 95% bootstrap confidence intervals because
  the set is small.
- **Checkpoints are self-describing.** Each stores the backbone, class names,
  preprocessing recipe and input statistics, so evaluation and the app feed
  the model exactly what it was trained on, whatever the current configs say.
- **Quality model sees no photometric augmentation.** Blur, exposure and
  vignetting are what its labels describe; adding them to a "Good" image would
  make the label wrong, so `configs/train_quality.yaml` keeps only framing,
  rotation and flips.
- **Pinned model revision.** The Space downloads weights from a Hub model
  repository at a full commit sha and refuses to start with anything else, so
  what is served cannot change without a config edit.
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
| EyePACS training labels | Grading: train and validation | `tanlikesmath/diabetic-retinopathy-resized` (pinned), 8,000 per grade | ~7.8 GB |
| EyePACS test labels | Grading: train and validation | `benjaminwarner/resized-2015-2019-blindness-detection-images`, 6,000 per grade | ~18.6 GB |
| DDR | Grading: train and validation | `samriddhibagchi/ddr-dataset-credits-to-authors`, 6,000 per grade, ungradable dropped | ~14 GB |
| IDRiD | Grading: external test only (or site adaptation) | Kaggle mirror of the official CC BY 4.0 release, pinned version | ~0.4 GB |
| EyeQ | Quality: train and validation | Labels from the EyeQ GitHub repo, EyePACS images from the mirror above | ~7.8 GB |

APTOS needs a Kaggle API token at `~/.kaggle/kaggle.json` and acceptance of
the competition rules on kaggle.com. EyeQ's images are large; on a local
machine you can skip them with `--override sources.eyeq.enabled=false` and
train the quality model on Kaggle, where the dataset is attached rather than
downloaded; the same applies to the EyePACS test labels and DDR. Datasets are
subject to their own licences and are never redistributed by this repository.
APTOS and EyePACS come from Kaggle competitions whose rules generally restrict
use to research, so models trained on them should not be sold.

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
| Tune the referral threshold on validation | `make tune-referral CHECKPOINT=runs/<run>/best.pt` (or `scripts/tune_referral.py` with repeated `--checkpoint` for an ensemble) | local CPU or Kaggle |
| Web app | `make serve` | local CPU |
| Assemble the Hugging Face Space | `make space` | local |

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

1. Make the code available to Kaggle: upload it as a private Kaggle dataset
   (the notebook uses any attached input containing this `pyproject.toml`), or
   push it to GitHub and set `REPO_URL`.
2. Create a Kaggle notebook from `notebooks/kaggle_training.ipynb`, attach the
   inputs listed in its first cell, and enable the GPU and internet.
3. *Run all*. Fetching, preprocessing (the image cache goes to `/tmp`; it is
   larger than Kaggle's 20 GB output limit), grading training (8 epochs of
   about 40 minutes on a T4), threshold tuning, IDRiD evaluation and quality
   training run in order. *Save Version* keeps the `runs/` output, from which
   each `best.pt` can be downloaded.

Training runs record the git commit they were built from. Kaggle has no git
history, so write it beside the code first: `git rev-parse HEAD > GIT_COMMIT`.

## Tests and quality checks

```bash
make test      # pytest: unit and integration, no datasets or network needed
make lint      # ruff lint, ruff format check, mypy --strict, import contracts
make format    # apply ruff fixes and formatting
```

Tests use synthetic fundus-like images and randomly initialised tiny models,
so they pass on a clean checkout. Pre-commit runs ruff and mypy on every commit.

## Deployment

### Locally

Copy the quality run's `best.pt` to `models/quality.pt` and the grading run's
to `models/grading-v4.pt` (the files `configs/app.yaml` lists), then
`make serve` and open http://127.0.0.1:7860. The grading run ids and referral
threshold in that config must match the checkpoints; `make tune-referral`
prints the lines to paste. Outcomes are appended to `audit/screenings.csv`.

### Hugging Face Space (free CPU tier)

Weights are never committed to git; they live in a Hub model repository.

1. Upload both checkpoints:
   `hf upload <user>/dr-screening-models models/quality.pt quality.pt` and the
   same for `grading-v4.pt`. Note the commit sha the upload reports.
2. In `deploy/hf_space/configs/app.yaml` set `repo_id` and `revision` to that
   sha. The app refuses to start until `revision` is a full 40-character sha.
3. `make space` assembles `build/hf_space/` (entry point, package, configs,
   pinned requirements).
4. `hf upload <user>/<space> build/hf_space . --repo-type space`.

Gradio usage analytics are turned off. Uploads sit in Gradio's temporary cache
for at most about two minutes and are never written to the audit log.

## Known limitations

- Trained on clinic fundus-camera images. Phone-camera degradation
  augmentation is an approximation, not validation on phone captures.
- IDRiD has 516 images from one centre, so per-class metrics for grades 3 and
  4 rest on few examples; the confidence intervals show how wide they are.
  The site-adaptation results rest on only 103 test photos (about ±9 points).
- Candidate models were compared on IDRiD as well as validation, so the
  reported IDRiD scores of the chosen model are slightly optimistic. A final
  claim needs a fresh test set.
- The training data's licences generally restrict use to research; a
  commercial product needs licensed data.
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
