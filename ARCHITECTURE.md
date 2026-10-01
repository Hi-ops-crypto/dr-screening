# Architecture

This document explains how `src/drscreen` is divided and why. The guiding rule:
each module depends only on modules that are more stable and more general than
itself, and the deployed web app pulls in as little as possible.

## Modules and their responsibilities

| Module | Responsibility | May import |
|---|---|---|
| `config`, `exceptions`, `logging_setup` | Typed YAML loading, error hierarchy, logging setup | nothing internal except `exceptions` |
| `data` | Dataset fetching, manifest schema, preprocessing, augmentation, torch `Dataset` | `config`, `exceptions` |
| `models` | Build a classifier from a backbone name | `config`, `exceptions` |
| `evaluation` | Metrics and evaluation reports | `config`, `exceptions` |
| `training` | Training loop, class weighting, checkpoints, run logs | `data`, `models`, `evaluation` |
| `explain` | Grad-CAM over any convolutional classifier | nothing internal |
| `inference` | Validate input, quality gate, grade, referral guidance | `data.preprocessing`, `models`, `explain` |
| `risk` | Screening-priority tier from clinical factors | `config`, `exceptions` |
| `app` | Gradio UI and audit log | `inference`, `risk` |

## Why the boundaries are drawn here

**Data does not know about models.** Preprocessing turns a raw photograph into
a normalised array and nothing else. That makes it pure and unit-testable
without torch models, and lets training and inference share exactly one
implementation. Any divergence between the preprocessing a model was trained
with and the preprocessing it is served with silently degrades predictions,
so there must be only one copy.

**Inference does not import training.** Training code carries optimisers,
augmentation, samplers and dataset tooling that the deployed app never needs.
Keeping it out of the import graph keeps the Hugging Face Space small and makes
it impossible for serving to accidentally depend on training-only behaviour
such as augmentation.

**Evaluation is separate from training.** The same metrics run during training
(on the APTOS validation split) and afterwards on the external IDRiD set.
Putting them in their own module means the external evaluation does not need
the training loop.

**Explainability is standalone.** Grad-CAM takes a model, a target layer and an
input tensor. It knows nothing about DR, so it can be used from a notebook, the
evaluation report or the app.

**The app contains no model logic.** `app` turns user input into calls to
`inference.Screener` and `risk`, and turns results into UI components. Swapping
Gradio for another front end, or calling the screener from a batch job, needs
no changes outside `app`.

**Risk is independent of everything with weights.** The risk calculator is
pure logic over clinical numbers. It imports neither torch nor any model code,
so it is fast to test and can be audited on its own against its cited sources.

## Configuration

Each entry point loads one YAML file from `configs/` into a frozen Pydantic
model built from `config.StrictModel`. Unknown keys are rejected, so a typo or
an orphaned key is an error, not a silently ignored setting. Overrides from the
command line can only change keys that already exist in the file, so the file
is always the complete list of what a run can be configured with.

## Errors

Deliberate failures raise subclasses of `exceptions.DrScreenError` carrying a
message that says what to fix. Entry points catch `DrScreenError`, log it and
exit non-zero; anything else is a bug and propagates with its traceback.
