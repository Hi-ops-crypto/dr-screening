# Architecture

This document explains how `src/drscreen` is divided and why. The guiding rule:
each module depends only on modules that are more stable and more general than
itself, and the deployed web app pulls in as little as possible.

## Modules and their responsibilities

| Module | Responsibility | May import |
|---|---|---|
| `config`, `exceptions`, `logging_setup` | Typed YAML loading, error hierarchy, logging setup | nothing internal except `exceptions` |
| `cli` | Shared argument parsing and error handling for `scripts/` | `logging_setup`, `exceptions` |
| `devices`, `reproducibility` | Device selection; seeding and git provenance | `exceptions` |
| `data` | Dataset fetching, manifest schema, preprocessing, augmentation, torch `Dataset` | `config`, `exceptions` |
| `models` | Build a classifier from a backbone name; save and load self-describing checkpoints; average several checkpoints as an ensemble | `data.preprocessing`, `data.tensors` |
| `evaluation` | Metrics, batch prediction, reports, external evaluation of a checkpoint | `data`, `models`, `devices` |
| `training` | Training loop, class weighting, run logs | `data`, `models`, `evaluation`, `reproducibility` |
| `explain` | Grad-CAM over any convolutional classifier, heatmap overlay | nothing internal |
| `inference` | Validate input, quality gate, grade, referral guidance | `config`, `devices`, `data.preprocessing`, `data.tensors`, `models`, `explain` |
| `risk` | Screening-priority tier from clinical factors | `config`, `exceptions` |
| `app` | Gradio wiring and checkpoint download (`ui`), result HTML and theme (`views`), audit log | `config`, `inference`, `risk` |

## Why the boundaries are drawn here

**Inside `data`, preprocessing is pure.** `data.preprocessing` works on
arrays only: no file I/O, no pandas, no dataset knowledge. `data.image_io`
does reading and writing, `data.sources` knows each dataset's layout,
`data.fetch` downloads, and `data.prepare` runs preprocessing over a whole
manifest. Inference imports only `preprocessing` and `tensors`, so the
deployed app never carries dataset or download code.

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

**Checkpoints carry their own preprocessing.** `models.checkpoint` stores the
recipe and input statistics with the weights. Anything that loads a model
takes them from there, so a later edit to `configs/data.yaml` cannot silently
change what an existing model is fed. This is why `models` may import the
preprocessing schema, and nothing else from `data`.

**One ensemble implementation.** `models.ensemble` is the only code that
combines several checkpoints' probabilities, with or without flipped copies.
Evaluation and the app both call it, and a test checks that the app's
probabilities match it, so the numbers we report are the numbers users see.

**External rows are never chosen with.** Evaluation scores only rows whose
split is `external`; threshold tuning and checkpoint selection use only
`val` rows; training uses only `train` rows. IDRiD is entirely external unless
site adaptation is switched on, in which case only its official test subset
stays external. The rule is enforced by the split column rather than by
refusing a dataset name.

**Explainability is standalone.** Grad-CAM takes a model, a target layer and an
input tensor. It knows nothing about DR, so it can be used from a notebook, the
evaluation report or the app.

**The app contains no model logic.** `app` turns user input into calls to
`inference.Screener` and `risk`, and turns results into HTML. `views` holds all
markup and styling as plain strings, so the look can be tested without a
browser and changed without touching the handlers in `ui`. Swapping
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

## Enforcement

The import rules above are checked by `import-linter` as part of `make lint`;
contracts live in `pyproject.toml`. The app contract checks direct imports
only, since the app necessarily reaches models through `inference`.

## Errors

Deliberate failures raise subclasses of `exceptions.DrScreenError` carrying a
message that says what to fix. Entry points catch `DrScreenError`, log it and
exit non-zero; anything else is a bug and propagates with its traceback.
