"""HTML for the app's result panels, and the theme they are styled with.

Everything shown to the person comes from here as plain HTML strings, so the
look can be tested and changed without touching the handlers in ``ui``. Every
value interpolated into HTML is escaped.

Design: one decision per photo, stated in the largest type on the page; the
ICDR scale drawn as five probability bars with the referable zone bracketed;
photo and heatmap framed as circles, the way a retina is seen through the lens.
Colour always repeats what the text already says.
"""

import base64
from html import escape

import cv2
import gradio as gr
import numpy as np
import numpy.typing as npt

from drscreen.inference.referral import REFERABLE_MIN_GRADE
from drscreen.inference.screener import ScreeningResult
from drscreen.risk.calculator import RiskAssessment, Tier

JPEG_QUALITY = 88
PERCENT = 100
MIN_SHOWN, MAX_SHOWN = 0.005, 0.995  # beyond these a whole percentage rounds to 0 or 100

FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap"
)
HEAD = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    f'<link rel="stylesheet" href="{FONT_URL}">'
)
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Atkinson Hyperlegible"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="#F5F7F8",
    body_background_fill_dark="#11181F",
    button_primary_background_fill="#16202A",
    button_primary_background_fill_hover="#2A3744",
    button_primary_text_color="#FFFFFF",
    button_primary_background_fill_dark="#E9EEF1",
    button_primary_background_fill_hover_dark="#FFFFFF",
    button_primary_text_color_dark="#11181F",
    color_accent="#16202A",
    color_accent_soft="#E9EEF1",
    color_accent_soft_dark="#222D37",
    border_color_accent="#16202A",
    border_color_accent_dark="#E9EEF1",
)

CSS = """
.gradio-container {
  --ink: #16202A; --muted: #4A5866; --paper: #F5F7F8; --surface: #FFFFFF;
  --rule: #D5DCE1; --gold: #C8962E;
  --refer: #B3261E; --refer-tint: #FBEAE8;
  --watch: #8A5A00; --watch-tint: #FBF1DD;
  --routine: #2F6B3F; --routine-tint: #E7F2EA;
  --retake: #4A5866; --retake-tint: #EDF1F4;
  max-width: 1120px !important; margin: 0 auto !important;
  font-family: "Atkinson Hyperlegible", system-ui, sans-serif;
}
.dark .gradio-container {
  --ink: #E9EEF1; --muted: #A9B6C1; --paper: #11181F; --surface: #18222B;
  --rule: #2E3B47; --refer: #F08A80; --refer-tint: #3A1E1C;
  --watch: #E2B45A; --watch-tint: #33280F; --routine: #7FC495; --routine-tint: #16301F;
  --retake: #A9B6C1; --retake-tint: #222D37;
}
.masthead { padding: 8px 0 18px; border-bottom: 1px solid var(--rule); margin-bottom: 8px; }
.masthead__name { font-size: 1.75rem; font-weight: 700; color: var(--ink); margin: 0; }
.masthead__note { color: var(--muted); margin: 4px 0 0; max-width: 70ch; }
.tips { color: var(--muted); margin: 4px 0 0; padding-left: 1.1em; line-height: 1.6; }

.result {
  border-left: 6px solid var(--tone); background: var(--surface); color: var(--ink);
  padding: 20px 24px 22px; border-radius: 4px; line-height: 1.5;
}
.result--refer { --tone: var(--refer); --tint: var(--refer-tint); }
.result--watch { --tone: var(--watch); --tint: var(--watch-tint); }
.result--routine { --tone: var(--routine); --tint: var(--routine-tint); }
.result--retake, .result--empty { --tone: var(--retake); --tint: var(--retake-tint); }
.result--empty { border-left: 2px dashed var(--rule); background: transparent; }
.result--empty .result__decision { font-size: 1.25rem; color: var(--muted); }
.result--empty .result__detail { color: var(--muted); margin: 0; }
.result__decision {
  font-size: clamp(1.6rem, 1.2rem + 1.6vw, 2.25rem); font-weight: 700; line-height: 1.15;
  color: var(--tone); margin: 0 0 6px;
}
.result__detail { margin: 0 0 14px; max-width: 62ch; }
.result__grade { margin: 18px 0 0; }
.result__meta { color: var(--muted); font-size: 0.9rem; margin: 16px 0 0; max-width: 70ch; }

.scale { display: grid; grid-template-columns: repeat(5, 1fr); column-gap: 8px; margin-top: 8px; }
.scale__col { display: grid; grid-template-rows: 110px auto auto auto; justify-items: center; }
.scale__track {
  width: 100%; max-width: 56px; height: 110px; display: flex; align-items: flex-end;
  background: var(--paper); border-radius: 3px;
}
.scale__bar {
  width: 100%; height: calc(var(--p) * 100%); min-height: 2px;
  background: var(--muted); border-radius: 3px 3px 0 0;
}
.scale__col--referable .scale__bar { background: var(--refer); opacity: 0.55; }
.scale__col--predicted .scale__bar {
  opacity: 1; outline: 3px solid var(--gold); outline-offset: 2px;
}
.scale__pct { font-size: 0.85rem; color: var(--muted); margin-top: 6px; }
.scale__grade { font-weight: 700; font-size: 1.15rem; }
.scale__name { font-size: 0.8rem; color: var(--muted); text-align: center; line-height: 1.25; }
.scale__zone {
  grid-column: 3 / 6; border-top: 2px solid var(--refer); margin-top: 8px; padding-top: 4px;
  font-size: 0.8rem; color: var(--refer); text-align: center;
}

.views { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 20px; }
.views figure { margin: 0; text-align: center; }
.fundus {
  width: 200px; height: 200px; border-radius: 50%; object-fit: cover; display: block;
  box-shadow: 0 0 0 1px var(--rule);
}
.views figcaption { font-size: 0.85rem; color: var(--muted); margin-top: 8px; }

.factors { list-style: none; padding: 0; margin: 12px 0 0; }
.factors li {
  display: flex; justify-content: space-between; gap: 16px; padding: 8px 0;
  border-bottom: 1px solid var(--rule);
}
.factors li span { font-weight: 700; color: var(--tone); white-space: nowrap; }

button:focus-visible, input:focus-visible {
  outline: 3px solid var(--gold) !important; outline-offset: 2px;
}
@media (max-width: 640px) {
  .result { padding: 16px; }
  .views { gap: 16px; }
  .fundus { width: 128px; height: 128px; }
  .scale__name { font-size: 0.72rem; }
}
"""

MASTHEAD = """
<header class="masthead">
  <p class="masthead__name">DR screening</p>
  <p class="masthead__note">Research prototype. A screening aid, not a diagnosis: it is not
  a medical device, has not been clinically validated and must not be used for patient care.</p>
</header>
"""

PHOTO_TIPS = """
<ul class="tips">
  <li>Fit the whole retina in the frame.</li>
  <li>Hold steady so the vessels look sharp.</li>
  <li>Avoid glare and dark edges.</li>
</ul>
"""

EMPTY_SCREEN = """
<section class="result result--empty">
  <p class="result__decision">No photo screened yet</p>
  <p class="result__detail">Upload a fundus photo, then select Screen photo.</p>
</section>
"""

EMPTY_RISK = """
<section class="result result--empty">
  <p class="result__decision">No assessment yet</p>
  <p class="result__detail">Enter the values, then select Assess priority.</p>
</section>
"""

RISK_TONE = {Tier.HIGH: "refer", Tier.MEDIUM: "watch", Tier.LOW: "routine"}


def render_screening(result: ScreeningResult) -> str:
    """Render a screening outcome: the decision, the grade scale and the images.

    Args:
        result: Outcome from ``Screener.screen``.

    Returns:
        HTML for the result panel.
    """
    quality = result.quality
    if result.grade is None:
        return (
            '<section class="result result--retake" aria-live="polite">'
            '<p class="result__decision">Retake the photo</p>'
            f'<p class="result__detail">{escape(result.retake_message or "")}</p>'
            "</section>"
        )
    grade = result.grade
    headline, _, detail = grade.referral.guidance.partition(". ")
    if grade.referral.refer:
        tone = "refer"
    else:
        tone = "watch" if grade.grade > 0 else "routine"
    names = list(grade.probabilities)
    scale_label = ", ".join(
        f"grade {index} {name} {_percent(probability)}"
        for index, (name, probability) in enumerate(grade.probabilities.items())
    )
    columns = "".join(
        _scale_column(index, name, probability, predicted=index == grade.grade)
        for index, (name, probability) in enumerate(grade.probabilities.items())
    )
    confidence = (
        f" ({escape(_percent(quality.confidence))})" if quality.confidence is not None else ""
    )
    return (
        f'<section class="result result--{tone}" aria-live="polite">'
        f'<p class="result__decision">{escape(headline.rstrip("."))}</p>'
        + (f'<p class="result__detail">{escape(detail)}</p>' if detail else "")
        + f'<div class="scale" role="img" aria-label="Grade probabilities: {escape(scale_label)}">'
        f"{columns}"
        f'<div class="scale__zone">Referable: grade {REFERABLE_MIN_GRADE} or worse</div></div>'
        f'<p class="result__grade">Most likely grade: <strong>{grade.grade}, '
        f"{escape(names[grade.grade])}</strong>. Chance of grade {REFERABLE_MIN_GRADE} or "
        f"worse: <strong>{escape(_percent(grade.referable_probability))}</strong>.</p>"
        '<div class="views">'
        + _figure(grade.photo, "Photo", "The cropped fundus photo that was graded")
        + _figure(
            grade.heatmap,
            "Where the model looked",
            "Heatmap of the regions that most influenced the grade",
        )
        + "</div>"
        f'<p class="result__meta">Image quality: {escape(quality.label)}{confidence}. '
        "The bars rank the grades; they are not calibrated likelihoods. The heatmap shows "
        "where the model looked, not where lesions are.</p></section>"
    )


def render_risk(assessment: RiskAssessment, max_points: int) -> str:
    """Render a screening-priority assessment.

    Args:
        assessment: Output of ``risk.calculator.assess``.
        max_points: Highest total the configured bands allow.

    Returns:
        HTML for the result panel.
    """
    factors = (
        "".join(
            f"<li>{escape(line)}<span>+{points}</span></li>" for line, points in assessment.factors
        )
        or "<li>No factor added points.</li>"
    )
    notes = "".join(f'<p class="result__meta">{escape(note)}</p>' for note in assessment.notes)
    return (
        f'<section class="result result--{RISK_TONE[assessment.tier]}" aria-live="polite">'
        f'<p class="result__decision">{assessment.tier.value.capitalize()} screening priority</p>'
        f'<p class="result__detail">{assessment.points} of {max_points} points.</p>'
        f'<ul class="factors">{factors}</ul>{notes}'
        '<p class="result__meta">The tier combines published risk associations. Its cut-offs '
        "are a design choice, not a validated clinical score.</p></section>"
    )


def _scale_column(index: int, name: str, probability: float, *, predicted: bool) -> str:
    """Render one grade's bar on the scale.

    Args:
        index: The grade.
        name: Display name of the grade.
        probability: Model probability of the grade.
        predicted: Whether this is the most likely grade.

    Returns:
        HTML for the column.
    """
    classes = ["scale__col"]
    if index >= REFERABLE_MIN_GRADE:
        classes.append("scale__col--referable")
    if predicted:
        classes.append("scale__col--predicted")
    return (
        f'<div class="{" ".join(classes)}">'
        '<div class="scale__track">'
        f'<div class="scale__bar" style="--p:{probability:.3f}"></div></div>'
        f'<span class="scale__pct">{escape(_percent(probability))}</span>'
        f'<span class="scale__grade">{index}</span>'
        f'<span class="scale__name">{escape(name)}</span></div>'
    )


def _percent(probability: float) -> str:
    """Format a probability without implying certainty the model does not have.

    Args:
        probability: Value in ``[0, 1]``.

    Returns:
        A whole percentage, with ``<1%`` and ``>99%`` instead of 0% and 100%.
    """
    if probability >= MAX_SHOWN:
        return ">99%"
    if probability < MIN_SHOWN:
        return "<1%"
    return f"{probability * PERCENT:.0f}%"


def _figure(image: npt.NDArray[np.uint8], caption: str, alt: str) -> str:
    """Render an image as a circular figure with a caption.

    Args:
        image: RGB uint8 image.
        caption: Visible caption.
        alt: Text alternative for screen readers.

    Returns:
        HTML for the figure, with the image inlined as a JPEG data URI.
    """
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        return ""
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return (
        f'<figure><img class="fundus" src="data:image/jpeg;base64,{data}" alt="{escape(alt)}">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )
