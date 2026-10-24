"""Screening guidance for a graded photograph.

Grades 2 to 4 (moderate NPDR or worse) are "referable DR", the threshold used
by screening programmes and by Gulshan et al., JAMA 2016;316(22):2402-10.

An image is referred when its predicted grade is referable or when the
model's summed probability of the referable grades reaches a threshold tuned
on validation data (``make tune-referral``). The second condition catches
cases the model under-grades, at the cost of some unnecessary referrals. This
is the rule ``drscreen.evaluation.metrics.referral_decisions`` measures.

The recall intervals for grades 0 and 1 are the ones this prototype documents
in its README; a real programme sets its own.
"""

from dataclasses import dataclass

REFERABLE_MIN_GRADE = 2
MAX_GRADE = 4


@dataclass(frozen=True)
class Referral:
    """What to do after a screening result.

    Attributes:
        refer: Whether to refer to an ophthalmologist.
        guidance: The advice for the person reading the result.
    """

    refer: bool
    guidance: str


def referral_for(grade: int, referable_probability: float, threshold: float) -> Referral:
    """Return the screening guidance for a graded photograph.

    Args:
        grade: Predicted grade from 0 (no DR) to 4 (proliferative DR).
        referable_probability: Model probability of grade 2 or worse.
        threshold: Referable probability at which to refer.

    Returns:
        The referral decision and guidance.

    Raises:
        ValueError: If the grade is outside 0 to 4.
    """
    if not 0 <= grade <= MAX_GRADE:
        raise ValueError(f"ICDR grades run from 0 to {MAX_GRADE}, got {grade}.")
    if grade >= REFERABLE_MIN_GRADE:
        return Referral(True, "Refer to an ophthalmologist soon.")
    if referable_probability >= threshold:
        return Referral(
            True,
            f"Refer to an ophthalmologist soon. The most likely grade is {grade}, but the "
            f"model gives a {referable_probability:.0%} chance of moderate or worse DR, "
            f"at or above the {threshold:.0%} screening threshold.",
        )
    if grade == 1:
        return Referral(False, "Repeat screening in 6 months.")
    return Referral(False, "Repeat screening in 12 months.")
