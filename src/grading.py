"""MATH-500 answer grading via math-verify's symbolic comparison.

math-verify natively supports tuples, intervals, and finite sets (see
its grader.py), but its LaTeX extraction only activates inside a LaTeX
environment delimiter -- per its own README: "the latex must be placed
in latex environment to be parsable." Both sides are wrapped in $...$
before parsing so multi-value answers go through LaTeX extraction
instead of falling back to plain-expression parsing.
"""

from math_verify import parse, verify


def wrap_latex(s: str) -> str:
    """Ensure s is wrapped in a single $...$ LaTeX environment."""
    s = s.strip().strip("$")
    return f"${s}$"


def grade_answer(gold: str, predicted: str) -> bool:
    """Return True if predicted is equivalent to gold under math-verify."""
    try:
        return verify(parse(wrap_latex(gold)), parse(wrap_latex(predicted)))
    except Exception:
        return False
