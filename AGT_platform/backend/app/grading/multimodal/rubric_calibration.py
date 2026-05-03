"""
Rubric-anchored partial credit: half-step raw scores on each criterion scale map to
calibrated credits in [0, 1] via monotone anchor curves (not score/max linear).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

# Default half-step anchor table for a 0..4 rubric scale (inclusive).
# Keys are raw rubric levels r; values are calibrated credit g(r).
DEFAULT_ANCHOR_HALFSTEPS_0_TO_4: dict[float, float] = {
    0.0: 0.00,
    0.5: 0.12,
    1.0: 0.25,
    1.5: 0.40,
    2.0: 0.55,
    2.5: 0.68,
    3.0: 0.80,
    3.5: 0.90,
    4.0: 1.00,
}

# Integer reference credits at r = 0,1,2,3,4 (used to interpolate other maxima).
_REFERENCE_INTEGER_CREDITS_0_TO_4: list[float] = [0.0, 0.25, 0.55, 0.80, 1.0]

RawScorePolicy = Literal["regenerate", "nearest_half", "ceil_half"]


@dataclass(frozen=True)
class RawScoreValidation:
    ok: bool
    raw: float
    max_raw: float
    nearest_valid: float
    message: str = ""


def ceiling_half_point_on_grid(raw: float, max_raw: float) -> float:
    """
    Map a raw value to the **smallest** valid half-step on ``[0, R]`` that is **>=** the
    clamped raw value (round-up / ceiling on the 0.5 grid), capped at ``R``.

    Valid outputs are ``0, 0.5, 1, …, R``. This matches the grading prompt: non-conforming
    decimals are corrected upward to the next half increment, never above ``max_points``.
    """
    R = _snap_max_raw(max_raw)
    if R <= 0:
        return 0.0
    x = float(raw)
    if x <= 0:
        return 0.0
    if x >= R:
        return R
    k = int(math.ceil(x * 2.0 - 1e-12))
    y = k / 2.0
    return max(0.0, min(R, y))


def snap_half_nearest_display(x: float, max_raw: float) -> float:
    """
    Round to the **nearest** valid half-step on ``[0, R]`` (for aggregated means).

    Differs from :func:`ceiling_half_point_on_grid`, which is for invalid LLM outputs.
    """
    R = _snap_max_raw(max_raw)
    if R <= 0:
        return 0.0
    xc = max(0.0, min(R, float(x)))
    y = round(xc * 2.0) / 2.0
    return max(0.0, min(R, y))


def blended_display_points(
    raw_on_scale: float,
    calibrated_credit: float,
    max_points: float,
) -> float:
    """
    Combine consensus raw rubric level with calibrated credit on the same point scale.

    ``points = raw + g * (M - raw)`` moves from the snapped raw level toward full ``M``
    as ``g`` approaches 1 (``g`` is calibrated credit in ``[0, 1]``).
    """
    M = _snap_max_raw(max_points)
    if M <= 0:
        return 0.0
    r = max(0.0, min(M, float(raw_on_scale)))
    g = max(0.0, min(1.0, float(calibrated_credit)))
    return r + g * (M - r)


def finalize_criterion_display_scores(
    raw_mean: float,
    calibrated_credit: float,
    max_points: float,
) -> tuple[float, float]:
    """
    Return ``(raw_rubric_score_snapped, score_snapped)`` both on the half-point grid in ``[0, M]``.
    """
    M = _snap_max_raw(max_points)
    raw_s = snap_half_nearest_display(raw_mean, M)
    blended = blended_display_points(raw_s, calibrated_credit, M)
    score_s = snap_half_nearest_display(blended, M)
    return raw_s, score_s


def _snap_max_raw(R: float) -> float:
    try:
        r = float(R)
    except (TypeError, ValueError):
        return 0.0
    if r <= 0:
        return 0.0
    # Treat near-integers as integers (rubric max_points are usually ints).
    if abs(r - round(r)) < 1e-6:
        return float(int(round(r)))
    return r


def validate_raw_score_increment(
    raw: float,
    max_raw: float,
    *,
    eps: float = 1e-5,
) -> RawScoreValidation:
    """
    Raw scores must lie on the half-point grid in [0, R]: 0, 0.5, 1, ..., R.
    """
    R = _snap_max_raw(max_raw)
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return RawScoreValidation(
            ok=False, raw=0.0, max_raw=R, nearest_valid=0.0, message="non_numeric_raw"
        )
    if R <= 0:
        return RawScoreValidation(
            ok=abs(x) < eps,
            raw=x,
            max_raw=R,
            nearest_valid=0.0,
            message="" if abs(x) < eps else "non_positive_scale",
        )
    if x < -eps or x > R + eps:
        x_c = max(0.0, min(R, x))
        nv = ceiling_half_point_on_grid(x_c, R)
        return RawScoreValidation(
            ok=False,
            raw=x,
            max_raw=R,
            nearest_valid=nv,
            message="out_of_range",
        )
    k = round(x * 2.0)
    snapped = k / 2.0
    if abs(x - snapped) > eps:
        nv = ceiling_half_point_on_grid(x, R)
        return RawScoreValidation(
            ok=False,
            raw=x,
            max_raw=R,
            nearest_valid=nv,
            message="not_half_step_grid",
        )
    if snapped < -eps or snapped > R + eps:
        nv = max(0.0, min(R, snapped))
        return RawScoreValidation(
            ok=False, raw=x, max_raw=R, nearest_valid=nv, message="snapped_out_of_range"
        )
    return RawScoreValidation(ok=True, raw=snapped, max_raw=R, nearest_valid=snapped)


def nearest_half_point_on_grid(raw: float, max_raw: float) -> float:
    """Snap to a valid half-step using **ceiling** (see :func:`ceiling_half_point_on_grid`)."""
    return ceiling_half_point_on_grid(raw, max_raw)


def _ref_float_on_0_to_4(t: float) -> float:
    """Piecewise linear on integer anchors 0..4 at parameter t in [0,4]."""
    t = max(0.0, min(4.0, float(t)))
    if t <= 0.0:
        return 0.0
    if t >= 4.0:
        return 1.0
    lo = int(math.floor(t))
    hi = min(lo + 1, 4)
    frac = t - lo
    y0 = _REFERENCE_INTEGER_CREDITS_0_TO_4[lo]
    y1 = _REFERENCE_INTEGER_CREDITS_0_TO_4[hi]
    return y0 + frac * (y1 - y0)


def interpolate_anchor_map_for_scale(max_raw: float) -> dict[float, float]:
    """
    Build a half-step grid map for general integer R >= 1.

    * R == 4: exact :data:`DEFAULT_ANCHOR_HALFSTEPS_0_TO_4`.
    * Else: g(r) = ref_float( min(4, r * 4 / R) ) for each half-step r in [0, R].
    """
    R = _snap_max_raw(max_raw)
    if R <= 0:
        return {0.0: 0.0}
    if abs(R - 4.0) < 1e-9:
        return dict(DEFAULT_ANCHOR_HALFSTEPS_0_TO_4)  # copy — do not mutate module default
    out: dict[float, float] = {}
    n_steps = int(round(2.0 * R))
    for k in range(n_steps + 1):
        r = k / 2.0
        if r > R + 1e-9:
            break
        t = min(4.0, r * (4.0 / R))
        out[round(r, 6)] = round(_ref_float_on_0_to_4(t), 6)
    # Enforce endpoints
    out[0.0] = 0.0
    out[round(R, 6)] = 1.0
    return out


def get_anchor_map_for_criterion(rubric_row: dict[str, Any] | None) -> dict[float, float]:
    """
    Return mapping raw_level -> calibrated_credit for this rubric row.

    Optional row keys:
    * ``calibration_anchor_table``: dict[str|float, float] raw -> credit
    * ``calibration_max_raw`` / ``max_points``: scale top R
    """
    row = rubric_row or {}
    custom = row.get("calibration_anchor_table") or row.get("anchor_table")
    if isinstance(custom, dict) and custom:
        out: dict[float, float] = {}
        for kr, vr in custom.items():
            try:
                rk = float(kr)
                out[round(rk, 6)] = float(vr)
            except (TypeError, ValueError):
                continue
        if out:
            return out
    try:
        R = float(
            row.get("calibration_max_raw")
            or row.get("max_points")
            or row.get("max_score")
            or 0
        )
    except (TypeError, ValueError):
        R = 0.0
    return interpolate_anchor_map_for_scale(R)


def map_raw_score_to_calibrated_credit(
    raw: float,
    anchor_map: dict[float, float],
    *,
    eps: float = 1e-5,
) -> float:
    """
    Monotone lookup: exact key match, else linear interpolate between bracketing
    half-step keys in anchor_map.
    """
    if not anchor_map:
        return 0.0
    x = float(raw)
    keys = sorted(anchor_map.keys())
    if x <= keys[0] + eps:
        return float(anchor_map[keys[0]])
    if x >= keys[-1] - eps:
        return float(anchor_map[keys[-1]])
    # find bracket
    lo_i = 0
    for i in range(len(keys) - 1):
        if keys[i] <= x + eps and keys[i + 1] >= x - eps:
            lo_i = i
            break
    x0, x1 = keys[lo_i], keys[lo_i + 1]
    if abs(x1 - x0) < eps:
        return float(anchor_map[x0])
    t = (x - x0) / (x1 - x0)
    y0, y1 = float(anchor_map[x0]), float(anchor_map[x1])
    return y0 + t * (y1 - y0)


def compute_weighted_question_score(
    rows: list[tuple[float, float]],
    *,
    scale_to_100: bool = True,
) -> tuple[float, list[dict[str, float]]]:
    """
    S = (sum_j w_j * g_j) / (sum_j w_j); optionally * 100.

    ``rows`` are (weight_wj, calibrated_credit_gj).
    Returns (score, per-row contribution audit).
    """
    audit: list[dict[str, float]] = []
    w_sum = 0.0
    acc = 0.0
    for w, g in rows:
        wf = max(0.0, float(w))
        gf = max(0.0, min(1.0, float(g)))
        w_sum += wf
        contrib = wf * gf
        acc += contrib
        audit.append(
            {
                "weight": wf,
                "calibrated_credit": gf,
                "weighted_contribution": contrib,
            }
        )
    if w_sum <= 0:
        return (0.0, audit)
    s = acc / w_sum
    if scale_to_100:
        s *= 100.0
    return (float(s), audit)


def compute_mean_calibrated_question_score(
    calibrated_credits: list[float],
    *,
    scale_to_100: bool = True,
) -> tuple[float, list[dict[str, float]]]:
    """Unweighted mean of calibrated credits in ``[0, 1]``, optionally scaled to 0–100."""
    if not calibrated_credits:
        return 0.0, []
    g_vals = [max(0.0, min(1.0, float(g))) for g in calibrated_credits]
    m = sum(g_vals) / len(g_vals)
    s = m * 100.0 if scale_to_100 else m
    audit = [{"calibrated_credit": g} for g in g_vals]
    return float(s), audit


def anchor_map_monotone_increasing(anchor_map: dict[float, float], *, eps: float = 1e-6) -> bool:
    keys = sorted(anchor_map.keys())
    last = -1.0
    for k in keys:
        v = float(anchor_map[k])
        if v < last - eps:
            return False
        last = v
    return True


def format_anchor_map_for_log(anchor_map: dict[float, float]) -> str:
    keys = sorted(anchor_map.keys())
    parts = [f"{k:g}->{anchor_map[k]:.4f}" for k in keys]
    return "{" + ", ".join(parts) + "}"
