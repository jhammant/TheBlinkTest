"""Psychopathy risk assessment based on blink rate analysis.

Research background:
- Normal spontaneous blink rate: 15-20 blinks/min (Bentivoglio et al., 1997)
- Psychopathy correlates with reduced spontaneous blink rate (Lykken, 1957;
  Forth & Hare, 1989; Baskin-Sommers et al., 2013)
- Lower blink rates are associated with reduced emotional reactivity and
  diminished startle response, both hallmarks of psychopathic traits
- Blink rate alone is NOT diagnostic — many factors affect blink rate including
  focus, fatigue, dry eyes, contact lenses, medication, screen time

IMPORTANT DISCLAIMER: This is an experimental analysis tool for research and
educational purposes only. Blink rate is ONE of many potential indicators and
cannot diagnose psychopathy. Do not use this to make judgments about individuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from blinkcounter.core.models import Person, AnalysisResult, BlinkClassification


@dataclass
class BlinkAssessment:
    """Assessment result for a single person."""

    person_label: str
    blinks_per_minute: float
    classification: BlinkClassification
    analyzable_seconds: float
    confidence: str  # "High", "Medium", "Low" based on analyzable time
    risk_score: int  # 0-100, higher = more anomalous blink rate
    risk_level: str  # "Minimal", "Low", "Moderate", "Elevated", "High"
    interpretation: str  # Human-readable explanation
    caveats: list[str]  # Factors that may affect reliability
    trust_score: float = 1.0  # 0.0-1.0 — how much to trust this result
    trusted: bool = True  # False = result should not be used for conclusions


def assess_person(person: Person) -> BlinkAssessment:
    """Generate a blink rate assessment for a person.

    Args:
        person: Person with blink detection results.

    Returns:
        BlinkAssessment with risk scoring and interpretation.
    """
    rate = person.blinks_per_minute
    classification = person.classification
    analyzable = person.analyzable_duration if person.analyzable_duration >= 1.0 else person.total_visible_duration

    # Confidence based on analyzable time
    if analyzable >= 120:
        confidence = "High"
    elif analyzable >= 60:
        confidence = "Medium"
    elif analyzable >= 20:
        confidence = "Low"
    else:
        confidence = "Very Low"

    # Trust score: 0.0-1.0 based on data quality
    trust_score = _calculate_trust(person, analyzable)
    trusted = trust_score >= 0.5  # Below 0.5 = result should not be used

    # Risk score: 0-100 based on deviation from normal range (15-20/min)
    # Lower blink rate = higher risk score
    if rate <= 0:
        risk_score = 95
    elif rate < 5:
        risk_score = 85
    elif rate < 8:
        risk_score = 70
    elif rate < 10:
        risk_score = 55
    elif rate < 12:
        risk_score = 40
    elif rate < 15:
        risk_score = 25
    elif rate < 20:
        risk_score = 10  # Normal range
    elif rate < 25:
        risk_score = 5   # Slightly above normal (stress/anxiety)
    else:
        risk_score = 15  # High blink rate — could be stress, not psychopathy

    # Adjust risk score by confidence and trust
    if confidence == "Very Low":
        risk_score = min(risk_score, 30)  # Cap at 30 if insufficient data
    elif confidence == "Low":
        risk_score = int(risk_score * 0.8)
    if not trusted:
        risk_score = min(risk_score, 20)  # Cap if we don't trust the data

    # Risk level
    if risk_score >= 75:
        risk_level = "High"
    elif risk_score >= 50:
        risk_level = "Elevated"
    elif risk_score >= 30:
        risk_level = "Moderate"
    elif risk_score >= 15:
        risk_level = "Low"
    else:
        risk_level = "Minimal"

    # Interpretation
    interpretation = _build_interpretation(rate, classification, risk_level, confidence, analyzable)

    # Caveats
    caveats = _build_caveats(rate, confidence, analyzable, person)

    # Add trust warning to caveats if not trusted
    if not trusted:
        caveats.insert(0, f"LOW TRUST ({trust_score:.0%}) — insufficient or inconsistent data, result should not be relied upon")

    return BlinkAssessment(
        person_label=person.label,
        blinks_per_minute=rate,
        classification=classification,
        analyzable_seconds=analyzable,
        confidence=confidence,
        risk_score=risk_score,
        risk_level=risk_level,
        interpretation=interpretation,
        caveats=caveats,
        trust_score=trust_score,
        trusted=trusted,
    )


def _calculate_trust(person: Person, analyzable: float) -> float:
    """Calculate trust score (0.0-1.0) for a person's blink rate measurement.

    Trust is reduced by:
    - Low analyzable time (< 60s = very unreliable)
    - Low ratio of analyzable to visible time (face often at bad angle)
    - Very high blink rate (> 30/min likely has false positives)
    - Very high blink count with low analyzable time (suspicious)
    """
    trust = 1.0

    # Analyzable time factor: need at least 60s for any trust
    if analyzable < 10:
        trust *= 0.1
    elif analyzable < 30:
        trust *= 0.3
    elif analyzable < 60:
        trust *= 0.6
    elif analyzable < 120:
        trust *= 0.8

    # Analyzable ratio: if only 30% of visible time was analyzable, data is spotty
    if person.total_visible_duration > 0:
        ratio = analyzable / person.total_visible_duration
        if ratio < 0.3:
            trust *= 0.4
        elif ratio < 0.5:
            trust *= 0.7

    # Suspiciously high rate likely means false positives
    rate = person.blinks_per_minute
    if rate > 40:
        trust *= 0.3  # Very likely false positives
    elif rate > 30:
        trust *= 0.6  # Probably some false positives

    return round(trust, 2)


def calculate_cross_video_trust(per_video_rates: list[float]) -> tuple[float, str]:
    """Calculate trust for cross-video results based on consistency.

    Args:
        per_video_rates: List of blink rates from different videos.

    Returns:
        (trust_score, explanation) tuple.
    """
    if len(per_video_rates) < 2:
        return 0.4, "Only 1 video — insufficient for reliable assessment"

    import statistics
    mean_rate = statistics.mean(per_video_rates)
    if mean_rate < 0.1:
        return 0.3, "Near-zero rate across videos — may be detection issue"

    stdev = statistics.stdev(per_video_rates) if len(per_video_rates) > 1 else 0
    cv = stdev / mean_rate if mean_rate > 0 else 0  # Coefficient of variation

    if cv > 0.5:
        return 0.2, f"Very high variance between videos (CV={cv:.0%}) — results unreliable"
    elif cv > 0.35:
        return 0.4, f"High variance between videos (CV={cv:.0%}) — results questionable"
    elif cv > 0.2:
        return 0.7, f"Moderate variance between videos (CV={cv:.0%})"
    else:
        return 0.9, f"Consistent across videos (CV={cv:.0%}) — reliable"


def _build_interpretation(
    rate: float,
    classification: BlinkClassification,
    risk_level: str,
    confidence: str,
    analyzable: float,
) -> str:
    """Build human-readable interpretation."""
    if rate <= 0:
        return (
            "No blinks detected. This is extremely unusual and could indicate "
            "a deliberate effort to suppress blinking (as seen in acting roles), "
            "or a detection issue. If genuine, this is a significant anomaly."
        )
    elif rate < 8:
        return (
            f"Blink rate of {rate:.1f}/min is significantly below the normal range "
            f"(15-20/min). Research associates very low blink rates with reduced "
            f"emotional reactivity, a trait linked to psychopathic characteristics. "
            f"However, this could also indicate intense focus, fatigue, or "
            f"environmental factors."
        )
    elif rate < 12:
        return (
            f"Blink rate of {rate:.1f}/min is below normal (15-20/min). "
            f"Mildly reduced blink rates can occur during focused tasks, "
            f"reading, or screen use. While below average, this alone is not "
            f"strongly indicative of any particular trait."
        )
    elif rate < 15:
        return (
            f"Blink rate of {rate:.1f}/min is at the lower end of normal. "
            f"This is within the range seen in focused but otherwise typical "
            f"individuals. No significant anomaly detected."
        )
    elif rate < 20:
        return (
            f"Blink rate of {rate:.1f}/min is within the normal range (15-20/min). "
            f"This is a typical, healthy blink rate with no anomalous patterns."
        )
    elif rate < 30:
        return (
            f"Blink rate of {rate:.1f}/min is above normal (15-20/min). "
            f"Elevated blink rates are associated with stress, anxiety, fatigue, "
            f"or stimulant use. This is the opposite of what's seen in psychopathy."
        )
    else:
        return (
            f"Blink rate of {rate:.1f}/min is significantly elevated. This may "
            f"indicate high stress, anxiety, or could be a measurement artifact "
            f"from head movements or video quality issues."
        )


def _build_caveats(
    rate: float,
    confidence: str,
    analyzable: float,
    person: Person,
) -> list[str]:
    """Build list of caveats that affect reliability."""
    caveats = []

    if confidence in ("Low", "Very Low"):
        caveats.append(
            f"Limited analyzable time ({analyzable:.0f}s) — results may not be representative"
        )

    if person.total_visible_duration > 0:
        pct = (person.analyzable_duration / person.total_visible_duration) * 100
        if pct < 50:
            caveats.append(
                f"Only {pct:.0f}% of visible time was analyzable (face angle/quality issues)"
            )

    if rate > 30:
        caveats.append(
            "Very high rate may indicate detection false positives from head movement"
        )

    caveats.append("Blink rate is affected by: lighting, screen use, fatigue, medication, dry eyes")
    caveats.append("This analysis cannot diagnose any condition — consult qualified professionals")

    return caveats


def format_assessment(assessment: BlinkAssessment) -> str:
    """Format assessment as a readable report string."""
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  BLINK RATE PSYCHOPATHY RISK ASSESSMENT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Subject:        {assessment.person_label}")
    lines.append(f"  Blink Rate:     {assessment.blinks_per_minute:.1f} blinks/min")
    lines.append(f"  Classification: {assessment.classification.value}")
    lines.append(f"  Analyzed:       {assessment.analyzable_seconds:.0f}s of video")
    lines.append(f"  Confidence:     {assessment.confidence}")
    lines.append("")
    lines.append("-" * 60)

    # Risk meter visual
    bar_width = 40
    filled = int(assessment.risk_score / 100 * bar_width)
    bar = "#" * filled + "-" * (bar_width - filled)

    lines.append("")
    lines.append(f"  RISK SCORE:     {assessment.risk_score}/100")
    lines.append(f"  RISK LEVEL:     {assessment.risk_level}")
    lines.append(f"  [{bar}]")
    lines.append(f"   0    20    40    60    80   100")
    lines.append(f"   Minimal  Low  Moderate Elevated High")
    lines.append("")

    # Normal range reference
    lines.append("-" * 60)
    lines.append("  REFERENCE RANGES:")
    lines.append("    < 10/min  = Very Low  (potential psychopathy indicator)")
    lines.append("   10-15/min  = Low       (below average)")
    lines.append("   15-20/min  = Normal    (healthy range)")
    lines.append("    > 20/min  = High      (stress/anxiety indicator)")
    lines.append("")

    # Interpretation
    lines.append("-" * 60)
    lines.append("  INTERPRETATION:")
    # Word wrap interpretation at 56 chars
    words = assessment.interpretation.split()
    line = "    "
    for word in words:
        if len(line) + len(word) + 1 > 58:
            lines.append(line)
            line = "    " + word
        else:
            line += " " + word if line.strip() else "    " + word
    if line.strip():
        lines.append(line)
    lines.append("")

    # Caveats
    if assessment.caveats:
        lines.append("-" * 60)
        lines.append("  CAVEATS:")
        for caveat in assessment.caveats:
            lines.append(f"    * {caveat}")
        lines.append("")

    # Additional behavioral indicators
    lines.append("-" * 60)
    lines.append("  ADDITIONAL BEHAVIORAL INDICATORS:")
    indicators = _build_behavioral_indicators(assessment.blinks_per_minute, assessment.analyzable_seconds)
    for indicator in indicators:
        lines.append(f"    {indicator['icon']} {indicator['label']}: {indicator['value']}")
        lines.append(f"      {indicator['detail']}")
    lines.append("")

    lines.append("=" * 60)
    lines.append("  DISCLAIMER: This is an experimental research tool.")
    lines.append("  Blink rate alone CANNOT diagnose psychopathy.")
    lines.append("  Many factors affect blink rate. Do not use this")
    lines.append("  to make judgments about individuals.")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def _build_behavioral_indicators(rate: float, analyzable: float) -> list[dict]:
    """Build additional behavioral indicators from blink rate."""
    indicators = []

    # Stress/Anxiety indicator (high blink rate)
    if rate > 25:
        stress = "HIGH"
        detail = "Elevated blink rate often correlates with stress, anxiety, or cognitive load"
    elif rate > 20:
        stress = "MODERATE"
        detail = "Slightly elevated — may indicate mild stress or active engagement"
    elif rate >= 15:
        stress = "NORMAL"
        detail = "Blink rate suggests relaxed, comfortable state"
    elif rate >= 10:
        stress = "LOW"
        detail = "Below-average stress response — could indicate calm focus or emotional flatness"
    else:
        stress = "VERY LOW"
        detail = "Unusually low emotional arousal detected"
    indicators.append({"icon": "~", "label": "Stress/Anxiety", "value": stress, "detail": detail})

    # Focus/Attention indicator (lower rate = more focused)
    if rate < 10:
        focus = "INTENSE"
        detail = "Very low blink rate suggests deep concentration or hyperfocus"
    elif rate < 15:
        focus = "HIGH"
        detail = "Reduced blinking often seen during focused tasks (reading, presenting)"
    elif rate < 20:
        focus = "MODERATE"
        detail = "Normal attention level — engaged but not intensely focused"
    else:
        focus = "DIFFUSE"
        detail = "Higher blink rate may indicate divided attention or cognitive fatigue"
    indicators.append({"icon": "~", "label": "Focus Level", "value": focus, "detail": detail})

    # Deception indicator (research shows increased blink rate when lying)
    if rate > 25:
        deception = "ELEVATED"
        detail = "Research shows blink rate increases during deception (Leal & Vrij, 2008)"
    elif rate > 20:
        deception = "SLIGHT"
        detail = "Marginally elevated — insufficient alone for deception assessment"
    elif rate >= 12:
        deception = "BASELINE"
        detail = "No deception indicators from blink rate (normal range)"
    else:
        deception = "N/A"
        detail = "Low blink rate is not associated with deception"
    indicators.append({"icon": "~", "label": "Deception Signal", "value": deception, "detail": detail})

    # Fatigue indicator
    if rate > 20:
        fatigue = "POSSIBLE"
        detail = "Elevated blink rate can indicate drowsiness or eye fatigue"
    elif rate >= 15:
        fatigue = "UNLIKELY"
        detail = "Normal blink rate suggests alert, rested state"
    else:
        fatigue = "UNLIKELY"
        detail = "Low blink rate typically indicates alertness (or emotional flatness)"
    indicators.append({"icon": "~", "label": "Fatigue", "value": fatigue, "detail": detail})

    # Emotional engagement
    if rate < 8:
        emotion = "LOW"
        detail = "Very low blink rate associated with reduced emotional processing"
    elif rate < 15:
        emotion = "MODERATE"
        detail = "Below-average emotional reactivity indicated"
    elif rate < 22:
        emotion = "NORMAL"
        detail = "Typical emotional engagement patterns"
    else:
        emotion = "HEIGHTENED"
        detail = "Elevated blinking may reflect strong emotional response"
    indicators.append({"icon": "~", "label": "Emotional Engagement", "value": emotion, "detail": detail})

    # Confidence assessment based on data quality
    if analyzable >= 120:
        data_quality = "STRONG"
        detail = f"Based on {analyzable:.0f}s of analyzable video — reliable sample"
    elif analyzable >= 60:
        data_quality = "ADEQUATE"
        detail = f"Based on {analyzable:.0f}s — reasonable but more data improves accuracy"
    else:
        data_quality = "LIMITED"
        detail = f"Only {analyzable:.0f}s analyzed — results should be treated with caution"
    indicators.append({"icon": "~", "label": "Data Quality", "value": data_quality, "detail": detail})

    return indicators


def format_multi_video_assessment(
    person_label: str,
    per_video_rates: dict[str, float],
    aggregate_rate: float,
    total_analyzable: float,
) -> str:
    """Format assessment for a person analyzed across multiple videos."""
    from blinkcounter.core.models import Person, BlinkEvent

    # Create a synthetic person for the aggregate assessment
    synthetic = Person(
        id="aggregate",
        label=person_label,
        total_visible_duration=total_analyzable,
        analyzable_duration=total_analyzable,
        blink_events=[
            BlinkEvent(timestamp=0, person_id="aggregate", ear_value=0.15)
        ] * int(aggregate_rate * total_analyzable / 60),
    )

    assessment = assess_person(synthetic)

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("  CROSS-VIDEO BLINK RATE ANALYSIS")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Subject appears in {len(per_video_rates)} video(s)")
    lines.append("")

    for video, rate in per_video_rates.items():
        # Shorten video name
        short = video.split("/")[-1][:45] if "/" in video else video[:45]
        from blinkcounter.core.models import BlinkClassification
        cls = BlinkClassification.from_rate(rate)
        lines.append(f"    {short}")
        lines.append(f"      Rate: {rate:.1f}/min [{cls.value}]")

    lines.append("")
    lines.append(f"  AGGREGATE RATE: {aggregate_rate:.1f}/min")
    lines.append(f"  Total analyzed: {total_analyzable:.0f}s across all videos")
    lines.append("")

    # Add the full assessment
    lines.append(format_assessment(assessment))

    return "\n".join(lines)
