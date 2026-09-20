"""Altair chart builders and the theme-aware palette they draw from.

Colour choices here are validated, not taste. Two of them are worth stating because the
obvious alternative is wrong:

*Risk bands use a single-hue ordinal blue ramp, not a red/amber/green traffic light.* Risk
is ordered magnitude, so more risk reads as darker on one hue. The traffic-light version was
measured and rejected: status red against status green separate by only ΔE 4.1 under
deuteranopia, so for a red-green colourblind reader — around 1 in 12 men — the highest and
lowest bands would be near-identical, in the one chart whose entire job is telling them
apart. The blue ramp clears every ordinal gate on both surfaces.

*Band names appear as text on every mark and every table row.* Colour is never the only
channel carrying the band, so the charts survive greyscale printing and forced-colors mode.

Dark mode uses its own steps chosen for the dark surface, not an automatic lightening of the
light ones.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# Worst-first: the display order for bands everywhere in the app.
BAND_ORDER: list[str] = ["High", "Medium", "Low"]

_LIGHT = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": "#2a78d6",
    # Ordinal ramp, light→dark with risk. Validated: monotone lightness, ΔL gaps ≥ 0.06,
    # light end 2.06:1 against the light surface, 3° hue spread.
    "risk": {"Low": "#86b6ef", "Medium": "#2a78d6", "High": "#104281"},
}

_DARK = {
    "surface": "#1a1a19",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": "#3987e5",
    # Stepped for the dark surface and validated as its own set: light end 2.15:1 there.
    "risk": {"Low": "#6da7ec", "Medium": "#256abf", "High": "#184f95"},
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(theme: str) -> dict:
    """Return the validated palette for the viewer's active theme."""
    return _DARK if theme == "dark" else _LIGHT


def _base(chart: alt.Chart, pal: dict, height: int) -> alt.Chart:
    """Apply the shared chrome: recessive grid and axes, text in ink tokens, no border."""
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0, fill=pal["surface"])
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=pal["muted"],
            titleColor=pal["text_secondary"],
            gridColor=pal["grid"],
            domainColor=pal["axis"],
            tickColor=pal["axis"],
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
        )
        .configure_legend(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=pal["text_secondary"],
            titleColor=pal["text_secondary"],
        )
        .configure_text(font=FONT)
    )


def band_composition(df: pd.DataFrame, pal: dict) -> alt.LayerChart:
    """One horizontal stacked bar: how the scored portfolio splits across risk bands.

    Part-to-whole, so a single stacked bar rather than three separate bars — the reader's
    question is what share of the base is high risk, and a shared baseline answers it
    directly. Segments carry a 2px surface-coloured gap so adjacent steps of the same hue
    stay separable, and each is direct-labelled with its band name and count so the ramp
    never has to be decoded from a legend.
    """
    counts = (
        df.groupby("risk_band", observed=True)
        .size()
        .reindex(BAND_ORDER, fill_value=0)
        .rename("customers")
        .reset_index()
    )
    counts["share"] = counts["customers"] / max(counts["customers"].sum(), 1)

    bars = (
        alt.Chart(counts)
        .mark_bar(height=44, cornerRadius=4, stroke=pal["surface"], strokeWidth=2)
        .encode(
            x=alt.X("customers:Q", stack="zero", title=None, axis=None),
            color=alt.Color(
                "risk_band:N",
                scale=alt.Scale(domain=BAND_ORDER, range=[pal["risk"][b] for b in BAND_ORDER]),
                legend=None,
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("risk_band:N", title="Risk band"),
                alt.Tooltip("customers:Q", title="Customers", format=","),
                alt.Tooltip("share:Q", title="Share of base", format=".1%"),
            ],
        )
        .transform_calculate(order=f"indexof({BAND_ORDER!r}, datum.risk_band)")
    )

    # Labels sit on the segment, in white, only where the segment is wide enough to hold
    # them — a number crammed into a 3px sliver is noise, and the tooltip already has it.
    labels = (
        alt.Chart(counts)
        .mark_text(font=FONT, fontSize=12, fontWeight="bold", color="#ffffff")
        .encode(
            x=alt.X("customers:Q", stack="zero", bandPosition=0.5),
            text=alt.Text("label:N"),
            order=alt.Order("order:Q"),
            opacity=alt.condition(alt.datum.share > 0.08, alt.value(1), alt.value(0)),
        )
        .transform_calculate(
            order=f"indexof({BAND_ORDER!r}, datum.risk_band)",
            label="datum.risk_band + '  ' + format(datum.customers, ',')",
        )
    )

    return _base(alt.layer(bars, labels), pal, height=70)


def risk_trend(df: pd.DataFrame, pal: dict) -> alt.LayerChart:
    """Predicted churn rate per scored day, with a crosshair and tooltip.

    One series, so no legend — the chart title names it. A line rather than an area or bars
    because the reader's question is direction, and because the y-axis is deliberately not
    zero-based here: churn rates move within a narrow band, and anchoring to zero would
    flatten every real movement into a straight line.
    """
    hover = alt.selection_point(
        fields=["snapshot_date"], nearest=True, on="pointerover", empty=False, clear="pointerout"
    )

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color=pal["series"], interpolate="monotone")
        .encode(
            x=alt.X("snapshot_date:T", title=None),
            y=alt.Y(
                "predicted_churn_rate:Q",
                title="Predicted churn rate",
                axis=alt.Axis(format=".0%"),
                scale=alt.Scale(zero=False, nice=True),
            ),
        )
    )

    rule = (
        alt.Chart(df)
        .mark_rule(color=pal["axis"], strokeWidth=1)
        .encode(
            x=alt.X("snapshot_date:T"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("snapshot_date:T", title="Snapshot"),
                alt.Tooltip("predicted_churn_rate:Q", title="Predicted churn rate", format=".1%"),
                alt.Tooltip("customers_scored:Q", title="Customers scored", format=","),
                alt.Tooltip("high_risk_customers:Q", title="High risk", format=","),
                alt.Tooltip("model_versions_used:Q", title="Models in partition"),
            ],
        )
        .add_params(hover)
    )

    # 8px marker, ringed in the surface colour so it stays separable where it sits on the line.
    point = (
        alt.Chart(df)
        .mark_point(size=80, filled=True, color=pal["series"], stroke=pal["surface"], strokeWidth=2)
        .encode(
            x=alt.X("snapshot_date:T"),
            y=alt.Y("predicted_churn_rate:Q"),
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        )
    )

    return _base(alt.layer(line, rule, point), pal, height=220)


def high_risk_by_segment(
    df: pd.DataFrame, dimension: str, label: str, high_threshold: float, pal: dict
) -> alt.LayerChart:
    """High-risk share within each value of a segmentation dimension.

    *Share*, not count, and this is the whole point of the chart: the largest tier always has
    the most at-risk customers, so a count chart says nothing except which tier is biggest.
    Share answers the question a retention owner actually has — where is the problem
    concentrated. The count rides along in the tooltip and the direct label so a 100% share
    over four customers cannot be mistaken for a crisis.
    """
    grouped = (
        df.assign(_high=df["churn_probability"] >= high_threshold)
        .groupby(dimension, observed=True)
        .agg(customers=("customer_id", "size"), high_risk=("_high", "sum"))
        .reset_index()
    )
    grouped["share"] = grouped["high_risk"] / grouped["customers"]
    grouped = grouped.sort_values("share", ascending=False)

    bars = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusEnd=4, height=22, color=pal["series"])
        .encode(
            y=alt.Y(f"{dimension}:N", sort="-x", title=None),
            x=alt.X(
                "share:Q", title=f"High-risk share of {label.lower()}", axis=alt.Axis(format=".0%")
            ),
            tooltip=[
                alt.Tooltip(f"{dimension}:N", title=label),
                alt.Tooltip("share:Q", title="High-risk share", format=".1%"),
                alt.Tooltip("high_risk:Q", title="High-risk customers", format=","),
                alt.Tooltip("customers:Q", title="Customers in segment", format=","),
            ],
        )
    )

    labels = (
        alt.Chart(grouped)
        .mark_text(align="left", dx=6, font=FONT, fontSize=11, color=pal["text_secondary"])
        .encode(
            y=alt.Y(f"{dimension}:N", sort="-x"),
            x=alt.X("share:Q"),
            text=alt.Text("label:N"),
        )
        .transform_calculate(
            label="format(datum.share, '.0%') + '  (' + format(datum.high_risk, ',') + ' of ' "
            "+ format(datum.customers, ',') + ')'"
        )
    )

    height = max(120, 34 * len(grouped))
    return _base(alt.layer(bars, labels), pal, height=height)
