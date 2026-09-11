import pandas as pd
from dashboard import charts, risk


def test_both_palettes_define_every_band():
    for theme in ("light", "dark"):
        palette = charts.palette(theme)
        assert set(palette["risk"]) == set(charts.BAND_ORDER)


def test_light_and_dark_are_selected_not_shared():
    """Dark steps are chosen for the dark surface, not reused from light.

    The validated ordinal ramps differ per surface; if a future edit collapses them to one
    dict, the dark chart silently ships colours validated against the wrong background.
    """
    light, dark = charts.palette("light"), charts.palette("dark")
    assert light["surface"] != dark["surface"]
    assert light["risk"] != dark["risk"]
    assert light["series"] != dark["series"]


def test_unknown_theme_falls_back_to_light():
    assert charts.palette("solarized") == charts.palette("light")


def _banded(snapshot):
    return risk.with_risk_bands(snapshot, high_threshold=0.7, medium_threshold=0.4)


def test_charts_build_without_a_renderer(snapshot):
    """Altair specs must compile — a bad encoding otherwise only surfaces in the browser."""
    banded = _banded(snapshot)
    palette = charts.palette("light")

    assert charts.band_composition(banded, palette).to_dict()
    assert charts.high_risk_by_segment(banded, "membership_tier", "tier", 0.7, palette).to_dict()

    trend = pd.DataFrame(
        {
            "snapshot_date": pd.to_datetime(["2026-09-08", "2026-09-09", "2026-09-10"]),
            "predicted_churn_rate": [0.21, 0.23, 0.22],
            "customers_scored": [1000, 1010, 1020],
            "high_risk_customers": [80, 92, 88],
            "model_versions_used": [1, 1, 1],
        }
    )
    assert charts.risk_trend(trend, palette).to_dict()


def test_segment_chart_reports_share_not_raw_count(snapshot):
    """The largest segment must not automatically top the chart."""
    banded = _banded(snapshot)
    spec = charts.high_risk_by_segment(banded, "region", "region", 0.7, charts.palette("light"))
    encoding = spec.to_dict()["layer"][0]["encoding"]
    assert encoding["x"]["field"] == "share"
    assert encoding["x"]["axis"]["format"] == ".0%"
