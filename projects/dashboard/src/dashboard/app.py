"""The churn-risk dashboard: a retention worklist for business users.

Read by people who do not know what a model is. Three rules follow from that and shape
everything below:

1. *The page says how old it is, always.* A scoring dashboard whose data silently froze
   looks exactly like a quiet week. The scored snapshot date and its age sit in the header,
   and a snapshot older than the pipeline's daily cadence is called out rather than shown
   as if it were today's.
2. *Predicted is never presented as observed.* The mart carries no `churned` label, and
   every probability is worded as a prediction.
3. *Indicators are not explanations.* The drill-down contrasts a customer against the
   cohort scored the same night. That is a correlation, not the model's attribution for
   that customer, and the panel says so in as many words — see dashboard.risk.

The whole snapshot is fetched once per cache window and every filter below runs against
that in-memory frame, so no interaction on this page costs a BigQuery query.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from dashboard import charts, risk
from dashboard.data import build_client, fetch_risk_snapshot, fetch_risk_trend
from dashboard.identity import viewer_email
from dashboard.settings import get_settings

st.set_page_config(page_title="Customer Churn Risk", page_icon="📉", layout="wide")

settings = get_settings()


@st.cache_resource
def _client():
    """Return the process-wide BigQuery client (one per container, not one per rerun)."""
    return build_client(settings)


@st.cache_data(ttl=settings.cache_ttl_seconds, show_spinner="Loading the latest scoring run…")
def _snapshot() -> pd.DataFrame:
    """Return the newest scored snapshot with risk bands attached."""
    frame = fetch_risk_snapshot(_client(), settings)
    if frame.empty:
        return frame
    return risk.with_risk_bands(frame, settings.high_risk_threshold, settings.medium_risk_threshold)


@st.cache_data(ttl=settings.cache_ttl_seconds, show_spinner=False)
def _trend() -> pd.DataFrame:
    """Return the per-day scoring history."""
    return fetch_risk_trend(_client(), settings)


# Vertex AI writes the champion into ml.predictions as its full resource name —
# projects/{project}/locations/{region}/models/{id}, optionally suffixed with @{version}
# (post_training.register logs Model.resource_name). The console keys its Model Registry page
# on the last three of those, and on the literal version segment "default" when the resource
# name carries no @version — the same mapping the Vertex AI SDK uses for its own "View Model"
# links (google/cloud/aiplatform/utils/_ipython_utils.py).
_MODEL_RESOURCE = re.compile(
    r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)"
    r"/models/(?P<model>[^/@]+)(?:@(?P<version>[^/@]+))?$"
)


def model_registry_url(resource_name: str) -> str | None:
    """Return the Model Registry console URL for a Vertex AI model resource name.

    None when the value is not one. `model_version` is a free-text mart column, so a
    hand-written or legacy value has to degrade to unlinked text rather than to a link that
    lands the reader on a console error page.
    """
    match = _MODEL_RESOURCE.match(resource_name.strip())
    if match is None:
        return None
    return (
        "https://console.cloud.google.com/vertex-ai/models/locations/"
        f"{match['location']}/models/{match['model']}"
        f"/versions/{match['version'] or 'default'}?project={match['project']}"
    )


def _active_palette() -> dict:
    """Return the validated palette matching the viewer's light/dark preference."""
    theme = getattr(getattr(st, "context", None), "theme", None)
    return charts.palette(getattr(theme, "type", "light") or "light")


def _header(snapshot: pd.DataFrame) -> None:
    """Render the title, the freshness statement, and the signed-in viewer."""
    left, right = st.columns([3, 1])
    with left:
        st.title("Customer churn risk")

    scored_on = pd.to_datetime(snapshot["snapshot_date"].iloc[0]).date()
    age_days = (pd.Timestamp.now("UTC").date() - scored_on).days

    with right:
        email = viewer_email(dict(st.context.headers)) if hasattr(st, "context") else None
        st.caption(f"Signed in as **{email}**" if email else "Signed in as an unauthenticated user")
        # Linked to the Model Registry entry that produced these scores, so "which model
        # is this?" is one click rather than a hunt through the console.
        resource_name = str(snapshot["model_version"].iloc[0])
        display_name = resource_name.split("/")[-1]
        console_url = model_registry_url(resource_name)
        st.caption(
            f"Model: [`{display_name}`]({console_url})"
            if console_url
            else f"Model: `{display_name}`"
        )

    # Staleness is stated, not implied. The pipeline scores nightly, so anything past a day
    # means the run did not land and the numbers below describe a base that has since moved.
    if age_days > 1:
        st.warning(
            f"These scores are from **{scored_on:%d %b %Y}** — {age_days} days old. "
            "The nightly scoring run has not completed since then, so treat this as a "
            "historical view rather than today's picture.",
            icon="⚠️",
        )
    else:
        st.caption(
            f"Scored on **{scored_on:%d %b %Y}** from that day's customer snapshot. "
            "Every figure on this page is a *prediction*, not an observed outcome."
        )


def _kpi_row(summary: dict, trend: pd.DataFrame) -> None:
    """Render the four headline numbers, with day-over-day deltas where history allows."""
    previous = None
    if len(trend) >= 2:
        previous = trend.iloc[-2]

    columns = st.columns(4)
    columns[0].metric("Customers scored", f"{summary['customers_scored']:,}")

    delta = None
    if previous is not None:
        delta = int(summary["high_risk_customers"] - previous["high_risk_customers"])
    columns[1].metric(
        f"High risk (≥ {settings.high_risk_threshold:.0%})",
        f"{summary['high_risk_customers']:,}",
        delta=f"{delta:+,} vs previous run" if delta is not None else None,
        # More customers becoming high risk is bad news, so the usual "up is green"
        # colouring has to be inverted or the metric reads as an improvement.
        delta_color="inverse",
    )
    columns[2].metric("Share of base at high risk", f"{summary['high_risk_share']:.1%}")
    columns[3].metric(
        "Monthly value at high risk",
        f"${summary['monthly_value_at_risk']:,.0f}",
        help=(
            "Monthly recurring value carried by the high-risk group — the upper bound on "
            "what is exposed if none of them are retained. It is not a forecast of loss: "
            "most high-risk customers do not churn."
        ),
    )


def _filters(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Render the filter row and return the filtered frame."""
    columns = st.columns([1.2, 1.2, 1.2, 1.4])

    bands = columns[0].multiselect(
        "Risk band", charts.BAND_ORDER, default=["High"], help="Worst first."
    )
    tiers = columns[1].multiselect(
        "Membership tier", sorted(snapshot["membership_tier"].dropna().unique())
    )
    regions = columns[2].multiselect("Region", sorted(snapshot["region"].dropna().unique()))
    search = columns[3].text_input("Find a customer", placeholder="Customer ID contains…")

    filtered = snapshot
    if bands:
        filtered = filtered[filtered["risk_band"].isin(bands)]
    if tiers:
        filtered = filtered[filtered["membership_tier"].isin(tiers)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    if search:
        filtered = filtered[filtered["customer_id"].str.contains(search, case=False, na=False)]
    return filtered


def _worklist(filtered: pd.DataFrame) -> None:
    """Render the ranked retention worklist."""
    st.subheader("Retention worklist")
    st.caption(
        f"{len(filtered):,} customers match the filters above, ranked by predicted churn "
        "probability. Select a row to see what makes them stand out."
    )

    # Ranked once, then narrowed for display. The drill-down is handed `ranked`, not
    # `display`: st.dataframe reports a selection as a positional index into the frame it
    # was given, and the two share a row order, so the index maps either way — but only
    # `ranked` still carries every driver column. Passing the narrowed frame silently
    # dropped three of risk.DRIVERS (support contacts, activity events, campaign
    # participation) from the indicator panel, because customer_indicators skips a driver
    # that is absent from the row rather than raising on it.
    ranked = filtered.sort_values("churn_probability", ascending=False)
    display = ranked[
        [
            "customer_id",
            "risk_band",
            "churn_probability",
            "monthly_value",
            "membership_tier",
            "region",
            "member_since_days",
            "days_since_last_successful_payment",
            "payment_failure_rate_30d",
            "avg_engagement_30d",
        ]
    ]

    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="worklist",
        column_config={
            "customer_id": st.column_config.TextColumn("Customer"),
            "risk_band": st.column_config.TextColumn("Band"),
            "churn_probability": st.column_config.ProgressColumn(
                "Churn probability", format="%.0f%%", min_value=0, max_value=1
            ),
            "monthly_value": st.column_config.NumberColumn("Monthly value", format="$%.0f"),
            "membership_tier": st.column_config.TextColumn("Tier"),
            "region": st.column_config.TextColumn("Region"),
            "member_since_days": st.column_config.NumberColumn("Tenure (days)", format="%d"),
            "days_since_last_successful_payment": st.column_config.NumberColumn(
                "Days since payment",
                format="%d",
                help="Blank means no successful payment on record.",
            ),
            "payment_failure_rate_30d": st.column_config.NumberColumn(
                "Payment failures (30d)", format="percent"
            ),
            "avg_engagement_30d": st.column_config.NumberColumn("Engagement (30d)", format="%.1f"),
        },
    )

    _drilldown(ranked)


def _drilldown(ranked: pd.DataFrame) -> None:
    """Render the indicator panel for the selected customer.

    Takes the full ranked frame rather than the narrowed one the table displays: the
    selection is a positional index, identical across both, and every risk.DRIVERS column
    has to be on the row for customer_indicators to report it.
    """
    selection = st.session_state.get("worklist")
    rows = selection.get("selection", {}).get("rows", []) if selection else []
    if not rows:
        st.caption("Select a row above to see that customer's risk indicators.")
        return

    customer = ranked.iloc[rows[0]]
    st.markdown(f"#### {customer['customer_id']}")

    # Reference is computed over the whole scored cohort, never over the filtered view: "3×
    # the median" has to mean the same thing regardless of which filters happen to be set,
    # or the same customer would show different indicators to two people looking at once.
    reference = risk.cohort_reference(_snapshot())
    indicators = risk.customer_indicators(customer, reference, settings.indicator_ratio)

    st.caption(
        "**Risk indicators — not model reasoning.** These are the measures on which this "
        "customer differs most from everyone scored the same night. They correlate with "
        "churn, but they are not the model's attribution for this particular score."
    )

    if not indicators:
        st.info(
            "This customer is close to the cohort on every indicator the dashboard tracks. "
            "The score comes from a combination of factors rather than a single standout one."
        )
        return

    for indicator in indicators:
        value = risk.format_value(indicator["value"], str(indicator["fmt"]))
        st.markdown(f"- **{indicator['label']}** — {value}, {indicator['detail']}")


def main() -> None:
    """Render the whole page."""
    snapshot = _snapshot()

    if snapshot.empty:
        st.title("Customer churn risk")
        st.info(
            "No scored snapshot is available yet. The dashboard fills in after the nightly "
            "pipeline has run batch prediction at least once.",
            icon="🕒",
        )
        return

    trend = _trend()
    palette = _active_palette()

    _header(snapshot)
    st.divider()
    _kpi_row(risk.portfolio_summary(snapshot, settings.high_risk_threshold), trend)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("How the base splits")
        st.altair_chart(charts.band_composition(snapshot, palette), theme=None)
        st.subheader("Where risk concentrates")
        dimension = st.radio(
            "Segment by",
            options=["membership_tier", "region", "preferred_channel"],
            format_func=lambda c: {
                "membership_tier": "Membership tier",
                "region": "Region",
                "preferred_channel": "Channel",
            }[c],
            horizontal=True,
            label_visibility="collapsed",
        )
        st.altair_chart(
            charts.high_risk_by_segment(
                snapshot,
                dimension,
                {"membership_tier": "tier", "region": "region", "preferred_channel": "channel"}[
                    dimension
                ],
                settings.high_risk_threshold,
                palette,
            ),
            theme=None,
        )
    with right:
        st.subheader("Predicted churn rate over time")
        if len(trend) < 2:
            st.caption("The trend appears once the pipeline has scored more than one day.")
        else:
            st.altair_chart(charts.risk_trend(trend, palette), theme=None)
            st.caption(
                "The rate the model would act on at its registered decision threshold. A "
                "step change here that coincides with a new model version is a change in "
                "the model, not in customer behaviour."
            )

    st.divider()
    _worklist(_filters(snapshot))


# Streamlit executes this file as a script with __name__ == "__main__". The guard means an
# accidental plain import (a test collecting the module, a tool walking the package) does
# not run the page and its query as a side effect.
if __name__ == "__main__":
    main()
