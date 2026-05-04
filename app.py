from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_scorer import score_vulnerabilities
from report_generator import generate_report
from scanner import scan_image
from trust_checker import check_image_trust

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DockerScanner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* tighten top padding */
  .block-container { padding-top: 1.5rem !important; }

  /* metric card */
  .mcard {
    background: #ffffff;
    border-radius: 10px;
    padding: 14px 10px;
    text-align: center;
    border-top: 4px solid #ccc;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .mcard .mlabel {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #888;
    margin: 0;
  }
  .mcard .mvalue {
    font-size: 26px;
    font-weight: 800;
    line-height: 1.1;
    margin: 4px 0 0 0;
  }
  .mcard .msub {
    font-size: 10px;
    color: #aaa;
    margin: 2px 0 0 0;
  }

  /* trust pill */
  .trust-pill {
    display: inline-block;
    padding: 3px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
  }

  /* example buttons */
  div[data-testid="column"] button[kind="secondary"] {
    font-size: 12px !important;
    padding: 4px 8px !important;
  }

  /* history entry */
  .hist-entry {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    border-left: 3px solid #ccc;
    font-size: 13px;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEVERITY_COLOR = {
    "CRITICAL": "#e63946",
    "HIGH":     "#f4a261",
    "MEDIUM":   "#e9c46a",
    "LOW":      "#2a9d8f",
    "UNKNOWN":  "#8d99ae",
}

RISK_COLOR = {
    "CRITICAL": "#e63946",
    "HIGH":     "#f4a261",
    "MEDIUM":   "#e9c46a",
    "LOW":      "#2a9d8f",
    "MINIMAL":  "#2a9d8f",
}

TRUST_CFG = {
    "OFFICIAL":   {"icon": "✅", "color": "#2a9d8f", "bg": "#d8f3dc"},
    "VERIFIED":   {"icon": "✅", "color": "#2a9d8f", "bg": "#d8f3dc"},
    "COMMUNITY":  {"icon": "ℹ️",  "color": "#457b9d", "bg": "#dbe9f4"},
    "PRIVATE":    {"icon": "🔒", "color": "#6c757d", "bg": "#f0f0f0"},
    "SUSPICIOUS": {"icon": "🚨", "color": "#e63946", "bg": "#ffdde1"},
}

EXAMPLE_IMAGES = [
    "nginx:latest", "python:3.9", "ubuntu:20.04",
    "node:18", "alpine:latest", "redis:7",
]


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def _mcard(label: str, value: str, color: str, sub: str = "") -> str:
    sub_html = f'<p class="msub">{sub}</p>' if sub else ""
    return (
        f'<div class="mcard" style="border-top-color:{color}">'
        f'<p class="mlabel">{label}</p>'
        f'<p class="mvalue" style="color:{color}">{value}</p>'
        f'{sub_html}</div>'
    )


def _trust_pill(trust_level: str) -> str:
    cfg = TRUST_CFG.get(trust_level, {"icon": "❓", "color": "#888", "bg": "#eee"})
    return (
        f'<span class="trust-pill" '
        f'style="background:{cfg["bg"]};color:{cfg["color"]}">'
        f'{cfg["icon"]} {trust_level}</span>'
    )


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _gauge(score: float, risk_level: str) -> go.Figure:
    color = RISK_COLOR.get(risk_level, "#8d99ae")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 26}},
        title={"text": f"Risk Level: <b>{risk_level}</b>", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  25], "color": "#d8f3dc"},
                {"range": [25, 50], "color": "#fff3b0"},
                {"range": [50, 75], "color": "#ffe5c0"},
                {"range": [75, 100], "color": "#ffdde1"},
            ],
            "threshold": {"line": {"color": "#e63946", "width": 3},
                          "thickness": 0.8, "value": 75},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _severity_bar(counts: dict) -> go.Figure:
    sevs   = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]
    values = [counts.get(s, 0) for s in sevs]
    colors = [SEVERITY_COLOR[s] for s in sevs]
    fig = go.Figure(go.Bar(
        x=sevs, y=values, marker_color=colors,
        text=values, textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        title="Vulnerabilities by Severity", xaxis_title=None, yaxis_title="Count",
        height=250, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(gridcolor="#e5e7eb")
    return fig


def _fixability_pie(fixable: int, unfixable: int) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=["Fixable", "No fix yet"],
        values=[max(fixable, 0), max(unfixable, 0)],
        marker_colors=["#2a9d8f", "#e63946"],
        hole=0.55, textinfo="label+percent",
    ))
    fig.update_layout(
        title="Fix Availability", height=250,
        margin=dict(l=10, r=10, t=40, b=10), showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _sidebar() -> int:
    with st.sidebar:
        st.markdown("## 🔍 DockerScanner")
        st.caption("AI-powered container security analysis")
        st.divider()

        st.markdown("**Pipeline**")
        for step in [
            "🔒 Trust & provenance check",
            "🔎 Trivy CVE database scan",
            "🤖 scikit-learn ML risk scoring",
            "📝 Claude AI report generation",
        ]:
            st.markdown(f"&nbsp;&nbsp;{step}", unsafe_allow_html=True)

        st.divider()
        timeout = st.slider("Scan timeout (s)", min_value=60, max_value=600,
                            value=300, step=30)

        # Scan history
        history = st.session_state.get("scan_history", [])
        if history:
            st.divider()
            st.markdown("**Recent Scans**")
            for h in history:
                color = RISK_COLOR.get(h["level"], "#888")
                st.markdown(
                    f'<div class="hist-entry" style="border-left-color:{color}">'
                    f'<b>{h["image"]}</b><br>'
                    f'Score <b style="color:{color}">{h["score"]}/100</b> · '
                    f'{h["total"]} CVEs · {h["trust"]}<br>'
                    f'<span style="color:#aaa">{h["time"]}</span></div>',
                    unsafe_allow_html=True,
                )

    return timeout


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------
def _render_landing() -> None:
    st.markdown("### How it works")
    cols = st.columns(4)
    steps = [
        ("🔒", "Trust Check",   "Verify image origin, detect typosquatting & supply chain risks"),
        ("🔎", "CVE Scan",      "Trivy scans against NVD, OSV, GitHub Advisories & more"),
        ("🤖", "ML Scoring",    "scikit-learn ranks CVEs by severity, CVSS score & fixability"),
        ("📝", "AI Report",     "Claude generates a plain-English remediation report"),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div style="background:#f8f9fa;border-radius:12px;padding:20px 16px;'
                f'text-align:center;border:1px solid #e9ecef;min-height:130px;">'
                f'<div style="font-size:28px">{icon}</div>'
                f'<div style="font-weight:700;margin:8px 0 4px;font-size:15px">{title}</div>'
                f'<div style="font-size:12px;color:#777;line-height:1.4">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# CVE table with filters
# ---------------------------------------------------------------------------
def _render_cve_table(scored_vulns: list) -> None:
    if not scored_vulns:
        st.success("No vulnerabilities found — this image is clean!")
        return

    # Filter controls
    f1, f2, f3 = st.columns([3, 3, 1])
    with f1:
        sev_filter = st.multiselect(
            "Severity filter",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"],
            key="sev_filter",
        )
    with f2:
        search = st.text_input(
            "Search", placeholder="CVE ID, package name…", key="cve_search",
            label_visibility="collapsed",
        )
        st.caption("Search by CVE ID or package")
    with f3:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        fixable_only = st.toggle("Fixable only", key="fix_toggle")

    # Build dataframe
    rows = []
    for v in scored_vulns:
        rows.append({
            "Score":         v.priority_score,
            "CVE ID":        v.vuln_id,
            "Package":       v.pkg_name,
            "Installed":     v.installed_version,
            "Fixed Version": v.fixed_version or "—",
            "Severity":      v.severity,
            "CVSS":          round(v.cvss_score, 1) if v.cvss_score is not None else None,
            "Risk Tier":     v.risk_tier,
            "Fixable":       "✓" if v.has_fix else "✗",
        })

    df = pd.DataFrame(rows)

    # Apply filters
    if sev_filter:
        df = df[df["Severity"].isin(sev_filter)]
    if fixable_only:
        df = df[df["Fixable"] == "✓"]
    if search:
        q = search.strip().lower()
        df = df[
            df["CVE ID"].str.lower().str.contains(q, na=False) |
            df["Package"].str.lower().str.contains(q, na=False)
        ]

    if df.empty:
        st.info("No vulnerabilities match the current filters.")
        return

    def _colour_sev(val: str) -> str:
        c = SEVERITY_COLOR.get(val, "#8d99ae")
        return f"color: {c}; font-weight: 700"

    def _colour_fix(val: str) -> str:
        return "color: #2a9d8f; font-weight:700" if val == "✓" else "color: #e63946"

    styled = (
        df.style
        .map(_colour_sev, subset=["Severity"])
        .map(_colour_fix, subset=["Fixable"])
        .format({"Score": "{:.1f}", "CVSS": lambda x: f"{x:.1f}" if x is not None else "—"})
        .background_gradient(subset=["Score"], cmap="RdYlGn_r", vmin=0, vmax=100)
    )

    st.dataframe(styled, use_container_width=True, height=460)

    # Footer row: count + CSV download
    dl_col, cap_col = st.columns([1, 3])
    with dl_col:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV", csv,
            file_name=f"cves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with cap_col:
        st.caption(
            f"Showing {len(df)} of {len(rows)} vulnerabilities · "
            "ranked by ML priority score (highest = fix first)"
        )


# ---------------------------------------------------------------------------
# Trust section
# ---------------------------------------------------------------------------
def _render_trust_section(trust) -> None:
    cfg = TRUST_CFG.get(trust.trust_level, {"icon": "❓", "color": "#888", "bg": "#eee"})

    for w in trust.warnings:
        if trust.trust_level == "SUSPICIOUS":
            st.error(f"🚨 **Supply Chain Warning:** {w}")
        else:
            st.info(f"ℹ️ {w}")

    if trust.api_error:
        st.warning(f"⚠️ Trust verification incomplete: {trust.api_error}")

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Trust Level", f"{cfg['icon']} {trust.trust_level}")
    t2.metric("Official Image",       "Yes ✅" if trust.is_official else "No")
    t3.metric("Verified Publisher",   "Yes ✅" if trust.is_verified_publisher else "No")
    if trust.is_private_registry:
        t4.metric("Registry", "Private / Enterprise")
    elif trust.pull_count > 0:
        t4.metric("Docker Hub Pulls", f"{trust.pull_count:,}")
    else:
        t4.metric("Docker Hub Pulls", "Unknown")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Namespace:** `{trust.namespace}`")
        st.markdown(f"**Repository:** `{trust.repository}`")
        st.markdown(f"**Tag:** `{trust.tag}`")
        if trust.hub_url:
            st.markdown(f"**Docker Hub:** [{trust.hub_url}]({trust.hub_url})")
    with col_b:
        if trust.typosquat_candidates:
            st.markdown("**Similar official images detected:**")
            for c in trust.typosquat_candidates[:5]:
                st.markdown(f"- `{c}` — [hub.docker.com/_/{c}](https://hub.docker.com/_/{c})")
        else:
            st.markdown("**Typosquat check:** No similar official images found ✅")

    st.divider()
    st.markdown(
        "**Trust levels explained:**\n"
        "- **Official** — Docker Library image (highest trust)\n"
        "- **Verified** — Publisher identity confirmed by Docker Hub\n"
        "- **Community** — Public image, unverified publisher\n"
        "- **Private** — Non-Docker Hub registry (neutral — cannot verify via Hub)\n"
        "- **Suspicious** — Name closely resembles a known official image"
    )


# ---------------------------------------------------------------------------
# Top fixes summary
# ---------------------------------------------------------------------------
def _render_top_fixes(scored_vulns: list) -> None:
    fixable = [v for v in scored_vulns if v.has_fix][:5]
    if not fixable:
        st.info("No vulnerabilities with available fixes were found.")
        return

    st.markdown("**Top vulnerabilities to patch right now:**")
    for i, v in enumerate(fixable, 1):
        color = SEVERITY_COLOR.get(v.severity, "#888")
        st.markdown(
            f'<div style="background:#f8f9fa;border-radius:8px;padding:10px 14px;'
            f'margin-bottom:6px;border-left:4px solid {color}">'
            f'<b>{i}. {v.vuln_id}</b> &nbsp;'
            f'<span style="color:{color};font-weight:700">{v.severity}</span>'
            f'&nbsp;·&nbsp;<code>{v.pkg_name}</code><br>'
            f'<span style="font-size:12px;color:#555">Upgrade <b>{v.installed_version}</b>'
            f' → <b>{v.fixed_version}</b></span></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _set_quick_image(img: str) -> None:
    st.session_state["image_input"] = img


def main() -> None:
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []

    timeout = _sidebar()

    # Header
    st.markdown(
        '<h1 style="margin-bottom:2px">🔍 DockerScanner</h1>'
        '<p style="color:#666;margin-top:0;font-size:16px">'
        'AI-powered Docker image security analysis — CVE scanning, ML risk scoring & plain-English reports'
        '</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Search bar
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        image_name = st.text_input(
            label="image",
            placeholder="Enter a Docker image name, e.g.  nginx:latest  or  python:3.9-slim",
            label_visibility="collapsed",
            key="image_input",
        )
    with col_btn:
        scan_clicked = st.button("Scan", type="primary", use_container_width=True)

    # Quick-scan example buttons — on_click fires before rerun, so session state
    # is written before the text_input widget is instantiated (no key conflict).
    st.markdown('<p style="font-size:12px;color:#999;margin:4px 0 2px">Quick examples:</p>',
                unsafe_allow_html=True)
    ex_cols = st.columns(len(EXAMPLE_IMAGES))
    for col, img in zip(ex_cols, EXAMPLE_IMAGES):
        col.button(img, key=f"ex_{img}", on_click=_set_quick_image,
                   args=(img,), use_container_width=True)

    # No scan yet — show landing page
    if not scan_clicked and not st.session_state.get("_last_scan"):
        st.divider()
        _render_landing()
        return

    # Resolve image name (button click or leftover from last run)
    target = image_name.strip() if scan_clicked and image_name else ""
    if not target:
        if not scan_clicked:
            st.divider()
            _render_landing()
        else:
            st.warning("Please enter a Docker image name.")
        return

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    with st.status(f"Analysing **{target}** …", expanded=True) as status:
        st.write("Step 1/4 — Checking image trust & provenance …")
        trust_report = check_image_trust(target)

        st.write("Step 2/4 — Running Trivy vulnerability scan …")
        scan_result = scan_image(target, timeout=timeout)

        if not scan_result.scan_success:
            status.update(label="Scan failed", state="error", expanded=True)
            st.error(f"**Trivy error:** {scan_result.error_message}")
            return

        st.write(
            f"Step 3/4 — Found **{scan_result.total_count}** vulnerabilities. "
            "Running ML scoring …"
        )
        risk_score = score_vulnerabilities(scan_result)

        st.write(
            f"Step 4/4 — Risk score **{risk_score.overall_score}/100**. "
            "Generating Claude AI report …"
        )
        report_text = generate_report(scan_result, risk_score, trust_report)
        status.update(
            label=f"✅ Done — {scan_result.total_count} CVEs · Risk {risk_score.overall_score}/100",
            state="complete",
        )

    # Save to history
    st.session_state["_last_scan"] = True
    history = st.session_state.scan_history
    history.insert(0, {
        "image": target,
        "score": risk_score.overall_score,
        "level": risk_score.risk_level,
        "total": scan_result.total_count,
        "trust": trust_report.trust_level,
        "time":  datetime.now().strftime("%H:%M:%S"),
    })
    st.session_state.scan_history = history[:6]

    # ------------------------------------------------------------------
    # Results header
    # ------------------------------------------------------------------
    st.divider()

    img_col, pill_col = st.columns([3, 1])
    with img_col:
        st.markdown(f"### Results for `{target}`")
    with pill_col:
        st.markdown(
            f'<div style="text-align:right;padding-top:8px">'
            f'{_trust_pill(trust_report.trust_level)}</div>',
            unsafe_allow_html=True,
        )

    # Metric cards
    fs = risk_score.feature_summary
    risk_color = RISK_COLOR.get(risk_score.risk_level, "#888")
    cards = [
        ("Risk Score",  f"{risk_score.overall_score:.0f}/100", risk_color,  risk_score.risk_level),
        ("Total CVEs",  str(fs["total"]),   "#457b9d", f"{fs['unique_packages']} packages"),
        ("Critical",    str(fs["critical"]), SEVERITY_COLOR["CRITICAL"], ""),
        ("High",        str(fs["high"]),     SEVERITY_COLOR["HIGH"],     ""),
        ("Fixable",     str(fs["fixable"]),  "#2a9d8f",
         f"{round(fs['fixable']/fs['total']*100) if fs['total'] else 0}% patchable"),
        ("No Fix Yet",  str(fs["unfixable"]), "#e63946", "vendor patch needed"),
    ]
    card_cols = st.columns(6)
    for col, (label, value, color, sub) in zip(card_cols, cards):
        col.markdown(_mcard(label, value, color, sub), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Trust alert banners
    if trust_report.trust_level == "SUSPICIOUS":
        st.error(
            "🚨 **Supply Chain Risk:** This image name closely resembles an official image. "
            "See the **Trust & Provenance** tab before deploying."
        )
    elif trust_report.trust_level in ("OFFICIAL", "VERIFIED"):
        cfg = TRUST_CFG[trust_report.trust_level]
        st.success(f"{cfg['icon']} **{trust_report.trust_level}** — "
                   f"{'Official Docker Library image' if trust_report.is_official else 'Verified publisher on Docker Hub'}.")

    # Charts
    st.divider()
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        st.plotly_chart(_severity_bar(scan_result.count_by_severity()), use_container_width=True)
    with ch2:
        st.plotly_chart(_gauge(risk_score.overall_score, risk_score.risk_level), use_container_width=True)
    with ch3:
        st.plotly_chart(_fixability_pie(fs["fixable"], fs["unfixable"]), use_container_width=True)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    trust_icon = TRUST_CFG.get(trust_report.trust_level, {}).get("icon", "🔒")
    tab_cve, tab_fixes, tab_trust, tab_report = st.tabs([
        "CVE Details",
        "Top Fixes",
        f"{trust_icon} Trust & Provenance",
        "AI Security Report",
    ])

    with tab_cve:
        _render_cve_table(risk_score.scored_vulnerabilities)

    with tab_fixes:
        st.subheader("Highest-Priority Fixable Vulnerabilities")
        st.caption("These are the ML top-ranked CVEs that have a patch available — fix these first.")
        _render_top_fixes(risk_score.scored_vulnerabilities)

    with tab_trust:
        st.subheader("Image Trust & Provenance")
        _render_trust_section(trust_report)

    with tab_report:
        st.markdown(
            f"*Claude report for `{target}` · "
            f"Risk: {risk_score.overall_score}/100 [{risk_score.risk_level}] · "
            f"Trust: {trust_report.trust_level}*"
        )
        st.divider()
        if report_text.startswith("**Report generation failed"):
            st.error(report_text)
        else:
            st.markdown(report_text)


if __name__ == "__main__":
    main()
