from __future__ import annotations

"""ML-based vulnerability risk scoring using scikit-learn.

Pipeline
--------
1. Extract numeric features from each CVE.
2. MinMaxScaler normalises features to [0, 1].
3. Weighted dot-product produces a per-CVE priority score (0-100).
4. KMeans (k=4) clusters CVEs into risk tiers independent of the
   per-CVE score, giving the model two complementary ML outputs.
5. A severity-weighted aggregate over the whole image yields the
   overall 0-100 image risk score.
"""

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# Numeric encoding for ordinal severity
SEVERITY_NUM = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}

# Exponential weights used for the *image-level* aggregate score.
# Reflect real-world triage: a single CRITICAL matters far more than
# many LOWs.
SEVERITY_WEIGHT = {"CRITICAL": 10.0, "HIGH": 4.0, "MEDIUM": 1.5, "LOW": 0.4, "UNKNOWN": 0.1}

# Feature weights for the per-CVE priority score (must sum to 1.0)
FEATURE_WEIGHTS = np.array([
    0.35,   # severity_num   – severity ordinal
    0.30,   # cvss_score     – CVSS V3/V2 numeric
    0.20,   # has_fix        – actionability: can we patch it?
    0.15,   # pkg_vuln_count – package-level exposure breadth
])

RISK_TIER_LABELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def _estimate_cvss(severity: str) -> float:
    """Return a reasonable CVSS estimate when no official score exists."""
    return {"CRITICAL": 9.0, "HIGH": 7.5, "MEDIUM": 5.0, "LOW": 2.0, "UNKNOWN": 3.0}.get(
        severity, 3.0
    )


@dataclass
class ScoredVulnerability:
    vuln_id: str
    pkg_name: str
    severity: str
    cvss_score: Optional[float]        # None if no official score available
    has_fix: bool
    priority_score: float              # 0-100, ML-ranked
    risk_tier: str                     # KMeans-derived cluster label
    installed_version: str
    fixed_version: Optional[str]
    description: str


@dataclass
class ImageRiskScore:
    overall_score: float               # 0-100
    risk_level: str                    # CRITICAL / HIGH / MEDIUM / LOW / MINIMAL
    scored_vulnerabilities: list       # ScoredVulnerability, sorted by priority_score desc
    feature_summary: dict


def score_vulnerabilities(scan_result) -> ImageRiskScore:
    """Score and rank all vulnerabilities in *scan_result*.

    Returns an :class:`ImageRiskScore` with an overall image risk score,
    a human-readable risk level, and a ranked list of scored CVEs.
    """
    vulns = scan_result.vulnerabilities

    if not vulns:
        return ImageRiskScore(
            overall_score=0.0,
            risk_level="MINIMAL",
            scored_vulnerabilities=[],
            feature_summary={"total": 0, "critical": 0, "high": 0,
                             "medium": 0, "low": 0, "unknown": 0,
                             "fixable": 0, "unfixable": 0, "unique_packages": 0},
        )

    # ------------------------------------------------------------------
    # 1. Build a feature DataFrame
    # ------------------------------------------------------------------
    rows = []
    for v in vulns:
        official_cvss = v.cvss_score
        cvss_used = official_cvss if official_cvss is not None else _estimate_cvss(v.severity)
        rows.append({
            "vuln_id":           v.vuln_id,
            "pkg_name":          v.pkg_name,
            "severity":          v.severity,
            "severity_num":      SEVERITY_NUM.get(v.severity, 0),
            "cvss_score":        cvss_used,
            "cvss_official":     official_cvss,
            "has_fix":           1 if v.fixed_version else 0,
            "installed_version": v.installed_version,
            "fixed_version":     v.fixed_version,
            "description":       v.description,
        })

    df = pd.DataFrame(rows)

    # Per-package vulnerability count (higher = systemic risk in that package)
    pkg_counts = df["pkg_name"].value_counts()
    df["pkg_vuln_count"] = df["pkg_name"].map(pkg_counts).astype(float)

    # ------------------------------------------------------------------
    # 2. Scale features and compute weighted priority scores
    # ------------------------------------------------------------------
    feature_cols = ["severity_num", "cvss_score", "has_fix", "pkg_vuln_count"]
    X = df[feature_cols].values.astype(float)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)          # shape (n, 4), values in [0, 1]

    raw_scores = X_scaled @ FEATURE_WEIGHTS     # dot product → weighted sum ∈ [0, 1]
    df["priority_score"] = raw_scores * 100     # scale to [0, 100]

    # ------------------------------------------------------------------
    # 3. KMeans clustering → risk tier labels
    # ------------------------------------------------------------------
    n_clusters = min(4, len(df))
    if n_clusters >= 2:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(X_scaled)
        # Rank cluster centres by their own weighted score to assign tier names
        centre_scores = km.cluster_centers_ @ FEATURE_WEIGHTS
        sorted_idx = np.argsort(centre_scores)[::-1]           # highest → lowest
        id_to_tier = {int(sorted_idx[i]): RISK_TIER_LABELS[i] for i in range(n_clusters)}
        df["risk_tier"] = [id_to_tier[int(c)] for c in cluster_ids]
    else:
        df["risk_tier"] = df["severity"].apply(
            lambda s: s if s in RISK_TIER_LABELS else "MEDIUM"
        )

    # ------------------------------------------------------------------
    # 4. Overall image risk score
    # ------------------------------------------------------------------
    sev_counts = df["severity"].value_counts()
    total = len(df)

    weighted_sum = sum(
        SEVERITY_WEIGHT.get(sev, 0.1) * cnt for sev, cnt in sev_counts.items()
    )
    max_possible = SEVERITY_WEIGHT["CRITICAL"] * total
    base_risk = (weighted_sum / max_possible) * 100 if max_possible > 0 else 0.0

    # Logarithmic volume boost: penalise images with many vulns
    volume_boost = min(20.0, float(np.log1p(total) * 3.5))

    overall_score = min(100.0, base_risk * 0.80 + volume_boost)

    if overall_score >= 75:
        risk_level = "CRITICAL"
    elif overall_score >= 50:
        risk_level = "HIGH"
    elif overall_score >= 25:
        risk_level = "MEDIUM"
    elif overall_score > 0:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    # ------------------------------------------------------------------
    # 5. Sort and return
    # ------------------------------------------------------------------
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)

    scored_vulns = [
        ScoredVulnerability(
            vuln_id=row["vuln_id"],
            pkg_name=row["pkg_name"],
            severity=row["severity"],
            cvss_score=row["cvss_official"],
            has_fix=bool(row["has_fix"]),
            priority_score=round(float(row["priority_score"]), 2),
            risk_tier=row["risk_tier"],
            installed_version=row["installed_version"],
            fixed_version=row["fixed_version"],
            description=row["description"],
        )
        for _, row in df.iterrows()
    ]

    feature_summary = {
        "total":           total,
        "critical":        int(sev_counts.get("CRITICAL", 0)),
        "high":            int(sev_counts.get("HIGH", 0)),
        "medium":          int(sev_counts.get("MEDIUM", 0)),
        "low":             int(sev_counts.get("LOW", 0)),
        "unknown":         int(sev_counts.get("UNKNOWN", 0)),
        "fixable":         int(df["has_fix"].sum()),
        "unfixable":       int(total - df["has_fix"].sum()),
        "unique_packages": int(df["pkg_name"].nunique()),
    }

    return ImageRiskScore(
        overall_score=round(overall_score, 2),
        risk_level=risk_level,
        scored_vulnerabilities=scored_vulns,
        feature_summary=feature_summary,
    )


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scanner import scan_image
    target = sys.argv[1] if len(sys.argv) > 1 else "nginx:latest"
    result = scan_image(target)
    if not result.scan_success:
        print(f"Scan failed: {result.error_message}", file=sys.stderr)
        sys.exit(1)
    risk = score_vulnerabilities(result)
    print(f"Image        : {target}")
    print(f"Risk score   : {risk.overall_score}/100  [{risk.risk_level}]")
    print(f"Summary      : {risk.feature_summary}")
    print(f"Top 3 CVEs   :")
    for v in risk.scored_vulnerabilities[:3]:
        print(f"  {v.priority_score:5.1f}  {v.vuln_id}  {v.pkg_name}  [{v.severity}]  tier={v.risk_tier}")
