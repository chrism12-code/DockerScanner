"""Quick batch experiment runner — no Claude API calls, just Trivy + ML scoring."""
from __future__ import annotations
import json, sys
from scanner import scan_image
from ml_scorer import score_vulnerabilities

EXPERIMENT_1 = [
    "nginx:latest",
    "python:3.9",
    "node:18",
    "ubuntu:20.04",
    "alpine:latest",
]

EXPERIMENT_2 = [
    ("nginx:1.20",   "nginx:latest"),
    ("python:3.8",   "python:3.12"),
    ("ubuntu:18.04", "ubuntu:22.04"),
]

def scan_and_score(image):
    r = scan_image(image, timeout=300)
    if not r.scan_success:
        return None, f"FAILED: {r.error_message}"
    s = score_vulnerabilities(r)
    fs = s.feature_summary
    return s, {
        "image": image,
        "risk_score": s.overall_score,
        "risk_level": s.risk_level,
        "total": fs["total"],
        "critical": fs["critical"],
        "high": fs["high"],
        "medium": fs["medium"],
        "low": fs["low"],
        "fixable": fs["fixable"],
        "unfixable": fs["unfixable"],
        "packages": fs["unique_packages"],
    }

print("=" * 60)
print("EXPERIMENT 1 — Risk scores across different Docker images")
print("=" * 60)
exp1_results = []
for img in EXPERIMENT_1:
    print(f"  Scanning {img} ...", end=" ", flush=True)
    _, data = scan_and_score(img)
    if isinstance(data, str):
        print(data)
    else:
        exp1_results.append(data)
        print(f"score={data['risk_score']}/100  [{data['risk_level']}]  "
              f"total={data['total']}  fixable={data['fixable']}")

print()
print("=" * 60)
print("EXPERIMENT 2 — Risk score: old vs new image versions")
print("=" * 60)
exp2_results = []
for old, new in EXPERIMENT_2:
    print(f"  {old} vs {new}")
    _, d_old = scan_and_score(old)
    _, d_new = scan_and_score(new)
    if isinstance(d_old, str) or isinstance(d_new, str):
        print(f"    SKIP (scan failed)")
        continue
    delta = round(d_new["risk_score"] - d_old["risk_score"], 2)
    exp2_results.append({"old": d_old, "new": d_new, "delta": delta})
    print(f"    {old}: {d_old['risk_score']}/100  ({d_old['total']} CVEs)")
    print(f"    {new}: {d_new['risk_score']}/100  ({d_new['total']} CVEs)  delta={delta:+.1f}")

# Save for reference
with open("experiment_results.json", "w") as f:
    json.dump({"exp1": exp1_results, "exp2": exp2_results}, f, indent=2)
print("\nResults saved to experiment_results.json")
