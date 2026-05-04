from __future__ import annotations

"""Generates plain-English security reports via the Claude API.

Prompt caching is enabled on the (large, static) system prompt so that
repeated scans only pay the full token cost once per cache TTL window.
"""

import os
import sys
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Cached at the API level — only billed on first request per 5-min window.
_SYSTEM_PROMPT = """\
You are a senior cybersecurity engineer who specialises in container and supply-chain security.
Your audience is a software development team that includes both engineers and non-technical
stakeholders (product managers, executives).

When writing security reports you:
- Lead with business impact, not jargon
- Prioritise findings by risk, not by CVE ID
- Give concrete, copy-pasteable remediation commands where possible
- Distinguish between "fix now" (exploitable/critical) and "fix next sprint" (medium/low)
- Are honest about uncertainty — if a vulnerability's exploitability is unclear, say so
- Keep each section concise; bullet points over walls of text
"""


def _build_trust_section(trust_report) -> str:
    if trust_report is None:
        return ""
    lines = [
        "## Image Trust & Provenance",
        f"Trust level: {trust_report.trust_level}",
        f"Official Docker image: {'Yes' if trust_report.is_official else 'No'}",
        f"Verified publisher: {'Yes' if trust_report.is_verified_publisher else 'No'}",
        f"Registry: {'Private/enterprise registry' if trust_report.is_private_registry else 'Docker Hub'}",
    ]
    if trust_report.pull_count > 0:
        lines.append(f"Docker Hub pull count: {trust_report.pull_count:,}")
    if trust_report.typosquat_candidates:
        lines.append(
            f"SUPPLY CHAIN WARNING — image name resembles official image(s): "
            f"{', '.join(trust_report.typosquat_candidates)}. Possible typosquatting."
        )
    for w in trust_report.warnings:
        lines.append(f"Note: {w}")
    return "\n".join(lines)


def _build_prompt(scan_result, risk_score, trust_report=None) -> str:
    fs = risk_score.feature_summary
    top_vulns = risk_score.scored_vulnerabilities[:20]

    vuln_lines: list[str] = []
    for i, v in enumerate(top_vulns, 1):
        if v.has_fix and v.fixed_version:
            fix_note = f"Fix: upgrade to {v.fixed_version}"
        else:
            fix_note = "No fix available yet"

        cvss_note = f"CVSS {v.cvss_score:.1f}" if v.cvss_score is not None else "CVSS N/A"
        desc = v.description[:220].rstrip() + ("…" if len(v.description) > 220 else "")
        vuln_lines.append(
            f"{i:2}. [{v.severity:<8}] {v.vuln_id}  |  pkg: {v.pkg_name} {v.installed_version}"
            f"  |  {cvss_note}  |  priority={v.priority_score:.1f}/100  |  {fix_note}\n"
            f"     {desc}"
        )

    fixable_pct = (
        round(fs["fixable"] / fs["total"] * 100) if fs["total"] > 0 else 0
    )

    trust_section = _build_trust_section(trust_report)

    return f"""\
## Scan Target
Image: {scan_result.image_name}
Overall Risk Score: {risk_score.overall_score}/100  ({risk_score.risk_level})

{trust_section}

## Vulnerability Summary
| Severity  | Count |
|-----------|-------|
| CRITICAL  | {fs['critical']}     |
| HIGH      | {fs['high']}     |
| MEDIUM    | {fs['medium']}     |
| LOW       | {fs['low']}     |
| UNKNOWN   | {fs['unknown']}     |
| **TOTAL** | **{fs['total']}** |

Fixable (patch available): {fs['fixable']} / {fs['total']} ({fixable_pct}%)
Unique affected packages: {fs['unique_packages']}

## ML-Ranked Top Vulnerabilities
{chr(10).join(vuln_lines)}

---

Write a security incident report with EXACTLY these five sections:

### 1. Executive Summary
2-3 sentences. Suitable for a non-technical stakeholder. State what was scanned,
the headline risk rating, and the single most important action required.

### 2. Risk Assessment
Interpret the risk score of {risk_score.overall_score}/100. Explain what drives the score
(volume of critical/high CVEs, % fixable, etc.) and what it means operationally
(e.g., "do not deploy to production without remediation").

### 3. Critical Findings
Detail the top 5 most dangerous vulnerabilities. For each: what it is, why it matters,
potential impact if exploited, and whether a fix exists.

### 4. Remediation Plan
Ordered action list. Group by urgency:
- **Immediate (today):** CRITICAL issues with available fixes
- **Short-term (this sprint):** HIGH issues and CRITICAL without fixes
- **Backlog:** MEDIUM/LOW issues

Where possible include concrete steps (e.g., "Update nginx from 1.23.1 to 1.25.4").

### 5. Prevention Recommendations
3-5 actionable practices to reduce future vulnerability exposure for this image
(e.g., use distroless base images, pin digest not tag, add Trivy to CI pipeline).
{"" if trust_report is None or trust_report.trust_level not in ("SUSPICIOUS", "COMMUNITY") else chr(10) + "### 6. Supply Chain Assessment" + chr(10) + "Given the trust level above, assess the supply chain risk and recommend verification steps."}
"""


def generate_report(scan_result, risk_score, trust_report=None, model: str = "claude-sonnet-4-6") -> str:
    """Call the Claude API and return a Markdown security report string."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "**Report generation failed:** `ANTHROPIC_API_KEY` is not set. "
            "Add it to your `.env` file and restart the app."
        )

    client = anthropic.Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},   # prompt caching
                }
            ],
            messages=[
                {"role": "user", "content": _build_prompt(scan_result, risk_score, trust_report)}
            ],
        )
        return response.content[0].text
    except anthropic.AuthenticationError:
        return "**Report generation failed:** Invalid API key. Check `ANTHROPIC_API_KEY` in your `.env`."
    except anthropic.RateLimitError:
        return "**Report generation failed:** Claude API rate limit reached. Wait a moment and retry."
    except anthropic.APIError as exc:
        return f"**Report generation failed:** Claude API error — {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"**Report generation failed:** Unexpected error — {exc}"


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from scanner import scan_image
    from ml_scorer import score_vulnerabilities

    target = sys.argv[1] if len(sys.argv) > 1 else "nginx:latest"
    print(f"Scanning {target} …")
    result = scan_image(target)
    if not result.scan_success:
        print(f"Scan failed: {result.error_message}", file=sys.stderr)
        sys.exit(1)
    risk = score_vulnerabilities(result)
    print(f"Generating report for risk score {risk.overall_score}/100 …\n")
    report = generate_report(result, risk)
    print(report)
