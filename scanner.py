from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"]


@dataclass
class Vulnerability:
    vuln_id: str
    pkg_name: str
    installed_version: str
    fixed_version: Optional[str]
    severity: str
    cvss_score: Optional[float]
    description: str
    title: str = ""
    references: list = field(default_factory=list)


@dataclass
class ScanResult:
    image_name: str
    vulnerabilities: list
    scan_success: bool
    error_message: str = ""
    raw_output: dict = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        return len(self.vulnerabilities)

    def count_by_severity(self) -> dict:
        counts = {s: 0 for s in SEVERITY_ORDER}
        for v in self.vulnerabilities:
            sev = v.severity if v.severity in counts else "UNKNOWN"
            counts[sev] += 1
        return counts


def _extract_cvss_score(vuln_data: dict) -> Optional[float]:
    cvss = vuln_data.get("CVSS") or {}
    # Prefer NVD V3, then other sources V3, then V2 fallback
    for source in ("nvd", "redhat", "ghsa", "alma", "oracle"):
        src = cvss.get(source, {})
        if "V3Score" in src:
            return float(src["V3Score"])
    for source in ("nvd", "redhat"):
        src = cvss.get(source, {})
        if "V2Score" in src:
            return float(src["V2Score"])
    return None


def _run_trivy(image_name: str, timeout: int) -> tuple:
    """Returns (success: bool, data: dict, error: str)."""
    cmd = ["trivy", "image", "--format", "json", "--quiet", image_name]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        # Trivy exits non-zero when vulns are found but still emits valid JSON
        if not stdout.strip():
            return False, {}, stderr.strip() or "Trivy produced no output."
        data = json.loads(stdout)
        return True, data, ""
    except subprocess.TimeoutExpired:
        return False, {}, f"Scan timed out after {timeout}s. Try increasing the timeout."
    except FileNotFoundError:
        return False, {}, (
            "Trivy executable not found. "
            "Install it from https://aquasecurity.github.io/trivy/latest/getting-started/installation/"
        )
    except json.JSONDecodeError as exc:
        return False, {}, f"Could not parse Trivy JSON output: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, {}, str(exc)


def _parse_trivy_output(data: dict, image_name: str) -> ScanResult:
    vulnerabilities = []
    for result in data.get("Results") or []:
        for v in result.get("Vulnerabilities") or []:
            fixed = (v.get("FixedVersion") or "").strip()
            vulnerabilities.append(
                Vulnerability(
                    vuln_id=v.get("VulnerabilityID", "UNKNOWN"),
                    pkg_name=v.get("PkgName", ""),
                    installed_version=v.get("InstalledVersion", ""),
                    fixed_version=fixed or None,
                    severity=v.get("Severity", "UNKNOWN").upper(),
                    cvss_score=_extract_cvss_score(v),
                    description=v.get("Description", ""),
                    title=v.get("Title", ""),
                    references=v.get("References") or [],
                )
            )
    return ScanResult(
        image_name=image_name,
        vulnerabilities=vulnerabilities,
        scan_success=True,
        raw_output=data,
    )


def scan_image(image_name: str, timeout: int = 300) -> ScanResult:
    """Run a Trivy scan on *image_name* and return a structured ScanResult."""
    success, data, error = _run_trivy(image_name.strip(), timeout)
    if not success:
        return ScanResult(
            image_name=image_name,
            vulnerabilities=[],
            scan_success=False,
            error_message=error,
        )
    return _parse_trivy_output(data, image_name)


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "nginx:latest"
    print(f"Scanning {target} …")
    result = scan_image(target)
    if result.scan_success:
        print(f"Total vulnerabilities : {result.total_count}")
        print(f"Severity breakdown    : {result.count_by_severity()}")
        if result.vulnerabilities:
            sample = result.vulnerabilities[0]
            print(f"Sample CVE            : {sample.vuln_id} ({sample.severity})")
    else:
        print(f"ERROR: {result.error_message}", file=sys.stderr)
        sys.exit(1)
