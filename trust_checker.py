from __future__ import annotations

"""Image trust and provenance verification.

Checks three things independently:
1. Typosquatting — fuzzy-matches the image repo name against ~50 known official
   images to catch look-alike names (e.g. 'nignx' vs 'nginx').
2. Docker Hub publisher status — queries the public Hub API to confirm whether
   the image is an official Docker Library image or from a verified publisher.
3. Registry type — distinguishes Docker Hub images from private/enterprise
   registries. Private registries are flagged as PRIVATE (neutral), not
   suspicious — they are standard in enterprise environments.
"""

import difflib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Known official Docker Hub images used for typosquat comparison
# ---------------------------------------------------------------------------
OFFICIAL_IMAGES: frozenset[str] = frozenset([
    "nginx", "python", "node", "ubuntu", "alpine", "debian", "redis",
    "postgres", "mysql", "mongo", "elasticsearch", "kibana", "grafana",
    "jenkins", "wordpress", "php", "ruby", "golang", "rust", "openjdk",
    "gcc", "maven", "memcached", "rabbitmq", "httpd", "tomcat", "haproxy",
    "traefik", "caddy", "vault", "consul", "prometheus", "influxdb",
    "cassandra", "couchdb", "neo4j", "mariadb", "percona", "sonarqube",
    "registry", "busybox", "amazonlinux", "centos", "fedora", "rockylinux",
    "almalinux", "oraclelinux", "swift", "erlang", "elixir", "haskell",
    "tensorflow", "jupyter", "pytorch", "flink", "spark", "kafka",
    "zookeeper", "nats", "etcd", "nextcloud", "ghost", "drupal",
])

# Similarity ratio (0-1) above which a non-exact name is flagged as suspicious
TYPOSQUAT_THRESHOLD = 0.78


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class TrustReport:
    image_name: str
    namespace: str          # "library" for official images
    repository: str
    tag: str
    is_private_registry: bool
    is_official: bool       # Docker Library official image
    is_verified_publisher: bool
    pull_count: int
    star_count: int
    trust_level: str        # OFFICIAL | VERIFIED | COMMUNITY | SUSPICIOUS | PRIVATE
    typosquat_candidates: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    hub_url: str = ""
    api_error: str = ""
    same_name_as_official: bool = False  # different namespace but same name as official


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_image_ref(image_name: str) -> tuple[str, str, str, bool]:
    """Return (namespace, repository, tag, is_private_registry).

    Private registry is defined as: the first path segment contains a '.' or
    ':' (i.e. it looks like a hostname/host:port), or there are 3+ path
    segments (which implies a registry path prefix).
    """
    # Strip digest reference  (e.g. image@sha256:abc...)
    if "@" in image_name:
        image_name = image_name.split("@")[0]

    parts = image_name.split("/")

    # Pull the tag off the final path segment
    last = parts[-1]
    if ":" in last:
        parts[-1], tag = last.rsplit(":", 1)
    else:
        tag = "latest"

    first = parts[0]
    is_private = "." in first or ":" in first or len(parts) >= 3

    if is_private:
        registry = first
        repo = "/".join(parts[1:]) if len(parts) > 1 else ""
        return registry, repo, tag, True

    if len(parts) == 1:
        return "library", parts[0], tag, False

    return parts[0], parts[1], tag, False


def _fetch_hub_info(namespace: str, repository: str, timeout: int = 10) -> dict:
    url = f"https://hub.docker.com/v2/repositories/{namespace}/{repository}/"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "DockerScanner/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _find_typosquat_candidates(repo_name: str) -> list[str]:
    """Return official image names that are suspiciously similar to *repo_name*."""
    repo_lower = repo_name.lower()
    if repo_lower in OFFICIAL_IMAGES:
        return []   # exact match — no concern
    hits: list[tuple[float, str]] = []
    for official in OFFICIAL_IMAGES:
        ratio = difflib.SequenceMatcher(None, repo_lower, official).ratio()
        if ratio >= TYPOSQUAT_THRESHOLD:
            hits.append((ratio, official))
    hits.sort(reverse=True)
    return [name for _, name in hits]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check_image_trust(image_name: str, timeout: int = 10) -> TrustReport:
    """Analyse *image_name* for provenance and supply-chain risks."""
    namespace, repository, tag, is_private = _parse_image_ref(image_name.strip())
    typosquat_candidates = _find_typosquat_candidates(repository)

    # ------------------------------------------------------------------
    # Private / non-Docker-Hub registry
    # ------------------------------------------------------------------
    if is_private:
        warnings = []
        if typosquat_candidates:
            warnings.append(
                f"Image name '{repository}' closely resembles the official Docker Hub "
                f"image(s): {', '.join(typosquat_candidates)}. "
                "Confirm this is the correct image before deploying."
            )
        return TrustReport(
            image_name=image_name,
            namespace=namespace,
            repository=repository,
            tag=tag,
            is_private_registry=True,
            is_official=False,
            is_verified_publisher=False,
            pull_count=0,
            star_count=0,
            trust_level="PRIVATE",
            typosquat_candidates=typosquat_candidates,
            warnings=warnings,
            hub_url="",
            same_name_as_official=False,
        )

    # ------------------------------------------------------------------
    # Docker Hub image — query the public API
    # ------------------------------------------------------------------
    hub_url = (
        f"https://hub.docker.com/_/{repository}"
        if namespace == "library"
        else f"https://hub.docker.com/r/{namespace}/{repository}"
    )

    is_official = False
    is_verified = False
    pull_count = 0
    star_count = 0
    api_error = ""

    try:
        data = _fetch_hub_info(namespace, repository, timeout)
        # Docker Hub API inconsistently sets is_official for library/* images.
        # The library namespace is exclusively used by Docker's official image program,
        # so namespace alone is a reliable signal.
        is_official = bool(data.get("is_official", False)) or namespace == "library"
        is_verified = bool(data.get("is_trusted", False))
        pull_count = int(data.get("pull_count", 0))
        star_count = int(data.get("star_count", 0))
    except urllib.error.HTTPError as exc:
        api_error = (
            "Image not found on Docker Hub — it may be private, misspelled, or local-only."
            if exc.code == 404
            else f"Docker Hub API returned HTTP {exc.code}."
        )
    except urllib.error.URLError as exc:
        api_error = f"Could not reach Docker Hub (network error): {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        api_error = f"Unexpected error querying Docker Hub: {exc}"

    same_name_as_official = (
        namespace != "library" and repository.lower() in OFFICIAL_IMAGES
    )

    # ------------------------------------------------------------------
    # Trust level (priority order: SUSPICIOUS > OFFICIAL > VERIFIED > COMMUNITY)
    # ------------------------------------------------------------------
    if typosquat_candidates and not is_official:
        trust_level = "SUSPICIOUS"
    elif is_official:
        trust_level = "OFFICIAL"
    elif is_verified:
        trust_level = "VERIFIED"
    else:
        trust_level = "COMMUNITY"

    # ------------------------------------------------------------------
    # Human-readable warnings
    # ------------------------------------------------------------------
    warnings: list[str] = []

    if typosquat_candidates and not is_official:
        similar = ", ".join(f"'{c}'" for c in typosquat_candidates[:3])
        warnings.append(
            f"The image name '{repository}' is very similar to the official Docker Hub "
            f"image(s) {similar}. This could be a typosquatting attempt. "
            f"If you intended the official image, use: docker pull {typosquat_candidates[0]}"
        )

    if same_name_as_official and not is_official:
        warnings.append(
            f"'{namespace}/{repository}' shares its name with the official Docker image "
            f"'{repository}' (library/{repository}) but is a different image published under "
            f"a different namespace. Verify you trust the '{namespace}' publisher."
        )

    if trust_level == "COMMUNITY" and 0 < pull_count < 10_000:
        warnings.append(
            f"Low pull count ({pull_count:,} pulls) — this community image has limited "
            "adoption. Review the Dockerfile source before deploying to production."
        )

    return TrustReport(
        image_name=image_name,
        namespace=namespace,
        repository=repository,
        tag=tag,
        is_private_registry=False,
        is_official=is_official,
        is_verified_publisher=is_verified,
        pull_count=pull_count,
        star_count=star_count,
        trust_level=trust_level,
        typosquat_candidates=typosquat_candidates,
        warnings=warnings,
        hub_url=hub_url,
        api_error=api_error,
        same_name_as_official=same_name_as_official,
    )


# ---------------------------------------------------------------------------
# Quick CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    targets = sys.argv[1:] or ["nginx:latest", "nignx:latest", "bitnami/nginx:latest"]
    for t in targets:
        r = check_image_trust(t)
        print(f"\n{'='*60}")
        print(f"Image      : {r.image_name}")
        print(f"Parsed     : {r.namespace}/{r.repository}:{r.tag}")
        print(f"Trust      : {r.trust_level}")
        print(f"Official   : {r.is_official}  |  Verified: {r.is_verified_publisher}")
        print(f"Pulls      : {r.pull_count:,}")
        print(f"Typosquats : {r.typosquat_candidates}")
        for w in r.warnings:
            print(f"WARNING    : {w}")
        if r.api_error:
            print(f"API error  : {r.api_error}")
