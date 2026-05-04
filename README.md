# DockerScanner

AI-powered Docker container security analysis. Scans a Docker image for CVEs using Trivy, scores and prioritizes vulnerabilities with a scikit-learn ML model, and generates a plain-English security report using the Claude API.

## Pipeline

1. **Trust & Provenance Check** — verifies image origin on Docker Hub, detects typosquatting
2. **CVE Scan** — Trivy scans against NVD, OSV, and GitHub Advisory databases
3. **ML Risk Scoring** — scikit-learn ranks CVEs and produces a 0–100 image risk score
4. **AI Report Generation** — Claude generates a plain-English remediation report

## Requirements

- Python 3.9+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running)
- [Trivy](https://aquasecurity.github.io/trivy/latest/getting-started/installation/) (`scoop install trivy` on Windows)
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=your_key_here

# 3. Run the app
streamlit run app.py
```

## Project Structure

```
app.py                  # Streamlit dashboard
scanner.py              # Trivy integration
ml_scorer.py            # scikit-learn risk scoring pipeline
report_generator.py     # Claude API report generation
trust_checker.py        # Docker Hub provenance & typosquat detection
run_experiments.py      # Batch experiment runner
requirements.txt
.env.example
```

## Course

COT6930 — Generative Intelligence and Software Development Lifecycles  
Florida Atlantic University
