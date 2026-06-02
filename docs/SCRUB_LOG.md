# Scrub Log — `foundation-model/`

Record of the sanitization pass that turned the imported SST code into a clean,
standalone personal project. Done June 2026.

## Customer data — DELETED
- `results/ruddenklau_evaluation/` (entire directory: a real farm's lab
  measurements across 4 fields, per-field result JSON, and the analysis summary).
- `scripts/evaluate_on_ruddenklau.py` (loaded that farm's data from a private bucket).

No customer or field data remains anywhere in the tree.

## Infrastructure identifiers — REPLACED with placeholders / env-driven config
| Was | Now |
|---|---|
| GCP project `soilmetrix-project-1` | `your-gcp-project` |
| GCS buckets `soilmetrix-research`, `soilmetrix-ml-data`, `preprod_automation`, `soilmetrix_automation` | `your-research-bucket` / `your-eval-bucket` |
| Service account `vishnu-service@…iam.gserviceaccount.com` | removed (now uses Application Default Credentials) |
| Artifact registry `…/soilmetrix-project-1/cloud-run-source-deploy/…` | `…/your-gcp-project/sst/…` |
| Company repo URL `github.com/SoilMetrix-Inc/soilmetrix-platform` | `github.com/your-username/soil-state-transformer` |
| WandB projects `soilmetrix-sst`, `soilmetrix-foundation-model` | `soil-state-transformer` |
| Maintainer `SoilMetrix Team` | `Luke Robinson` |
| Absolute local path into the old monorepo (in a results JSON) | stripped to a relative path |

## Production-pipeline coupling — REMOVED
- References to the former `services/data-ml/` production code, `model_predictor.py`,
  the production ensemble baseline, and "production bucket" evaluation were removed
  or genericized from `config.yml`, `requirements.txt`, `README.md`, `CLAUDE.md`,
  and notebooks `05` / `10`.

## Docs — REWRITTEN
- `foundation-model/README.md` and `foundation-model/CLAUDE.md` rewritten from
  scratch as personal-project docs (no branch policy, no company repo, no
  production integration).
- `config.yml` `gcp:` block trimmed to `project_id` + `research_bucket` placeholders.

## Verification
```bash
# Returns nothing:
grep -rIiE 'soilmetrix|ruddenklau|bw-[0-9]|preprod|vishnu|gserviceaccount|SoilMetrix-Inc' foundation-model/
```

## Git history
After scrubbing, git history was re-initialized from a clean tree so that no
earlier commit retains the customer data or company identifiers.
