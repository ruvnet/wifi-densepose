# PyPI release runbook — `wifi-densepose` + `ruview`

Operations doc for the `.github/workflows/pip-release.yml` CI workflow.

## Auth

Production uses the GitHub Actions secret `PYPI_API_TOKEN`. It is a
project token issued by the rUv PyPI account with upload scope for both
`wifi-densepose` and `ruview`.

TestPyPI uses a separate `TESTPYPI_API_TOKEN` secret issued by
test.pypi.org. PyPI and TestPyPI accounts and tokens are independent.

## Refreshing the token

The canonical copy of the token lives in GCP Secret Manager,
project `cognitum-20260110`, entry name `PYPI_TOKEN`. To push a
fresh copy into GitHub Actions:

```bash
gcloud secrets versions access latest \
    --secret=PYPI_TOKEN \
    --project=cognitum-20260110 \
  | tr -d '\r\n\xef\xbb\xbf' \
  | gh secret set PYPI_API_TOKEN --repo ruvnet/RuView
```

The `tr` step strips any BOM / CRLF that PowerShell pipes or
Windows editors may have introduced — without it, twine fails with
`UnicodeEncodeError: 'latin-1' codec can't encode character '﻿'`.

## Triggering a release

Two paths:

- **Tag push** — `git tag v2.X.Y-pip && git push origin v2.X.Y-pip` —
  publishes the v2 wheel matrix. `v1.99.0-pip` triggers the tombstone
  job instead.
- **Manual dispatch** — `gh workflow run pip-release.yml --ref <branch>
  -f target=v2-wheels -f publish_to=pypi`. Use `publish_to=testpypi`
  for a dry-run target if a TestPyPI token is also set as
  `TESTPYPI_API_TOKEN`.

## Release-day sequence

Per ADR-117 §7.3, the tombstone publishes first so it claims the
"current" slot in pip's resolver:

1. `git tag v1.99.0-pip && git push origin v1.99.0-pip` →
   tombstone live at `https://pypi.org/project/wifi-densepose/1.99.0/`
2. Verify: `pip install wifi-densepose==1.99.0; python -c "import
   wifi_densepose"` → ImportError with migration URL.
3. Confirm `archive/v1/data/proof/expected_features_v2.sha256` is
   committed and non-empty. Production publishing fails closed without it.
4. `git tag v2.0.0-pip && git push origin v2.0.0-pip` → the v2
   `wifi-densepose` wheel matrix and matching `ruview` wheel/sdist are
   published together. Their versions and dependency pin are checked in CI.
5. Verify both `https://pypi.org/project/wifi-densepose/2.0.0/` and
   `https://pypi.org/project/ruview/2.0.0/`.

## Off-loop manual gates

- **Q3** (ADR-117 §11.3) — generate
  `archive/v1/data/proof/expected_features_v2.sha256` from the v2 Rust
  pipeline before a production v2 publish. The workflow enforces this gate.
- **OIDC Trusted Publisher** — not used. The workflow is token-based;
  this is a deliberate choice to keep the secret refresh entirely in
  GCP. If the project migrates to OIDC later, remove `password:`
  from `pypa/gh-action-pypi-publish` calls and add the publisher
  registration on pypi.org.
