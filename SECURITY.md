# Security policy

## Supported version

Security fixes target the latest commit on `main`. Experimental branches and
locally modified upstream checkouts are not supported releases.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue. Use the
repository's **Security** tab to open a private GitHub Security Advisory:

<https://github.com/robodreamer/embodied-policy-lab/security/advisories/new>

Include a minimal reproduction, affected commit, impact, and any suggested
mitigation. Do not include live credentials or private model-access tokens.

## Scope and trust boundaries

Embodied Policy Lab runs third-party simulators, checkpoints, setup scripts,
and local HTTP/WebSocket services. Review upstream licenses and code before
running them, keep the dashboard bound to loopback, and treat downloaded model
artifacts as untrusted inputs. The network audit reports observed destinations;
it is evidence, not a firewall or sandbox.

This project is a research workbench and must not be used as a safety layer for
physical robot operation.
