# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in ThinkyLM, please report it
by opening a GitHub issue with the `security` label, or by emailing
the author directly (contact available on GitHub profile).

Please do not report security issues in public issues without first
notifying the maintainer.

## Scope

ThinkyLM is an educational portfolio project. Security considerations include:

- **API input validation**: All API inputs are validated by Pydantic
- **Maximum input lengths**: Enforced to prevent resource exhaustion
- **No authentication**: The API has no authentication — do not expose it publicly without adding authentication

## What This Project Does Not Do

- Store user data
- Connect to external services (no API calls to LLM providers)
- Execute arbitrary code from user input
