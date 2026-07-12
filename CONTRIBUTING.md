# Contributing to ThinkyLM

Thank you for your interest in contributing to ThinkyLM.

## Getting Started

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make your changes
5. Run tests: `python -m pytest tests/ -v`
6. Run linting: `ruff check thinkylm/ tests/`
7. Submit a pull request

## Code Style

- Use type hints on all functions
- Write docstrings for all public functions and classes
- Follow the existing code structure
- Run `ruff check` before committing

## Data Contributions

Only contribute data that is:
- Original content you wrote yourself
- Public domain (pre-1928, or clearly marked)
- Under a permissive licence (MIT, Apache 2.0, CC0, CC-BY)

Do NOT contribute:
- Copyrighted text
- LLM-generated text (OpenAI TOS restrictions)
- Personal or private information

## Test Requirements

All new features must include tests.
Tests must pass on CPU without CUDA.
Tests should complete in under 10 seconds each.

## What Not to Contribute

- Pretrained model weights from other projects
- Large datasets (use the manifest format instead)
- Any files that should be in .gitignore
