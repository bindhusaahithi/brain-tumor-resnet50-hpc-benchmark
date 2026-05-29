# Contributing

## Development Workflow

1. Install dependencies with `pip install -r requirements.txt`.
2. Make focused changes.
3. Run `ruff check .`.
4. Run `pytest`.
5. Regenerate visualizations if benchmark metrics change.

## Code Style

- Keep scripts thin and put reusable logic in `src/brain_tumor_hpsc/`.
- Store configuration in `config/config.yaml`.
- Do not commit data, model weights, caches, or local environment files.
- Keep medical claims conservative and evidence-backed.
