# Agent.md

This is a PyTorch Vision Transformer MVP project.

## Project rules

- Do not modify the core VisionTransformer architecture unless explicitly asked.
- Use CIFAR-10 for initial MVP experiments.
- Keep datasets under `data/` and do not commit them.
- Keep checkpoints under `checkpoints/` and do not commit them.
- Save plots under `results/figures/`.
- Update README when adding runnable scripts.
- Use the conda environment `vit_research` for project development.
- Do not create a `.venv` for this project.
- Remind the user when the current changes form a reasonable commit checkpoint, suggest a clear commit message, but do not commit unless explicitly asked.

## Learning mode

- After modifying code, explain every changed file.
- Explain important PyTorch syntax with examples.
- Include tensor shapes where relevant.
- Update `docs/LEARNING_NOTES.md` after meaningful changes.
- When the user proposes a new research or development idea, record the idea and reasoning in `docs/DEVELOPMENT_MAP.md`.

## Test commands

- Run `python vit.py` after changing the model.
- Run a short smoke test before long training.
