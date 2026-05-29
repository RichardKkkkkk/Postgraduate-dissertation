# Agent.md

This is a PyTorch Vision Transformer MVP project.

The user develops this repository from both a laptop and a desktop. Treat the
repository documents as the shared memory between Codex sessions.

## Project rules

- Do not modify the core VisionTransformer architecture unless explicitly asked.
- Use CIFAR-10 for initial MVP experiments.
- Keep datasets under `data/` and do not commit them.
- Keep checkpoints under `checkpoints/` and do not commit them.
- Save plots under `results/figures/`.
- Update README when adding runnable scripts.
- When changing a shared or unified training interface, also update README with:
  - the supported CLI parameters
  - which parameters are model-specific
  - the default values the user should expect
- Use the conda environment `vit_research` for project development.
- Do not create a `.venv` for this project.
- Remind the user when the current changes form a reasonable commit checkpoint, suggest a clear commit message, but do not commit unless explicitly asked.

## Cross-device workflow

- At the start of a session, read `docs/PROJECT_LOG.md` first, then skim `docs/DEVELOPMENT_MAP.md` and `README.md` before making changes.
- Use `docs/PROJECT_LOG.md` for short chronological progress notes: what changed, what was learned, what is blocked, and what should happen next.
- Use `docs/DEVELOPMENT_MAP.md` for the research roadmap and reasoning behind major direction changes.
- Use `docs/LEARNING_NOTES.md` for explanations of PyTorch syntax, tensor shapes, and implementation details.
- If local code and docs disagree, call out the mismatch before continuing. Update the log with the mismatch and the chosen resolution.
- Before ending a session with meaningful work, update `docs/PROJECT_LOG.md` with the latest status and next action.
- When switching machines, remind the user to `git status`, commit useful changes, push on the current machine, then pull on the other machine.
- Do not rely on chat history as the only project memory; important context must live in tracked docs.

## Learning mode

- After modifying code, explain every changed file.
- Explain important PyTorch syntax with examples.
- Include tensor shapes where relevant.
- Update `docs/LEARNING_NOTES.md` after meaningful changes.
- When the user proposes a new research or development idea, record the idea and reasoning in `docs/DEVELOPMENT_MAP.md`.

## Test commands

- Run `python vit.py` after changing the model.
- Run a short smoke test before long training.
- For docs-only changes, no training smoke test is required; verify the changed docs and `git status`.
