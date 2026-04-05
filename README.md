# lidar-analytics

Git-tracked code and configuration for the LiDAR analytics workflow.

This repository is intended to sync code between WSL environments on different Windows machines by using Git as the source of truth. Large datasets, checkpoints, logs, and generated experiment outputs stay local and are excluded by `.gitignore`.

## Track in Git

- `experiments/neilson_car/scripts/`
- `experiments/neilson_car/configs/`
- `experiments/neilson_car/*.md`
- `experiments/neilson_car/algorithms.json`
- `experiments/neilson_car/training_manifest*.json`
- `setup/`

## Keep local only

- `data/`
- `logs/`
- `20250410/`
- `DataCity_SMTG/`
- `experiments/neilson_car/runs/`
- `experiments/neilson_car/checkpoints/`
- `experiments/neilson_car/benchmark_runs*/`

## Recommended sync workflow

1. Initialize the repo once on one machine.
2. Add a remote such as GitHub or a private bare Git server.
3. Commit code and config changes.
4. Push from one machine and pull on the other.

Example:

```bash
git add .
git commit -m "Initial code sync baseline"
git remote add origin <your-remote-url>
git push -u origin main
```
