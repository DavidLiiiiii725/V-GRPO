# V-GRPO

V-GRPO (`Task-Global Variance Normalization for GRPO`) reference implementation scaffold for MATH reproduction.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Baseline GRPO run (smoke)

```bash
python scripts/train_grpo.py --run_name grpo_baseline --total_steps 10
```

### V-GRPO run (smoke)

```bash
python scripts/train_vgrpo.py --run_name vgrpo --total_steps 10
```

### Unit tests

```bash
pytest -q
```

### Build comparison table

```bash
python scripts/eval_math.py --output results/tables/grpo.json
python scripts/eval_math.py --output results/tables/vgrpo.json
python scripts/compare_results.py --grpo results/tables/grpo.json --vgrpo results/tables/vgrpo.json
```

## Structure

- `config.py`: centralized hyperparameters
- `src/variance_tracker.py`: per-task EMA variance tracker
- `src/advantage.py`: GRPO and V-GRPO advantage functions
- `src/reward.py`: rule-based MATH reward
- `src/grpo_trainer.py`: trainer core logic for denominator swap and clipping
- `src/data.py`: dataset loading and level grouping
- `src/eval.py`: per-level evaluation
- `src/metrics.py`: mechanism diagnostic metrics
- `OPEN_QUESTIONS.md`: ambiguity decisions
