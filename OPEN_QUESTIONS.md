# OPEN QUESTIONS / DECISIONS

1. **Table 1 vs Table 2 mismatch**  
   Decision: prioritize the setup aligned with Section 3/Table 1 style reporting for reproduction; log any divergence explicitly.

2. **Task type labeling**  
   Decision: single-task run uses `task_type="math"`, while code keeps multi-task interfaces.

3. **EMA initialization**  
   Decision: cold-start uses first observed batch variance directly (no blending with `init_var`).

4. **`sigma_G = 0` behavior in baseline GRPO**  
   Decision: keep stable denominator with epsilon in formula; resulting zero-centered equal rewards produce zero advantages.

5. **EMA update timing (pre/post advantage)**  
   Decision: default to update-before-compute in trainer (`update_before_compute=True`) and keep flag configurable.

6. **Reward scale (0/1 vs 0/100)**  
   Decision: default to `0/1`, configurable via `config.py`.

7. **KL penalty inclusion**  
   Decision: retain KL coefficient in config path to keep parity between baseline and V-GRPO settings.
