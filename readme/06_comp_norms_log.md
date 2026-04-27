# `comp_norms.log` Metrics Reference

This document describes every column in root-level `comp_norms.log`.

Primary sources:
- `sumo_rl_rs/logging/rp_logger_callback.py` (file writing)
- `rt_gnn_rl/policy/actor_critic.py` (`pop_comp_norm_stats()` and stat accumulation)
- `plot_comp_norms.py` (subset of required plotting columns)

## What this log is

`comp_norms.log` is a **rollout-level diagnostic log** for actor competitor-fusion behavior.

Rows are written at callback rollout end (`_on_rollout_end`) by pulling
`gnn_ac.pop_comp_norm_stats()` from the policy.

## Header

```text
ts,ep,norm_h,norm_z,p_has_comp,logit_base,logit_comp,logit_ind,bias_base,norm_w_h,norm_w_c,norm_w_s,norm_w_d,attn_entropy,max_attn,ratio_comp_base,ratio_comp_gap,norm_u,std_comp,mean_num_comp,max_num_comp,mean_score,std_score,count
```

## Aggregation bases used in code

From `EgoActorCritic.pop_comp_norm_stats()`:

- `count` = number of candidate-task logits accumulated (`_comp_log_count`)
- `comp_count` = number of candidate-task entries that had competitors (`_comp_log_comp_count`)
- `robot_count` = number of robot-level contributions for ratio/gap stats (`_comp_log_robot_count`)

Different fields are normalized by different bases (`count`, `comp_count`, or `robot_count`).

## Column definitions

| Column | Meaning | Computation / source |
|---|---|---|
| `ts` | Training timesteps at logging moment. | `self.num_timesteps` in callback. |
| `ep` | Episode index at logging moment. | `self.ep_idx` in callback. |
| `count` | Number of candidate items aggregated. | Integer `_comp_log_count`. |
| `norm_h` | Mean norm of candidate embedding $h_t$. | `sums[0] / count`. |
| `norm_z` | Mean norm of competitor-context embedding $z_t$. | `sums[1] / count`. |
| `p_has_comp` | Fraction of candidates that had competitors. | `sums[2] / count` where indicator is 1 if competitor set non-empty. |
| `logit_base` | Mean base actor logit before competitor correction. | `sums[3] / count`. |
| `logit_comp` | Mean competitor correction logit. | `sums[4] / count`. |
| `logit_ind` | Mean indicator bias term (`comp_bias * ind_t`). | `sums[5] / count`. |
| `bias_base` | Mean contribution of actor-head bias term. | `sums[6] / count`. |
| `norm_w_h` | Norm of `actor_head.weight` (broadcast in aggregate). | `sums[7] / count`. |
| `norm_w_c` | Norm of `comp_head.weight` (broadcast in aggregate). | `sums[8] / count`. |
| `norm_w_s` | Norm of `score_comp.weight` (broadcast in aggregate). | `sums[9] / count`. |
| `norm_w_d` | Absolute value of `comp_bias` (broadcast in aggregate). | `sums[10] / count`. |
| `attn_entropy` | Mean entropy of competitor attention weights. | `sums[11] / comp_count`. |
| `max_attn` | Mean max attention weight across competitor sets. | `sums[12] / comp_count`. |
| `ratio_comp_base` | Strength ratio of competitor correction vs base logits. | `sums[13] / robot_count`, where per-robot value is $\frac{\operatorname{mean}|logit_{comp}|}{\operatorname{mean}|logit_{base}| + 10^{-8}}$. |
| `ratio_comp_gap` | Competitor strength vs base top-2 separation. | `sums[14] / robot_count`, where per-robot value is $\frac{\operatorname{mean}|logit_{comp}|}{|top1-top2| + 10^{-8}}$. |
| `norm_u` | Mean norm of transformed competitor features $u_{tc}$. | `sums[15] / comp_count`. |
| `std_comp` | Mean per-robot std of competitor logits. | `sums[16] / robot_count` using `std(unbiased=False)`. |
| `mean_num_comp` | Mean number of competitors per candidate (when competitors exist). | `sums[17] / comp_count`. |
| `max_num_comp` | Mean per-robot maximum competitor count among its candidates. | `sums[18] / robot_count`. |
| `mean_score` | Mean raw attention score (`score_comp`) over competitor sets. | `sums[19] / comp_count`. |
| `std_score` | Mean std of raw attention scores over competitor sets. | `sums[20] / comp_count` with `unbiased=False`. |

## How competitor context is formed

In `_compute_comp_context(...)` for each candidate task $t$:

1. Build competitor list `comps` from graph neighbors of task node `t`, excluding ego node and candidate task set.
2. Build feature vectors through `phi_comp([h_t, x_c, a_tc]) -> u_tc`.
3. Score each competitor with `score_comp(u_tc) -> s_tc`.
4. Attention weights: $\alpha = \operatorname{softmax}(s_{tc})$.
5. Context vector: $z_t = \sum_c \alpha_c u_{tc}$.

Diagnostics (`attn_entropy`, `max_attn`, `norm_u`, `mean_score`, `std_score`, `num_comp`) are extracted from these tensors.

## When values can be near zero

- If competitor fusion is disabled, many competitor-specific fields become zero.
- If few candidates have competitors (`p_has_comp` low), `attn_*` and `num_comp` stats may be small or less stable.
- Early training can show tiny `logit_comp` and low ratios while `comp_head` weights are near zero.

## Plot usage

`plot_comp_norms.py --log comp_norms.log` requires the subset:
- `ts`
- `logit_base`
- `logit_comp`
- `ratio_comp_base`
- `ratio_comp_gap`
- `attn_entropy`
- `max_attn`

and produces:
- `logit_base.png`
- `logit_comp.png`
- `ratio_comp_base.png`
- `ratio_comp_gap.png`
- `attn_entropy.png`
- `max_attn.png`
