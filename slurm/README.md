# Slurm Job Submission Scripts

Quick reference for submitting training jobs to Mahti for multiple method variants.

## Quick Start

### From your local machine (test/preview):

```powershell
cd slurm
.\slurm_submit.ps1 1hop_rnd 2hop 1hop_ctc -DryRun
```

### On Mahti (actual submission):

```bash
cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
bash slurm/slurm_submit.sh 1hop_rnd 2hop 1hop_ctc
```

### Baseline evaluation on Mahti:

```bash
cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
sbatch --job-name=rp-eval-1hop slurm/run_eval_baselines.sbatch configs/rp_gnn_1hop-1.yaml
```

## Usage

### Method Patterns

Methods are matched against config filenames using patterns. **Base methods (without variant) match ONLY base configs**, variants are separate:

- `1hop_rnd` → finds all `rp_gnn_1hop-*_rnd.yaml` (1hop with rnd variant)
- `1hop_ctc` → finds all `rp_gnn_1hop-*_ctc.yaml` (1hop with ctc variant)
- `1hop` → finds all `rp_gnn_1hop-[0-9]*.yaml` (only base, NO variants)
- `2hop` → finds all `rp_gnn_2hop-[0-9]*.yaml` (only base, NO variants)
- `1hop_1hop_critic` → finds all `rp_gnn_1hop_1hop_critic-[0-9]*.yaml` (only base)
- `1hop_1hop_critic_rnd` → finds all `rp_gnn_1hop_1hop_critic-*_rnd.yaml` (only rnd variant)

### Examples

Submit all variant runs of base methods:
```bash
bash slurm/slurm_submit.sh 1hop 2hop 1hop_1hop_critic
```

Submit only specific variants:
```bash
bash slurm/slurm_submit.sh 1hop_ctc 1hop_rnd 2hop_ctc 2hop_rnd
```

Continue training for specific methods (requires prior run):
```bash
bash slurm/slurm_submit.sh 1hop --mode continue
```

Dry-run to preview what would be submitted:
```bash
bash slurm/slurm_submit.sh 1hop_rnd 2hop --dry-run
```

## Options

| Option | Description |
|--------|-------------|
| `--mode new` (default) | Start fresh training |
| `--mode continue` | Resume from checkpoint (if available) |
| `--dry-run` | Preview commands without submitting |

## Script Files

- `slurm_submit.sh` - Main Bash script for Mahti
- `slurm_submit.ps1` - PowerShell helper for local testing/previewing
- `run_train.sbatch` - Slurm job template
- `run_eval_baselines.sbatch` - Slurm job template for `eval_baselines.py`

## Features

- Automatic config discovery by method pattern
- Batch submission of multiple configs
- Config snapshots stored in job directory
- Automatic stdout/stderr capture per job
- Job naming based on run_name from config
- Dry-run mode for safety

## Notes

- Episode directories will be automatically pruned per-episode (configured in all YAML files as `prune_episode_dir_after_metrics: true`)
- Each job gets a unique directory under `/scratch/project_2012159/kbocheni/smas-rl-gnn/jobs/`
- Monitor jobs with: `squeue -u kbocheni`
- Check logs in: `/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/`
