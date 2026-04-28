# Running code at Mahti

## General requirements

- SUMO_HOME must be set and SUMO tools must be importable
- Tested with use_gui = false (SUMO command line mode)
- Default config path is rp_gnn.yaml, which points to small_net.sumocfg
- Core Python deps include torch, torch-geometric, stable-baselines3, gymnasium
- Extra deps used in scripts include OmegaConf, pandas, matplotlib

## Initial setup at Mahti

### SUMO installation
- Complile at scratch dir (Mahti docs recommend using $TMPDIR)
~~~bash
export ACCOUNT=project_2012159
mkdir -p /scratch/$ACCOUNT/$USER
cd /scratch/$ACCOUNT/$USER
~~~
- Install XercesC library (Apache XML parser)
~~~bash
wget https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.5.tar.gz
tar xzf xerces-c-3.2.5.tar.gz
cd xerces-c-3.2.5
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/scratch/project_2012159/kbocheni/xerces-local
make -j8
make install
~~~

- Building SUMO locally from source
~~~bash
git clone --depth 1 --recursive --shallow-submodules https://github.com/eclipse-sumo/sumo.git
cd sumo
mkdir build && cd build
module purge
module load gcc cmake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/scratch/project_2012159/kbocheni/sumo-local \
  -DXercesC_INCLUDE_DIR=/scratch/project_2012159/kbocheni/xerces-local/include \
  -DXercesC_LIBRARY=/scratch/project_2012159/kbocheni/xerces-local/lib64/libxerces-c.so
make -j8
make install
~~~
- Copying to projappl
~~~bash
rsync -a /scratch/project_2012159/kbocheni/sumo-local/ /projappl/project_2012159/kbocheni_temp/sumo-local/
rsync -a /scratch/project_2012159/kbocheni/xerces-local/ /projappl/project_2012159/kbocheni_temp/xerces-local/
~~~
- Setting environment variables
~~~bash
export SUMO_HOME=/projappl/project_2012159/kbocheni_temp/sumo-local
export XERCES_HOME=/scratch/project_2012159/kbocheni/xerces-local
export SUMO_TOOLS=/projappl/project_2012159/kbocheni_temp/sumo-local/share/sumo/tools
export PYTHONPATH=$SUMO_TOOLS:$PYTHONPATH
export PATH=$SUMO_HOME/bin:$PATH
export LD_LIBRARY_PATH=$XERCES_HOME/lib64:$SUMO_HOME/lib64:$SUMO_HOME/lib:$LD_LIBRARY_PATH
~~~
- Quick check
~~~bash
sumo --version
python -c "import traci, sumolib; print('TraCI OK', traci.__file__)"
# could take time
python -c "import rt_gnn_rl; print('rt_gnn_rl OK')"
python -c "from utils.sumo_bootstrap import _imports; _imports(); print('SUMO bootstrap OK')"
~~~

### Creating venv and installing dependencies
~~~bash
# 1) Go to project
cd /projappl/project_2012159/kbocheni_temp

# 2) Load base modules (names can vary slightly on Mahti)
module purge
module load python-data
module load gcc

# 4) clone the repo 
git clone https://github.com/helsinki-sda-group/smas-rl-gnn.git smas-rl-gnn
cd smas-rl-gnn

# 3) Create virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# 4) Install Python deps
pip install stable-baselines3 gymnasium numpy pandas matplotlib omegaconf

# 5) Install torch and torch-geometric
# CPU-only (simplest/most robust):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
~~~

## Quick sanity check before Slurm

- Check that rp-gnn.yaml contains Mahti-compatible paths:
~~~bash
env:
  sumo_cfg: /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/configs/small_net.sumocfg
<...>
logging:
  run_name: rp_gnn_debug
  out_dir: /scratch/project_2012159/kbocheni/smas-rl-gnn/runs
  model_save_dir: /scratch/project_2012159/kbocheni/smas-rl-gnn/runs/rp_gnn_debug/!saved_models
~~~
- Launch to get the results of a test run in scratch partition.
~~~bash
cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
source .venv/bin/activate

export SUMO_HOME=/projappl/project_2012159/kbocheni_temp/sumo-local
export XERCES_HOME=/scratch/project_2012159/kbocheni/xerces-local
export SUMO_TOOLS=/projappl/project_2012159/kbocheni_temp/sumo-local/share/sumo/tools
export PYTHONPATH=$SUMO_TOOLS:$PYTHONPATH
export PATH=$SUMO_HOME/bin:$PATH
export LD_LIBRARY_PATH=$XERCES_HOME/lib64:$SUMO_HOME/lib64:$SUMO_HOME/lib:$LD_LIBRARY_PATH

python -c "import traci, sumolib; print('TraCI OK')"
python train.py --config configs/rp_gnn.yaml --sumoport 8813
~~~

## Slurm training

### Setting up configurations

- Create separate `rp_gnn_[version].yaml` file per configuration:
~~~bash
cp configs/rp_gnn.yaml configs/rp_gnn_1.yaml
~~~
- Set unique run_name each yaml file, e.g. `rp_gnn_1`, `rp_gnn_2`. For example, in `rp_gnn_1.yaml`
~~~bash
logging:
  run_name: rp_gnn_1
~~~

- Create directory for slurm outputs (is made once):
~~~bash
mkdir -p /scratch/project_2012159/kbocheni/smas-rl-gnn/slurm
~~~

### Batch-generate config variants (no manual copy/rename)

- Use `scripts/generate_ctc_configs.sh` to create variants for all `configs/rp_gnn_*.yaml` files (excluding `rp_gnn.yaml` and `rp_toy.yaml`).
- Generated files get suffix `_ctc.yaml` and automatically update:
  - `env.conflict_resolution: closest_then_capacity`
  - `logging.run_name: <old_run_name>_ctc`
  - `logging.model_save_dir: <logging.out_dir>/<new_run_name>/!saved_models`

~~~bash
cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
chmod +x scripts/generate_ctc_configs.sh

# preview only
scripts/generate_ctc_configs.sh --dry-run

# generate YAML files only
scripts/generate_ctc_configs.sh

# generate and submit all generated configs with slurm/run_train.sbatch
scripts/generate_ctc_configs.sh --submit
~~~

- Optional custom suffix:
~~~bash
scripts/generate_ctc_configs.sh --suffix _ctc_v2
~~~

### Sbatch script
The script `run_train.sbatch` is saved at `smas-rl-gnn\slurm`.
~~~bash
#!/bin/bash
#SBATCH --job-name=rp-train
#SBATCH --account=project_2012159
#SBATCH --partition=small
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.out
#SBATCH --error=/scratch/project_2012159/kbocheni/smas-rl-gnn/slurm/%x-%j.err

set -euo pipefail

REPO=/projappl/project_2012159/kbocheni_temp/smas-rl-gnn
RUNROOT=/scratch/project_2012159/kbocheni/smas-rl-gnn/jobs
CFG=${1:-configs/rp_gnn.yaml}
RUN_NAME=$(grep 'run_name:' "$REPO/$CFG" | awk '{print $2}' || echo "unknown")
JOBDIR=${RUNROOT}/job_${RUN_NAME}_${SLURM_JOB_ID}

mkdir -p "$JOBDIR" 
cp "$REPO/$CFG" "$JOBDIR/rp_gnn.yaml"
cd "$JOBDIR"

module purge
module load python-data
source "$REPO/.venv/bin/activate"

export SUMO_HOME=/projappl/project_2012159/kbocheni_temp/sumo-local
export XERCES_HOME=/projappl/project_2012159/kbocheni_temp/xerces-local
export SUMO_TOOLS=/projappl/project_2012159/kbocheni_temp/sumo-local/share/sumo/tools
export PYTHONPATH="$REPO:$SUMO_TOOLS:${PYTHONPATH:-}"
export PATH="$SUMO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$XERCES_HOME/lib64:$SUMO_HOME/lib64:$SUMO_HOME/lib:${LD_LIBRARY_PATH:-}"

PORT=$((8500 + SLURM_JOB_ID % 1000))
python "$REPO/train.py" --config "$REPO/$CFG" --sumoport "$PORT"
~~~

### Directory layout
~~~bash
/scratch/project_2012159/kbocheni/smas-rl-gnn/
│
├── slurm/                          # created by mkdir -p in sbatch
│   ├── [JOB_NAME]-[SLURM_JOB_ID].out       # stdout (--output)
│   └── [JOB_NAME]-[SLURM_JOB_ID].err       # stderr (--error)
│
├── jobs/                           # created by mkdir -p in sbatch
│   └── job_[RUN_NAME]_[SLURM_JOB_ID]/               # JOBDIR — CWD for the job
│       ├── rp_gnn.yaml        # config snapshot (if you add the cp)
│       ├── monitor.csv             # SB3 training monitor
│       ├── conflicts.log
│       ├── comp_norms.log
│       └── training_metrics_*.log
│
└── runs/                           # created by train.py (from out_dir in yaml)
    └── [RUN_NAME]/               # run_name in yaml
        ├── episode_0000/           # per-episode logs
        ├── episode_0001/
        ├── ...
        └── !saved_models/          # model_save_dir in yaml
            ├── model_episode0_ts0.zip
            └── ...
~~~

### Running batch job at Mahti

**Note**: it is recommended to submit the task with `job_name` the same as `run_name`. In this case, outputs in `slurm` and `jobs` folder will be named consistently.

#### Train
~~~bash
cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
# fresh run
sbatch --job-name=rp_gnn_debug slurm/run_train.sbatch configs/rp_gnn.yaml
sbatch --job-name=rp_gnn_debug_1hop slurm/run_train.sbatch configs/rp_gnn_1hop.yaml
sbatch --job-name=rp_gnn_debug_2hop slurm/run_train.sbatch configs/rp_gnn_2hop.yaml
# continue from the latest checkpoint
sbatch --job-name=rp_gnn_debug_2hop slurm/run_train.sbatch configs/rp_gnn_2hop.yaml continue
# continue and write logs to old job folder
sbatch --job-name=rp_gnn_debug_2hop slurm/run_train.sbatch configs/rp_gnn_2hop.yaml continue 6574582
~~~

Recommended workflow for continuation:
1. Keep same `run_name` and `model_save_dir`.
2. Set `continue_training: true`.
3. Set `ppo.total_timesteps` to extra steps only.
4. Reuse same `JOBDIR` if you want one combined local log set.

#### Evaluation


#### Plotting


1. Plot training results (single run)
-  For the first use, make the helper script executable.
~~~bash
chmod +x /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_single_run.sh
~~~
-  The script looks for the latest matching job directory in `scratch/jobs/job_[RUN_NAME]_[SLURM_JOB_ID]` and writes plots to `projappl`, which is better for keeping the resulting PNG files long-term.
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_single_run.sh rp_gnn_debug_2hop
~~~
-  By default, plots are saved to:
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_single_run/rp_gnn_debug_2hop_job_<SLURM_JOB_ID>/
~~~
-  To plot a specific job instead of the latest one:
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_single_run.sh rp_gnn_debug_2hop --job-id 6574582
~~~
-  Optional arguments:
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_single_run.sh rp_gnn_debug_2hop --ma-window 20 --action-window 10
~~~

2. Plot comparison (ablation across several runs)
-  For the first use, make the helper script executable.
~~~bash
chmod +x /projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh
~~~
-  Configure plotting and aggregation defaults in `ablation_conf_mahti.yaml`.
-  Run comparison by passing run names (latest job for each run name is used automatically):
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh rp_gnn_debug_1hop rp_gnn_debug_2hop
~~~
-  Optionally set display labels in the same order as run names:
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh rp_gnn_debug_1hop rp_gnn_debug_2hop --labels "1 hop,2 hop"
~~~
-  Optionally select explicit job IDs instead of latest jobs:
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh rp_gnn_debug_1hop rp_gnn_debug_2hop --job-ids 6574001,6574582
~~~
-  Optional temporary override (without editing YAML):
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh rp_gnn_debug_1hop rp_gnn_debug_2hop --action-window 20
~~~
-  Outputs are written to:
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/plots_ablation/<RUN1>__<RUN2>/
~~~
-  This folder contains:
~~~bash
ablation_conf_generated.yaml
ablation_results/
action_comparison/
~~~
- Long example to keep:
~~~bash
/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh 1hop-2 1hop-3 2hop-1 2hop-5 1hop_critic-1 1hop_critic-2 1hop_critic-3 1hop_critic-4 1hop_critic-5

/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh 1hop-2 1hop-3 2hop-1 2hop-5

/projappl/project_2012159/kbocheni_temp/smas-rl-gnn/scripts/plot_ablation_runs.sh 1hop-1 1hop-2 1hop-3 1hop-4 1hop-5 2hop-1 2hop-2 2hop-3 2hop-4 2hop-5
~~~

## Github notes

### Updating the repo from remote

~~~bash
cd /projappl/project_2012159/kbocheni_temp/smas-rl-gnn
git pull --rebase origin main
~~~

### Mahti VS code 
- To stop tracking ./venv files in GitHub extension of Mahti VS code:
~~~bash
# ignore venv folders
printf "\n.venv/\nvenv/\n" >> .gitignore

# stop tracking already-tracked venv files
git rm -r --cached .venv venv 2>/dev/null || true

# clean already-staged noise if any
git restore --staged .venv venv 2>/dev/null || true
~~~

- To remove **modified** file from tracking:
~~~bash
git update-index --skip-worktree configs/rp_gnn.yaml
~~~
- Keep **untracked** file locally but remove it from Changes:
1. For Linux:
~~~bash
echo configs/rp_gnn_1hop.yaml >> .git/info/exclude
~~~
2. For Windows:
~~~bash
Add-Content .git/info/exclude "/configs/5-15-long-haul.rou.xml"
~~~

- To update from remote when there are local changes:
~~~bash
git stash push -u
git pull --rebase origin main
git stash pop
~~~
## Interpretation
~~~bash
policy learns:
    estimate P(win | state)

if P(win) high:
    act (even if conflict)
else:
    NOOP
~~~

So the real objective is: maximize alignment(policy, resolver)

Next plan: test all the same with random resolver and with pickup ETA resolver.
Then: more complex resolver. Diagnostic/training target: for each contested task, predict whether ego would win under the resolver.





