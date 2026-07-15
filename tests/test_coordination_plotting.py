import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


HEADER_PREFIX = (
    "pol seed ts | rew cap step dln wait trav comp nsv | "
    "pku pkr obs obsr pkv pkvr mwt cmp cmr anp anpr mtt pnc pncr | "
    "noop overld mcand cne_fr cne_mn dstep macmr msd ovrlap shared"
)

COORD_BLOCK = " | ecr unop ncpr psur offpr"


class CoordinationPlottingTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.python = sys.executable

    def _write_eval_log(self, path: Path, rows, include_coord=None):
        include_coord = include_coord or []
        header = HEADER_PREFIX
        if include_coord:
            header += " | " + " ".join(include_coord)

        with path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for row in rows:
                parts = [
                    f"{row.get('pol', 'eval')} {int(row['seed'])} {int(row['ts'])}",
                    f"{row.get('rew', 0.0):.3f} 0 0 {row.get('dln', 0.0):.3f} {row.get('wait', 0.0):.3f} {row.get('trav', 0.0):.3f} {row.get('comp', 0.0):.3f} 0",
                    "0 0 0 0 0 0 0 0 0 0 0 0 0 0",
                    "0 0 0 0 0 0 0 0 0 0",
                ]
                if include_coord:
                    coord_vals = []
                    for c in include_coord:
                        coord_vals.append(f"{float(row.get(c, 0.0)):.3f}")
                    parts.append(" ".join(coord_vals))
                f.write(" | ".join(parts) + "\n")

    def _write_training_log(self, path: Path, rows, include_coord=True):
        header = HEADER_PREFIX + (COORD_BLOCK if include_coord else "")
        with path.open("w", encoding="utf-8") as f:
            f.write(header + "\n")
            for row in rows:
                parts = [
                    f"{row.get('pol', 'train')} {int(row['seed'])} {int(row['ts'])}",
                    f"{row.get('rew', 0.0):.3f} 0 0 {row.get('dln', 0.0):.3f} {row.get('wait', 0.0):.3f} {row.get('trav', 0.0):.3f} {row.get('comp', 0.0):.3f} 0",
                    "0 0 0 0 0 0 0 0 0 0 0 0 0 0",
                    "0 0 0 0 0 0 0 0 0 0",
                ]
                if include_coord:
                    parts.append(
                        f"{row.get('ecr', 0.0):.3f} {row.get('unop', 0.0):.3f} {row.get('ncpr', 0.0):.3f} {row.get('psur', 0.0):.3f} {row.get('offpr', 0.0):.3f}"
                    )
                f.write(" | ".join(parts) + "\n")

    def _run(self, args, cwd=None):
        proc = subprocess.run(
            [self.python, *args],
            cwd=str(cwd or self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout

    def test_1_all_coordination_columns_single_log(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            log_path = td_path / "evaluation_metrics.log"
            out_dir = td_path / "out"
            rows = [
                {"seed": 1, "ts": 100, "rew": 1.0, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.2, "unop": 0.4, "ncpr": 0.6, "psur": 0.8, "offpr": 0.0},
                {"seed": 2, "ts": 100, "rew": 1.2, "wait": 0.2, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.4, "unop": 0.6, "ncpr": 0.8, "psur": 1.0, "offpr": 0.2},
                {"seed": 1, "ts": 200, "rew": 2.0, "wait": 0.3, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.3, "unop": 0.5, "ncpr": 0.7, "psur": 0.9, "offpr": 0.1},
                {"seed": 2, "ts": 200, "rew": 2.2, "wait": 0.4, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.5, "unop": 0.7, "ncpr": 0.9, "psur": 1.0, "offpr": 0.2},
            ]
            self._write_eval_log(log_path, rows, include_coord=["ecr", "unop", "ncpr", "psur", "offpr"])

            code, out = self._run(["plot_eval_results.py", str(log_path), "--output-dir", str(out_dir), "--ma", "2"])
            self.assertEqual(code, 0, msg=out)

            coord_dir = out_dir / "coord"
            expected_pngs = [
                "empty_cand_rate_vs_ts.png",
                "unforced_noop_rate_vs_ts.png",
                "nonconf_prop_rate_vs_ts.png",
                "prop_survival_rate_vs_ts.png",
                "offprop_assign_rate_vs_ts.png",
                "coord_metrics_vs_ts.png",
            ]
            for name in expected_pngs:
                self.assertTrue((coord_dir / name).exists(), f"Missing {name}")

            expected_csv = coord_dir / "empty_cand_rate_vs_ts_data.csv"
            self.assertTrue(expected_csv.exists())
            df = pd.read_csv(expected_csv)
            row100 = df[df["ts"] == 100].iloc[0]
            self.assertAlmostEqual(float(row100["mean"]), 0.3, places=6)
            self.assertAlmostEqual(float(row100["std"]), 0.141421356, places=5)
            self.assertEqual(int(row100["count"]), 2)

    def test_2_partial_coordination_columns(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            log_path = td_path / "evaluation_metrics.log"
            out_dir = td_path / "out"
            rows = [
                {"seed": 1, "ts": 100, "rew": 1.0, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.2, "unop": 0.4, "psur": 0.8},
                {"seed": 2, "ts": 100, "rew": 1.2, "wait": 0.2, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.4, "unop": 0.6, "psur": 1.0},
            ]
            self._write_eval_log(log_path, rows, include_coord=["ecr", "unop", "psur"])

            code, out = self._run(["plot_eval_results.py", str(log_path), "--output-dir", str(out_dir)])
            self.assertEqual(code, 0, msg=out)

            coord_dir = out_dir / "coord"
            self.assertTrue((coord_dir / "empty_cand_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "unforced_noop_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "prop_survival_rate_vs_ts.png").exists())
            self.assertFalse((coord_dir / "nonconf_prop_rate_vs_ts.png").exists())
            self.assertFalse((coord_dir / "offprop_assign_rate_vs_ts.png").exists())

    def test_3_old_log_without_coordination_columns(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            log_path = td_path / "evaluation_metrics.log"
            out_dir = td_path / "out"
            rows = [
                {"seed": 1, "ts": 100, "rew": 1.0, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4},
                {"seed": 2, "ts": 100, "rew": 1.2, "wait": 0.2, "trav": 0.2, "comp": 0.3, "dln": 0.4},
            ]
            self._write_eval_log(log_path, rows, include_coord=[])

            code, out = self._run(["plot_eval_results.py", str(log_path), "--output-dir", str(out_dir)])
            self.assertEqual(code, 0, msg=out)
            self.assertTrue((out_dir / "reward_vs_timesteps.png").exists())

    def test_4_compare_mode_mean_run_coordination(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            log1 = td_path / "evaluation_metrics_run1.log"
            log2 = td_path / "evaluation_metrics_run2.log"
            out_dir = td_path / "compare_out"

            rows1 = [
                {"seed": 1, "ts": 100, "rew": 1.0, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.2, "unop": 0.3, "ncpr": 0.4, "psur": 0.5, "offpr": 0.0},
                {"seed": 1, "ts": 200, "rew": 1.5, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.3, "unop": 0.4, "ncpr": 0.5, "psur": 0.6, "offpr": 0.1},
            ]
            rows2 = [
                {"seed": 1, "ts": 100, "rew": 1.1, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.25, "unop": 0.35, "ncpr": 0.45, "psur": 0.55, "offpr": 0.05},
                {"seed": 1, "ts": 200, "rew": 1.6, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.35, "unop": 0.45, "ncpr": 0.55, "psur": 0.65, "offpr": 0.15},
            ]
            self._write_eval_log(log1, rows1, include_coord=["ecr", "unop", "ncpr", "psur", "offpr"])
            self._write_eval_log(log2, rows2, include_coord=["ecr", "unop", "ncpr", "psur", "offpr"])

            code, out = self._run(
                [
                    "plot_eval_results.py",
                    "--compare-log",
                    f"run1={log1}",
                    "--compare-log",
                    f"run2={log2}",
                    "--mean-run",
                    "--output-dir",
                    str(out_dir),
                ]
            )
            self.assertEqual(code, 0, msg=out)

            coord_dir = out_dir / "coord"
            self.assertTrue((coord_dir / "empty_cand_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "empty_cand_rate_vs_ts_data.csv").exists())
            self.assertTrue((coord_dir / "coord_metrics_vs_ts.png").exists())

    def test_5_ablation_workflow_coordination_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model_a = td_path / "methodA-1"
            model_b = td_path / "methodB-1"
            model_a.mkdir(parents=True)
            model_b.mkdir(parents=True)

            rows_a = [
                {"seed": 1, "ts": 100, "rew": 1.0, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.2, "unop": 0.3, "ncpr": 0.4, "psur": 0.5, "offpr": 0.0},
                {"seed": 1, "ts": 200, "rew": 1.2, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.3, "unop": 0.4, "ncpr": 0.5, "psur": 0.6, "offpr": 0.1},
            ]
            rows_b = [
                {"seed": 1, "ts": 100, "rew": 0.9, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.25, "unop": 0.35, "ncpr": 0.45, "psur": 0.55, "offpr": 0.05},
                {"seed": 1, "ts": 200, "rew": 1.1, "wait": 0.1, "trav": 0.2, "comp": 0.3, "dln": 0.4, "ecr": 0.35, "unop": 0.45, "ncpr": 0.55, "psur": 0.65, "offpr": 0.15},
            ]
            self._write_training_log(model_a / "training_metrics_test.log", rows_a, include_coord=True)
            self._write_training_log(model_b / "training_metrics_test.log", rows_b, include_coord=True)

            out_root = td_path / "out"
            conf_path = td_path / "ablation_conf_test.yaml"
            conf_text = f"""
model_dirs:
  - {model_a}
  - {model_b}
read_params_from_yaml: false
exp_params:
  use_xy_pickup: [false, false]
  normalize_features: [false, false]
  logit_temperature: [1.0, 1.0]
  noop_init: [0.0, 0.0]
  freeze_noop_logit: [false, false]
training_only: true
plot_comp_eval: true
plot_comp_eval_ma: 2
plot_raw_eval: true
plot_raw_eval_std: true
plot_ma_std: true
mean_runs: true
coordination_plot_std: true
coordination_plot_ma: 2
k_eval: 2
output_dir: {out_root / 'ablation_results'}
output_file: ablation_aggregated.log
output_csv: ablation_aggregated.csv
coordination_output_dir: {out_root / 'coord_cmp'}
"""
            conf_path.write_text(conf_text, encoding="utf-8")

            code, out = self._run(["aggregate_ablation_results.py", "--config", str(conf_path)])
            self.assertEqual(code, 0, msg=out)

            agg_csv = out_root / "ablation_results" / "ablation_aggregated.csv"
            agg_log = out_root / "ablation_results" / "ablation_aggregated.log"
            coord_dir = out_root / "coord_cmp"

            self.assertTrue(agg_csv.exists())
            self.assertTrue(agg_log.exists())
            self.assertTrue(coord_dir.exists())

            df = pd.read_csv(agg_csv)
            self.assertTrue((df["metric"] == "ecr").any())
            self.assertTrue((df["metric"] == "unop").any())
            self.assertTrue((df["metric"] == "ncpr").any())
            self.assertTrue((df["metric"] == "psur").any())
            self.assertTrue((df["metric"] == "offpr").any())

            log_text = agg_log.read_text(encoding="utf-8")
            self.assertIn("ecr:", log_text)
            self.assertIn("unop:", log_text)
            self.assertIn("ncpr:", log_text)
            self.assertIn("psur:", log_text)
            self.assertIn("offpr:", log_text)

            self.assertTrue((coord_dir / "empty_cand_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "empty_cand_rate_vs_ts_data.csv").exists())
            self.assertTrue((coord_dir / "unforced_noop_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "nonconf_prop_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "prop_survival_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "offprop_assign_rate_vs_ts.png").exists())
            self.assertTrue((coord_dir / "coord_metrics_vs_ts.png").exists())


if __name__ == "__main__":
    unittest.main()
