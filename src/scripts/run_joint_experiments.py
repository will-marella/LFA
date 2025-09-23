import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.experiment.simulation import simulate_topic_disease_data
from src.gibbs_sampler import run_cgs_experiment
from src.mfvi_sampler import run_mfvi_experiment


PCGS_METRIC_COLUMNS = [
    'pcgs_num_iterations',
    'pcgs_beta_mae',
    'pcgs_beta_pearson_corr',
    'pcgs_theta_mae',
    'pcgs_theta_pearson_corr',
    'pcgs_run_time',
    'pcgs_r_hat_beta',
    'pcgs_r_hat_theta',
    'pcgs_r_hat_overall',
    'pcgs_converged'
]

MFVI_METRIC_COLUMNS = [
    'mfvi_num_iterations',
    'mfvi_final_elbo',
    'mfvi_beta_correlation',
    'mfvi_theta_correlation',
    'mfvi_beta_mse',
    'mfvi_theta_mse',
    'mfvi_run_time',
    'mfvi_mean_elbo_delta_tail',
    'mfvi_converged'
]

METHOD_METRIC_COLUMNS = {
    'pcgs': PCGS_METRIC_COLUMNS,
    'mfvi': MFVI_METRIC_COLUMNS
}

METHOD_PRIMARY_COLUMN = {
    'pcgs': 'pcgs_run_time',
    'mfvi': 'mfvi_run_time'
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired PCGS and MFVI experiments")
    parser.add_argument('--config_csv', required=True, help='Path to configuration table')
    parser.add_argument('--config_index', type=int,
                        help='Row index in configuration table (0-based)')
    parser.add_argument('--setup_id', type=str,
                        help='Identifier of row in configuration table')
    parser.add_argument('--results_csv', required=True,
                        help='Path to shared results CSV')
    parser.add_argument('--seed_start', type=int, help='Override seed range start (inclusive)')
    parser.add_argument('--seed_end', type=int, help='Override seed range end (inclusive)')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Optional log file (otherwise stdout)')
    parser.add_argument('--method', choices=['pcgs', 'mfvi'],
                        default=os.environ.get('METHOD'),
                        help='Which algorithm to run for each configuration')
    return parser.parse_args()


def setup_logging(log_file: Optional[str]) -> None:
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    handlers: List[logging.Handler]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers = [
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        handlers = [logging.StreamHandler(sys.stdout)]

    logging.basicConfig(level=logging.INFO, format=log_format,
                        datefmt=date_format, handlers=handlers)


def load_config(args: argparse.Namespace) -> pd.Series:
    config_df = pd.read_csv(args.config_csv)

    if args.setup_id is not None:
        matches = config_df[config_df['setup_id'] == args.setup_id]
        if matches.empty:
            raise ValueError(f"No configuration found for setup_id={args.setup_id}")
        if len(matches) > 1:
            raise ValueError(f"Multiple configurations found for setup_id={args.setup_id}")
        return matches.iloc[0]

    if args.config_index is None:
        raise ValueError("Either --setup_id or --config_index must be provided")

    if args.config_index < 0 or args.config_index >= len(config_df):
        raise IndexError(f"config_index {args.config_index} out of range (0..{len(config_df)-1})")

    return config_df.iloc[args.config_index]


def get_value(config_row: pd.Series, key: str, default):
    if key in config_row and not pd.isna(config_row[key]):
        return config_row[key]
    return default


def get_bool(config_row: pd.Series, key: str, default: bool) -> bool:
    value = get_value(config_row, key, default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'1', 'true', 'yes'}:
            return True
        if lowered in {'0', 'false', 'no'}:
            return False
    return bool(value)


def parse_seed_range(config_row: pd.Series, seed_start: Optional[int],
                     seed_end: Optional[int]) -> Iterable[int]:
    if seed_start is None:
        seed_start = int(get_value(config_row, 'seed_start', 1))
    if seed_end is None:
        seed_end = int(get_value(config_row, 'seed_end', seed_start))
    if seed_end < seed_start:
        raise ValueError(f"seed_end ({seed_end}) must be >= seed_start ({seed_start})")
    return range(seed_start, seed_end + 1)


def method_already_completed(results_csv: str, setup_id: str, seed: int, method: str) -> bool:
    if not os.path.exists(results_csv):
        return False

    df = pd.read_csv(results_csv)
    matches = df[(df['setup_id'] == setup_id) & (df['seed'] == seed)]
    if matches.empty:
        return False

    primary_column = METHOD_PRIMARY_COLUMN[method]
    if primary_column not in matches.columns:
        return False

    return matches[primary_column].notna().any()


def release_lock(lock_path: str) -> None:
    if os.path.exists(lock_path):
        os.remove(lock_path)


def safely_upsert_row(base_row: Dict, method_updates: Dict, results_csv: str) -> None:
    import time

    lock_path = results_csv + '.lock'
    Path(os.path.dirname(results_csv) or '.').mkdir(parents=True, exist_ok=True)

    while os.path.exists(lock_path):
        time.sleep(0.1)

    try:
        Path(lock_path).touch()
        if os.path.exists(results_csv):
            df = pd.read_csv(results_csv)
        else:
            df = pd.DataFrame()

        required_columns = set(base_row.keys()) | set(method_updates.keys()) | {'setup_id', 'seed'}
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan

        mask = (df['setup_id'] == base_row['setup_id']) & (df['seed'] == base_row['seed']) if not df.empty else pd.Series([], dtype=bool)

        if mask.any():
            idx = df[mask].index[0]
            for col, value in base_row.items():
                df.at[idx, col] = value
            for col, value in method_updates.items():
                df.at[idx, col] = value
        else:
            new_row = {col: np.nan for col in df.columns}
            new_row.update(base_row)
            new_row.update(method_updates)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(results_csv, index=False)
    finally:
        release_lock(lock_path)


def prepare_pcgs_metrics(metrics: Dict) -> Dict:
    return {
        'pcgs_num_iterations': metrics['num_iterations'],
        'pcgs_beta_mae': metrics['beta_mae'],
        'pcgs_beta_pearson_corr': metrics['beta_pearson_corr'],
        'pcgs_theta_mae': metrics['theta_mae'],
        'pcgs_theta_pearson_corr': metrics['theta_pearson_corr'],
        'pcgs_run_time': metrics['run_time'],
        'pcgs_r_hat_beta': metrics.get('r_hat_beta'),
        'pcgs_r_hat_theta': metrics.get('r_hat_theta'),
        'pcgs_r_hat_overall': metrics.get('r_hat_overall'),
        'pcgs_converged': metrics.get('converged')
    }


def prepare_mfvi_metrics(metrics: Dict) -> Dict:
    return {
        'mfvi_num_iterations': metrics['num_iterations'],
        'mfvi_final_elbo': metrics['final_elbo'],
        'mfvi_beta_correlation': metrics['beta_correlation'],
        'mfvi_theta_correlation': metrics['theta_correlation'],
        'mfvi_beta_mse': metrics['beta_mse'],
        'mfvi_theta_mse': metrics['theta_mse'],
        'mfvi_run_time': metrics['run_time'],
        'mfvi_mean_elbo_delta_tail': metrics.get('mean_elbo_delta_tail'),
        'mfvi_converged': metrics.get('converged')
    }


def build_base_row(config_row: pd.Series, seed: int, params: Dict) -> Dict:
    base = {
        'setup_id': config_row['setup_id'],
        'seed': seed,
        'timestamp': datetime.utcnow().isoformat(),
        **params,
        'pcgs_num_chains': int(get_value(config_row, 'pcgs_num_chains', 4)),
        'pcgs_max_iterations': int(get_value(config_row, 'pcgs_max_iterations', 10000)),
        'pcgs_window_size': int(get_value(config_row, 'pcgs_window_size', 500)),
        'pcgs_r_hat_threshold': float(get_value(config_row, 'pcgs_r_hat_threshold', 1.0)),
        'pcgs_post_convergence_samples': int(get_value(config_row, 'pcgs_post_convergence_samples', 100)),
        'mfvi_max_iterations': int(get_value(config_row, 'mfvi_max_iterations', 5000)),
        'mfvi_convergence_threshold': float(get_value(config_row, 'mfvi_convergence_threshold', 0.0)),
        'mfvi_fixed_iterations': get_bool(config_row, 'mfvi_fixed_iterations', True),
        'mfvi_delta_tail_window': int(get_value(config_row, 'mfvi_delta_tail_window', 50))
    }
    return base


def run_single_seed(config_row: pd.Series, seed: int, method: str) -> Tuple[Dict, Dict]:
    params = {
        'M': int(config_row['M']),
        'D': int(config_row['D']),
        'K': int(config_row['K']),
        'topic_prob': float(config_row['topic_prob']),
        'nontopic_prob': float(get_value(config_row, 'nontopic_prob', 0.01)),
        'alpha_sim': float(config_row['alpha_sim'])
    }

    logging.info(f"Simulating data for seed {seed} with params {params}")

    W, z, beta, theta = simulate_topic_disease_data(
        seed=seed,
        M=params['M'],
        D=params['D'],
        K=params['K'],
        topic_associated_prob=params['topic_prob'],
        nontopic_associated_prob=params['nontopic_prob'],
        alpha=np.ones(params['K'] + 1) * params['alpha_sim'],
        include_healthy_topic=True
    )

    base_row = build_base_row(config_row, seed, params)

    if method == 'pcgs':
        metrics = run_pcgs_pipeline(config_row, W, beta, theta, seed)
        method_updates = prepare_pcgs_metrics(metrics)
    elif method == 'mfvi':
        metrics = run_mfvi_pipeline(config_row, W, beta, theta)
        method_updates = prepare_mfvi_metrics(metrics)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return base_row, method_updates


def run_pcgs_pipeline(config_row: pd.Series, W, beta, theta, seed: int) -> Dict:
    pcgs_kwargs = dict(
        alpha=np.ones(int(config_row['K']) + 1) / 10,
        num_topics=int(config_row['K']) + 1,
        num_chains=int(get_value(config_row, 'pcgs_num_chains', 4)),
        max_iterations=int(get_value(config_row, 'pcgs_max_iterations', 10000)),
        beta=beta,
        theta=theta,
        window_size=int(get_value(config_row, 'pcgs_window_size', 500)),
        r_hat_threshold=float(get_value(config_row, 'pcgs_r_hat_threshold', 1.0)),
        post_convergence_samples=int(get_value(config_row, 'pcgs_post_convergence_samples', 100)),
        base_seed=seed
    )

    logging.info(
        "Running PCGS with num_chains=%s, max_iterations=%s", pcgs_kwargs['num_chains'], pcgs_kwargs['max_iterations']
    )

    _, metrics = run_cgs_experiment(W=W, **pcgs_kwargs)
    return metrics


def run_mfvi_pipeline(config_row: pd.Series, W, beta, theta) -> Dict:
    mfvi_kwargs = dict(
        alpha=np.ones(int(config_row['K']) + 1) / 10,
        num_topics=int(config_row['K']) + 1,
        beta=beta,
        theta=theta,
        max_iterations=int(get_value(config_row, 'mfvi_max_iterations', 5000)),
        convergence_threshold=float(get_value(config_row, 'mfvi_convergence_threshold', 0.0)),
        fixed_iterations=get_bool(config_row, 'mfvi_fixed_iterations', True),
        delta_tail_window=int(get_value(config_row, 'mfvi_delta_tail_window', 50))
    )

    logging.info(
        "Running MFVI with max_iterations=%s, fixed_iterations=%s", mfvi_kwargs['max_iterations'], mfvi_kwargs['fixed_iterations']
    )

    _, metrics = run_mfvi_experiment(W=W, **mfvi_kwargs)
    return metrics


def main():
    args = parse_args()
    if args.method is None:
        raise ValueError("Method not specified. Provide --method or set METHOD environment variable.")

    log_file = args.log_file
    if log_file is None and 'SLURM_JOB_ID' in os.environ:
        log_dir = Path('experiments/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        job_id = os.environ['SLURM_JOB_ID']
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        log_file = log_dir / f"joint_experiment_{args.method}_{job_id}_{task_id}.log"

    setup_logging(str(log_file) if log_file else None)

    config_row = load_config(args)
    setup_id = config_row['setup_id']
    logging.info(f"Loaded configuration for setup_id={setup_id}, method={args.method}")

    results_csv = args.results_csv
    seeds = parse_seed_range(config_row, args.seed_start, args.seed_end)

    for seed in seeds:
        logging.info(f"Processing seed {seed}")
        if method_already_completed(results_csv, setup_id, seed, args.method):
            logging.info(f"{args.method.upper()} results already present for setup_id={setup_id}, seed={seed}; skipping")
            continue

        try:
            base_row, method_updates = run_single_seed(config_row, seed, args.method)
            safely_upsert_row(base_row, method_updates, results_csv)
            logging.info(f"Successfully wrote {args.method.upper()} results for setup_id={setup_id}, seed={seed}")
        except Exception:
            logging.exception(f"Experiment failed for setup_id={setup_id}, seed={seed}")
            raise


if __name__ == '__main__':
    main()
