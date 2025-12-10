"""
Hybrid Hypothesis Validation: Neural Network + Selective CatBoost

This script implements a two-stage validation approach:
1. Quick NN prediction for all hypotheses (~0.01s each)
2. Full CatBoost validation for top candidates only

Speedup: 10-100x faster than full validation
Trade-off: NN approximation for low-scoring hypotheses

Usage:
    python run_hybrid_validation.py --sample-size 3000 --top-k 10000

Author: AI Assistant
Date: 2024
"""

import argparse
import pickle
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get script directory for relative paths
script_dir = Path(__file__).parent
DATA_DIR = script_dir / 'data'


def load_data():
    """Load training data and test features."""
    logger.info("Loading data...")
    
    with open(DATA_DIR / 'train_features_noisy.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(DATA_DIR / 'train_labels.pkl', 'rb') as f:
        y_train = pickle.load(f)
    
    with open(DATA_DIR / 'test_features_noisy.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Handle both DataFrame and numpy array formats
    if hasattr(train_data, 'columns'):
        # It's a DataFrame - extract feature columns only
        feature_cols = [col for col in train_data.columns if col not in ['account', 'label']]
        X_train = train_data[feature_cols].values
        X_test = test_data[feature_cols].values
        logger.info(f"Loaded DataFrames, extracted {len(feature_cols)} feature columns")
    else:
        # Already numpy arrays
        X_train = train_data
        X_test = test_data
    
    logger.info(f"Loaded X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}")
    return X_train, y_train, X_test


def load_hypotheses():
    """Load generated hypotheses."""
    logger.info("Loading hypotheses...")
    
    # Check for both possible filenames
    hypothesis_file = DATA_DIR / 'generated_hypotheses.pkl'
    if not hypothesis_file.exists():
        hypothesis_file = DATA_DIR / 'all_hypotheses.pkl'
    
    if not hypothesis_file.exists():
        raise FileNotFoundError(
            f"Hypotheses file not found: {DATA_DIR / 'generated_hypotheses.pkl'} or {DATA_DIR / 'all_hypotheses.pkl'}\n"
            f"Please run notebook 05 Cell 6 to generate hypotheses."
        )
    
    with open(hypothesis_file, 'rb') as f:
        hypotheses = pickle.load(f)
    
    logger.info(f"Loaded {len(hypotheses):,} hypotheses from {hypothesis_file.name}")
    return hypotheses


def save_results(results, output_dir=DATA_DIR):
    """Save validation results to disk."""
    logger.info("Saving results...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as pickle
    pkl_path = output_dir / 'hybrid_validation_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved pickle: {pkl_path}")
    
    # Save as CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / 'hybrid_validation_results.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV: {csv_path}")
    
    # Save timestamped backup
    backup_path = output_dir / f'hybrid_validation_results_{timestamp}.pkl'
    with open(backup_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved backup: {backup_path}")
    
    # Save summary statistics
    validated = [r for r in results if r['prediction_method'] == 'catboost_validated']
    predicted = [r for r in results if r['prediction_method'] == 'neural_network']
    
    summary = {
        'total_hypotheses': len(results),
        'catboost_validated': len(validated),
        'nn_predicted': len(predicted),
        'validation_ratio': len(validated) / len(results),
        'top_f1_score': max(r['f1_score'] for r in results),
        'mean_f1_score': np.mean([r['f1_score'] for r in results]),
        'median_f1_score': np.median([r['f1_score'] for r in results]),
        'timestamp': timestamp
    }
    
    summary_path = output_dir / 'hybrid_validation_summary.txt'
    with open(summary_path, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    logger.info("="*60)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid hypothesis validation with NN + CatBoost'
    )
    parser.add_argument(
        '--sample-size', 
        type=int, 
        default=2000,
        help='Number of hypotheses to validate for NN training (default: 2000)'
    )
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=5000,
        help='Number of top predictions to fully validate (default: 5000)'
    )
    parser.add_argument(
        '--nn-model-path',
        type=str,
        default=str(DATA_DIR / 'hypothesis_predictor.pkl'),
        help='Path to save/load NN model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DATA_DIR),
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("HYBRID HYPOTHESIS VALIDATION")
    logger.info("="*80)
    logger.info(f"Sample size for NN training: {args.sample_size:,}")
    logger.info(f"Top K for full validation: {args.top_k:,}")
    logger.info(f"NN model path: {args.nn_model_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*80 + "\n")
    
    # Load data
    X_train, y_train, X_test = load_data()
    hypotheses = load_hypotheses()
    
    # Check if we have enough hypotheses
    total_validation_needed = args.sample_size + args.top_k
    if total_validation_needed > len(hypotheses):
        logger.warning(
            f"Warning: sample_size ({args.sample_size}) + top_k ({args.top_k}) = "
            f"{total_validation_needed} exceeds total hypotheses ({len(hypotheses)})"
        )
        logger.info("Adjusting parameters...")
        args.sample_size = min(args.sample_size, len(hypotheses) // 3)
        args.top_k = min(args.top_k, len(hypotheses) - args.sample_size)
        logger.info(f"New sample_size: {args.sample_size}, top_k: {args.top_k}")
    
    # Run hybrid validation
    from hypothesis_predictor_nn import hybrid_validation_workflow
    
    results = hybrid_validation_workflow(
        hypotheses=hypotheses,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        sample_size=args.sample_size,
        top_k=args.top_k,
        nn_predictor_path=args.nn_model_path
    )
    
    # Save results
    summary = save_results(results, args.output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total time saved compared to full validation:")
    nn_only_count = len(hypotheses) - args.sample_size - args.top_k
    time_saved_seconds = nn_only_count * 3.5  # Assuming 3.5s per CatBoost validation
    time_saved_hours = time_saved_seconds / 3600
    logger.info(f"  Estimated: {time_saved_hours:.1f} hours")
    logger.info("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nValidation interrupted by user")
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        raise
