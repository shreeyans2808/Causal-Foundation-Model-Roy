#!/usr/bin/env python3
"""
Generic Causal Discovery Evaluation Script

This script evaluates predictions.npy files from the SEA Foundation Model using gcastle metrics.
It automatically handles different dataset formats and provides comprehensive evaluation results.

Usage:
    python evaluate_predictions.py --predictions path/to/predictions.npy [options]

Requirements:
    - gcastle>=1.0.4rc1
    - numpy>=1.21.0
    - pandas>=1.3.0
    - scikit-learn>=1.0.0
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from castle.metrics import MetricsDAG
    HAS_GCASTLE = True
except ImportError:
    HAS_GCASTLE = False
    print("⚠️  Warning: gcastle not found. Install with: pip install gcastle==1.0.4rc1")

try:
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  Warning: scikit-learn not found. Install with: pip install scikit-learn")

def load_predictions(predictions_path):
    """Load and parse predictions.npy file"""
    
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    
    print(f"📁 Loading predictions from: {predictions_path}")
    
    # Load the predictions file
    data = np.load(predictions_path, allow_pickle=True)
    
    # Handle different data structures
    if isinstance(data, dict):
        # Find the dataset key (first non-metadata key)
        dataset_keys = [k for k in data.keys() if isinstance(data[k], (dict, type(data[k])))]
        
        if len(dataset_keys) == 1:
            dataset_key = dataset_keys[0]
            dataset = data[dataset_key]
        elif 'data' in data:
            dataset = data['data']
        else:
            # Try to find the main dataset
            for key in data.keys():
                if isinstance(data[key], dict) and 'pred' in data[key]:
                    dataset = data[key]
                    dataset_key = key
                    break
            else:
                raise ValueError(f"Could not find predictions in file. Available keys: {list(data.keys())}")
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")
    
    # Extract predictions
    if 'pred' not in dataset:
        raise ValueError(f"No 'pred' key found in dataset. Available keys: {list(dataset.keys())}")
    
    predictions_1d = np.array(dataset['pred'])
    if predictions_1d.ndim > 1:
        predictions_1d = predictions_1d.flatten()
    
    # Extract ground truth if available
    ground_truth_1d = None
    if 'true' in dataset:
        ground_truth_1d = np.array(dataset['true'])
        if ground_truth_1d.ndim > 1:
            ground_truth_1d = ground_truth_1d.flatten()
    
    # Extract metadata
    metadata = {
        'time': dataset.get('time', [None])[0] if 'time' in dataset else None,
        'auc': dataset.get('auc', [None])[0] if 'auc' in dataset else None,
        'prc': dataset.get('prc', [None])[0] if 'prc' in dataset else None,
        'dataset_key': dataset_key if 'dataset_key' in locals() else 'unknown'
    }
    
    return predictions_1d, ground_truth_1d, metadata

def reconstruct_adjacency_matrix(predictions_1d):
    """Reconstruct adjacency matrix from 1D predictions"""
    
    n_predictions = len(predictions_1d)
    
    # Calculate number of variables: n*(n-1) = n_predictions
    # Solve: n^2 - n - n_predictions = 0
    import math
    n_vars = int((1 + math.sqrt(1 + 4 * n_predictions)) / 2)
    
    if n_vars * (n_vars - 1) != n_predictions:
        raise ValueError(f"Invalid prediction array size: {n_predictions}. Cannot form square matrix.")
    
    print(f"📊 Reconstructing {n_vars}×{n_vars} adjacency matrix from {n_predictions} predictions")
    
    # Reconstruct matrix
    adjacency_matrix = np.zeros((n_vars, n_vars))
    idx = 0
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                adjacency_matrix[i, j] = predictions_1d[idx]
                idx += 1
    
    return adjacency_matrix

def evaluate_with_thresholds(pred_matrix, gt_matrix=None, thresholds=None):
    """Evaluate predictions at multiple thresholds"""
    
    if thresholds is None:
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    
    if gt_matrix is not None:
        print(f"\n📊 Threshold Evaluation (Ground Truth Available)")
        print("   Thresh | Pred Edges | True Pos | Precision | Recall |   F1   | Specificity")
        print("   -------|------------|----------|-----------|--------|--------|------------")
        
        best_f1 = 0
        best_threshold = 0
        
        for thresh in thresholds:
            binary_pred = (pred_matrix > thresh).astype(int)
            
            # Calculate confusion matrix elements
            tp = np.sum((binary_pred == 1) & (gt_matrix == 1))
            fp = np.sum((binary_pred == 1) & (gt_matrix == 0))
            tn = np.sum((binary_pred == 0) & (gt_matrix == 0))
            fn = np.sum((binary_pred == 0) & (gt_matrix == 1))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            pred_edges = np.sum(binary_pred)
            
            results.append({
                'threshold': thresh,
                'pred_edges': pred_edges,
                'true_positives': tp,
                'false_positives': fp,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'specificity': specificity
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
            
            print(f"   {thresh:6.3f} | {pred_edges:10d} | {tp:8d} | {precision:9.3f} | {recall:6.3f} | {f1:6.3f} | {specificity:11.3f}")
        
        return results, best_threshold, best_f1
    else:
        print(f"\n📊 Threshold Analysis (No Ground Truth)")
        print("   Thresh | Pred Edges | Density | Top 1% | Top 5% | Top 10%")
        print("   -------|------------|---------|--------|--------|--------")
        
        n_vars = pred_matrix.shape[0]
        total_possible = n_vars * (n_vars - 1)
        
        # Calculate percentiles for reference
        flat_pred = pred_matrix[pred_matrix > 0]  # Exclude diagonal zeros
        p99 = np.percentile(flat_pred, 99) if len(flat_pred) > 0 else 0
        p95 = np.percentile(flat_pred, 95) if len(flat_pred) > 0 else 0
        p90 = np.percentile(flat_pred, 90) if len(flat_pred) > 0 else 0
        
        for thresh in thresholds:
            binary_pred = (pred_matrix > thresh).astype(int)
            pred_edges = np.sum(binary_pred)
            density = pred_edges / total_possible
            
            top1_count = np.sum(pred_matrix > p99)
            top5_count = np.sum(pred_matrix > p95)
            top10_count = np.sum(pred_matrix > p90)
            
            results.append({
                'threshold': thresh,
                'pred_edges': pred_edges,
                'density': density
            })
            
            print(f"   {thresh:6.3f} | {pred_edges:10d} | {density:7.4f} | {top1_count:6d} | {top5_count:6d} | {top10_count:7d}")
        
        return results, None, None

def evaluate_with_gcastle(pred_matrix, gt_matrix):
    """Evaluate using gcastle MetricsDAG"""
    
    if not HAS_GCASTLE:
        print("❌ gcastle not available. Skipping gcastle evaluation.")
        return None
    
    try:
        print(f"\n🏰 gcastle MetricsDAG Evaluation")
        print("-" * 50)
        
        # gcastle expects binary matrices, so we'll evaluate at optimal threshold
        # First find optimal threshold
        thresholds = np.linspace(0.01, 0.5, 50)
        best_f1 = 0
        best_thresh = 0.1
        
        for thresh in thresholds:
            binary_pred = (pred_matrix > thresh).astype(int)
            tp = np.sum((binary_pred == 1) & (gt_matrix == 1))
            fp = np.sum((binary_pred == 1) & (gt_matrix == 0))
            fn = np.sum((binary_pred == 0) & (gt_matrix == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        # Evaluate at best threshold
        binary_pred = (pred_matrix > best_thresh).astype(int)
        
        # Initialize MetricsDAG
        metrics = MetricsDAG(gt_matrix, binary_pred)
        
        # Calculate metrics
        results = {
            'threshold_used': best_thresh,
            'fdr': metrics.metrics['fdr'],
            'tpr': metrics.metrics['tpr'],
            'fpr': metrics.metrics['fpr'],
            'shd': metrics.metrics['shd'],
            'nnz': metrics.metrics['nnz'],
            'precision': metrics.metrics['precision'],
            'recall': metrics.metrics['recall'],
            'F1': metrics.metrics['F1'],
            'gscore': metrics.metrics['gscore']
        }
        
        print(f"   Threshold used: {best_thresh:.3f}")
        print(f"   Precision: {results['precision']:.3f}")
        print(f"   Recall: {results['recall']:.3f}")
        print(f"   F1-Score: {results['F1']:.3f}")
        print(f"   FDR (False Discovery Rate): {results['fdr']:.3f}")
        print(f"   TPR (True Positive Rate): {results['tpr']:.3f}")
        print(f"   FPR (False Positive Rate): {results['fpr']:.3f}")
        print(f"   SHD (Structural Hamming Distance): {results['shd']}")
        print(f"   G-Score: {results['gscore']:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ gcastle evaluation failed: {e}")
        return None

def calculate_auroc_auprc(pred_matrix, gt_matrix):
    """Calculate AUROC and AUPRC metrics"""
    
    if not HAS_SKLEARN:
        return None, None
    
    try:
        # Flatten matrices and remove diagonal elements
        n_vars = pred_matrix.shape[0]
        
        pred_flat = []
        gt_flat = []
        
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:  # Skip diagonal
                    pred_flat.append(pred_matrix[i, j])
                    gt_flat.append(gt_matrix[i, j])
        
        pred_flat = np.array(pred_flat)
        gt_flat = np.array(gt_flat)
        
        # Calculate AUROC
        auroc = roc_auc_score(gt_flat, pred_flat)
        
        # Calculate AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(gt_flat, pred_flat)
        auprc = auc(recall_curve, precision_curve)
        
        return auroc, auprc
        
    except Exception as e:
        print(f"⚠️  Could not calculate AUROC/AUPRC: {e}")
        return None, None

def print_summary_statistics(pred_matrix, gt_matrix=None, metadata=None):
    """Print summary statistics"""
    
    print(f"\n📈 Summary Statistics")
    print("-" * 50)
    print(f"   Matrix shape: {pred_matrix.shape}")
    print(f"   Total elements: {pred_matrix.size}")
    print(f"   Non-diagonal elements: {pred_matrix.size - pred_matrix.shape[0]}")
    print(f"   Min prediction: {np.min(pred_matrix):.6f}")
    print(f"   Max prediction: {np.max(pred_matrix):.6f}")
    print(f"   Mean prediction: {np.mean(pred_matrix):.6f}")
    print(f"   Median prediction: {np.median(pred_matrix[pred_matrix > 0]):.6f}")
    print(f"   Std prediction: {np.std(pred_matrix):.6f}")
    
    if gt_matrix is not None:
        print(f"\n   Ground Truth:")
        print(f"   GT shape: {gt_matrix.shape}")
        print(f"   GT edges: {np.sum(gt_matrix)}")
        print(f"   GT sparsity: {np.sum(gt_matrix)/(gt_matrix.shape[0]*(gt_matrix.shape[1]-1)):.4f}")
        
        # Calculate AUROC/AUPRC
        auroc, auprc = calculate_auroc_auprc(pred_matrix, gt_matrix)
        if auroc is not None:
            print(f"   AUROC: {auroc:.3f}")
            print(f"   AUPRC: {auprc:.3f}")
    
    if metadata:
        print(f"\n   Metadata:")
        if metadata['time']:
            try:
                exec_time = float(metadata['time'])
                print(f"   Execution time: {exec_time:.2f}s")
            except:
                print(f"   Execution time: {metadata['time']}")
        if metadata['dataset_key']:
            print(f"   Dataset: {metadata['dataset_key']}")

def save_results(results, output_path):
    """Save evaluation results to file"""
    
    try:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to: {output_path}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate SEA Foundation Model predictions')
    parser.add_argument('--predictions', '-p', required=True, 
                       help='Path to predictions.npy file')
    parser.add_argument('--ground_truth', '-gt', 
                       help='Path to ground truth adjacency matrix (.npy file)')
    parser.add_argument('--output', '-o', 
                       help='Path to save evaluation results (JSON format)')
    parser.add_argument('--thresholds', nargs='+', type=float,
                       help='Custom thresholds to evaluate (default: auto-generated)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("SEA FOUNDATION MODEL EVALUATION")
    print("=" * 80)
    
    # Load predictions
    try:
        predictions_1d, gt_1d, metadata = load_predictions(args.predictions)
        pred_matrix = reconstruct_adjacency_matrix(predictions_1d)
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        sys.exit(1)
    
    # Load ground truth if provided
    gt_matrix = None
    if args.ground_truth:
        try:
            gt_matrix = np.load(args.ground_truth)
            print(f"✅ Ground truth loaded: {gt_matrix.shape}")
            
            # Align dimensions if needed
            if gt_matrix.shape != pred_matrix.shape:
                print(f"⚠️  Dimension mismatch: GT {gt_matrix.shape} vs Pred {pred_matrix.shape}")
                if gt_matrix.shape[0] > pred_matrix.shape[0]:
                    gt_matrix = gt_matrix[:pred_matrix.shape[0], :pred_matrix.shape[1]]
                    print(f"   Trimmed GT to: {gt_matrix.shape}")
                else:
                    print(f"   Cannot align - GT smaller than predictions")
                    gt_matrix = None
        except Exception as e:
            print(f"⚠️  Could not load ground truth: {e}")
    
    # Print summary statistics
    print_summary_statistics(pred_matrix, gt_matrix, metadata)
    
    # Evaluate at thresholds
    threshold_results, best_thresh, best_f1 = evaluate_with_thresholds(
        pred_matrix, gt_matrix, args.thresholds)
    
    # gcastle evaluation
    gcastle_results = None
    if gt_matrix is not None:
        gcastle_results = evaluate_with_gcastle(pred_matrix, gt_matrix)
    
    # Compile results
    all_results = {
        'metadata': metadata,
        'summary_stats': {
            'matrix_shape': pred_matrix.shape,
            'min_pred': float(np.min(pred_matrix)),
            'max_pred': float(np.max(pred_matrix)),
            'mean_pred': float(np.mean(pred_matrix)),
            'std_pred': float(np.std(pred_matrix))
        },
        'threshold_results': threshold_results,
        'best_threshold': best_thresh,
        'best_f1': best_f1,
        'gcastle_results': gcastle_results
    }
    
    if gt_matrix is not None:
        auroc, auprc = calculate_auroc_auprc(pred_matrix, gt_matrix)
        all_results['auroc'] = auroc
        all_results['auprc'] = auprc
    
    # Save results
    if args.output:
        save_results(all_results, args.output)
    
    print(f"\n🎯 EVALUATION COMPLETE!")
    if best_f1:
        print(f"   Best F1: {best_f1:.3f} at threshold {best_thresh}")
    print("=" * 80)

if __name__ == "__main__":
    main()
