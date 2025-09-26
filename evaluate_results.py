import numpy as np
from castle.metrics import MetricsDAG

def evaluate_causal_discovery():
    """Evaluate causal discovery results using MetricsDAG"""
    
    # Load the predictions and ground truth
    pred_data = np.load('outputs/test_final/predictions.npy', allow_pickle=True).item()
    graph_data = np.load('outputs/test_final/graph.npy', allow_pickle=True)
    
    print("=== SEA Causal Foundation Model - Evaluation Results ===")
    print(f"Ground truth graph shape: {graph_data.shape}")
    
    # Extract the nested data
    nested_data = pred_data['data']
    predictions_raw = np.array(nested_data['pred'])
    true_graph_raw = np.array(nested_data['true'])
    
    print(f"Predictions raw shape: {predictions_raw.shape}")
    print(f"True values raw shape: {true_graph_raw.shape}")
    
    # Flatten if needed
    predictions_flat = predictions_raw.flatten()
    true_graph_flat = true_graph_raw.flatten()
    
    print(f"Predictions shape (flattened): {predictions_flat.shape}")
    print(f"True values shape (flattened): {true_graph_flat.shape}")
    
    # Handle the case where diagonal elements are excluded (9900 = 100*100 - 100)
    if len(predictions_flat) == 9900:
        # 100x100 matrix without diagonal elements
        n_nodes = 100
        # Reconstruct full adjacency matrix
        predictions_matrix = np.zeros((n_nodes, n_nodes))
        true_matrix = np.zeros((n_nodes, n_nodes))
        
        # Fill off-diagonal elements
        mask = ~np.eye(n_nodes, dtype=bool)  # All positions except diagonal
        predictions_matrix[mask] = predictions_flat
        true_matrix[mask] = true_graph_flat
    else:
        # Standard case with all elements
        n_nodes = int(np.sqrt(len(predictions_flat)))
        predictions_matrix = predictions_flat.reshape(n_nodes, n_nodes)
        true_matrix = true_graph_flat.reshape(n_nodes, n_nodes)
    
    print(f"Reshaped to adjacency matrices: {n_nodes}x{n_nodes}")
    print(f"Ground truth edges: {np.sum(true_matrix)} out of {n_nodes * n_nodes} possible")
    print(f"Prediction statistics: min={predictions_matrix.min():.6f}, max={predictions_matrix.max():.6f}, mean={predictions_matrix.mean():.6f}")
    
    # Create binary predictions using different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    print("\n=== Evaluation Results by Threshold ===")
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        # Convert predictions to binary adjacency matrix
        pred_binary = (predictions_matrix > threshold).astype(int)
        
        # Use MetricsDAG for evaluation
        metrics = MetricsDAG(true_matrix, pred_binary)
        
        # Get metrics
        precision = metrics.precision
        recall = metrics.recall
        f1_score = metrics.f1
        shd = metrics.shd  # Structural Hamming Distance
        
        print(f"\nThreshold: {threshold}")
        print(f"  Predicted edges: {np.sum(pred_binary)}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  SHD: {shd}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold
    
    print(f"\n=== Best Performance ===")
    print(f"Best F1-Score: {best_f1:.4f} at threshold {best_threshold}")
    
    # Additional analysis with best threshold
    best_pred = (predictions_matrix > best_threshold).astype(int)
    best_metrics = MetricsDAG(true_matrix, best_pred)
    
    print(f"\n=== Detailed Analysis (Threshold: {best_threshold}) ===")
    print(f"True Positives: {np.sum((true_matrix == 1) & (best_pred == 1))}")
    print(f"False Positives: {np.sum((true_matrix == 0) & (best_pred == 1))}")
    print(f"True Negatives: {np.sum((true_matrix == 0) & (best_pred == 0))}")
    print(f"False Negatives: {np.sum((true_matrix == 1) & (best_pred == 0))}")
    
    # Show some high-confidence predictions
    print(f"\n=== Top 10 Highest Confidence Predictions ===")
    flat_indices = np.argsort(predictions_flat)[-10:]
    for i, idx in enumerate(flat_indices[::-1]):  # Show in descending order
        row, col = idx // n_nodes, idx % n_nodes
        pred_val = predictions_flat[idx]
        true_val = true_graph_flat[idx]
        status = "✓ Correct" if (pred_val > best_threshold) == bool(true_val) else "✗ Wrong"
        print(f"  {i+1:2d}. Edge {row}→{col}: Prob={pred_val:.6f}, True={true_val}, {status}")
    
    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'predictions_matrix': predictions_matrix,
        'true_matrix': true_matrix
    }

if __name__ == "__main__":
    results = evaluate_causal_discovery()
    print("\n=== Evaluation Complete ===")
    print("The SEA Causal Foundation Model evaluation has finished successfully!")
