import json

# Load your two JSON files
with open("/Users/shreeyansarora/Downloads/SEA_Amartya/Causal-Foundation-Model-Roy/Causal-Foundation-Model-Baseline-master/hepar2_gcastle_evaluation.json") as f:
    original = json.load(f)

with open("/Users/shreeyansarora/Downloads/SEA_Amartya/Causal-Foundation-Model-Roy/Causal-Foundation-Model-Baseline-master/hepar2_evaluation_original.json") as f:
    predicted = json.load(f)

def compare_values(orig_val, pred_val):
    try:
        # Convert to float if possible
        orig_val = float(orig_val)
        pred_val = float(pred_val)
        # Simple "accuracy" as 1 - relative difference
        if orig_val == 0 and pred_val == 0:
            accuracy = 1.0
        elif orig_val == 0:
            accuracy = 0.0
        else:
            accuracy = max(0.0, 1 - abs(orig_val - pred_val) / abs(orig_val))
        return accuracy
    except:
        return None

# Compare metadata
metadata_accuracy = {}
for key in original["metadata"]:
    if key in predicted["metadata"]:
        metadata_accuracy[key] = compare_values(original["metadata"][key], predicted["metadata"][key])

# Compare summary_stats
summary_accuracy = {}
for key in original["summary_stats"]:
    if key in predicted["summary_stats"]:
        summary_accuracy[key] = compare_values(original["summary_stats"][key], predicted["summary_stats"][key])

# Compare threshold results (list of dicts)
threshold_accuracy = []
for orig_thr, pred_thr in zip(original["threshold_results"], predicted["threshold_results"]):
    thr_acc = {}
    for key in orig_thr:
        if key in pred_thr:
            thr_acc[key] = compare_values(orig_thr[key], pred_thr[key])
    threshold_accuracy.append(thr_acc)

# Compare gcastle results
gcastle_accuracy = {}
for key in original["gcastle_results"]:
    if key in predicted["gcastle_results"]:
        gcastle_accuracy[key] = compare_values(original["gcastle_results"][key], predicted["gcastle_results"][key])

# Compare overall AUROC and AUPRC
overall_accuracy = {
    "auroc": compare_values(original.get("auroc"), predicted.get("auroc")),
    "auprc": compare_values(original.get("auprc"), predicted.get("auprc"))
}

# Build final JSON
accuracy_json = {
    "metadata_accuracy": metadata_accuracy,
    "summary_stats_accuracy": summary_accuracy,
    "threshold_results_accuracy": threshold_accuracy,
    "gcastle_results_accuracy": gcastle_accuracy,
    "overall_accuracy": overall_accuracy
}

# Save to file
with open("accuracy_comparison.json", "w") as f:
    json.dump(accuracy_json, f, indent=2)

print("Accuracy comparison JSON saved as accuracy_comparison.json")
