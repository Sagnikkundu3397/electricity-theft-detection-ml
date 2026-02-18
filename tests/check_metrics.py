import json
import sys
import os

def check_metrics(metrics_file="models/metrics.json", threshold=0.2):
    print(f"--- Checking Model Metrics (Threshold F1 > {threshold}) ---")
    
    if not os.path.exists(metrics_file):
        print(f"[ERROR] Metrics file {metrics_file} not found!")
        sys.exit(1)
        
    try:
        with open(metrics_file, "r") as f:
            data = json.load(f)
            
        best_model = data.get("best_model")
        metrics = data.get("metrics", {})
        f1_score = metrics.get("F1-Score", 0)
        
        print(f"Best Model: {best_model}")
        print(f"F1-Score: {f1_score}")
        
        if f1_score < threshold:
            print(f"[FAILED] F1-Score {f1_score} is below threshold {threshold}!")
            sys.exit(1)
            
        print("[SUCCESS] Metrics validated successfully.")
        
    except Exception as e:
        print(f"[ERROR] Failed to check metrics: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    check_metrics()
