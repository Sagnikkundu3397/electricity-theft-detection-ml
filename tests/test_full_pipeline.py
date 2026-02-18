import os
import sys
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.train_pipeline import TrainPipeline

def test_full_pipeline():
    print("\n--- Starting Full Pipeline Test ---")
    start_time = time.time()
    
    try:
        pipeline = TrainPipeline()
        best_name, best_metrics, full_report = pipeline.run()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n[SUCCESS] Pipeline completed in {duration:.2f} seconds.")
        print(f"Best Model: {best_name}")
        print(f"Metrics: {best_metrics}")
        
        if duration > 120:
            print("[WARNING] Pipeline is still taking too long (> 2 mins)")
        else:
            print("[INFO] Pipeline duration is within acceptable limits.")
            
    except Exception as e:
        print(f"[FAILED] Pipeline test failed: {str(e)}")

if __name__ == "__main__":
    test_full_pipeline()
