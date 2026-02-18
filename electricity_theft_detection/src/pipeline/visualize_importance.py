import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance():
    importance_path = os.path.join("models", "feature_importance.csv")
    if not os.path.exists(importance_path):
        print(f"Error: {importance_path} not found. Please run training first.")
        return

    df = pd.read_csv(importance_path)
    
    # Premium styling
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=df, 
        palette='viridis',
        hue='Feature',
        legend=False
    )
    
    plt.title('Feature Importance: What Drives Theft Detection?', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Engineered Feature', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join("reports", "feature_importance.png")
    os.makedirs("reports", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Premium visualization saved to {output_path}")

if __name__ == "__main__":
    plot_feature_importance()
