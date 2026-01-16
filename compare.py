import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def load_metrics(run_dir):
    metrics_file = os.path.join(run_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Warning: No metrics.csv found in {run_dir}")
        return None
    return pd.read_csv(metrics_file)

def generate_comparison(baseline_dir, bottleneck_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df_base = load_metrics(baseline_dir)
    df_bottle = load_metrics(bottleneck_dir)
    
    if df_base is None or df_bottle is None:
        print("Could not load metrics from one or both directories.")
        return

    # Add model labels
    df_base['Model'] = 'Baseline'
    df_bottle['Model'] = 'Bottleneck'
    
    # Combined dataframe
    df = pd.concat([df_base, df_bottle], ignore_index=True)
    
    # 1. Loss Comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step', y='val_loss', hue='Model')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'))
    plt.close()
    
    # 2. Accuracy Comparison (2-hop vs 3-hop)
    # Check if we have these columns
    if 'val_acc' in df.columns and 'ood_acc' in df.columns:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.lineplot(data=df, x='step', y='val_acc', hue='Model')
        plt.title('2-Hop Test Accuracy (In-Distribution)')
        
        plt.subplot(1, 2, 2)
        sns.lineplot(data=df, x='step', y='ood_acc', hue='Model')
        plt.title('3-Hop OOD Accuracy (Generalization)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
        plt.close()
        
        # Generalization Gap
        df['gen_gap'] = df['val_acc'] - df['ood_acc']
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='step', y='gen_gap', hue='Model')
        plt.title('Generalization Gap (2-hop - 3-hop acc)')
        plt.savefig(os.path.join(output_dir, 'generalization_gap.png'))
        plt.close()

    # 3. Error Analysis Breakdown (at final step)
    error_cols = [c for c in df.columns if 'error' in c]
    if error_cols:
        final_base = df_base.iloc[-1][error_cols]
        final_bottle = df_bottle.iloc[-1][error_cols]
        
        error_data = []
        for col in error_cols:
            error_data.append({'Error Type': col, 'Rate': final_base[col], 'Model': 'Baseline'})
            error_data.append({'Error Type': col, 'Rate': final_bottle[col], 'Model': 'Bottleneck'})
            
        df_err = pd.DataFrame(error_data)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_err, x='Error Type', y='Rate', hue='Model')
        plt.title('Final Error Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'error_types_comparison.png'))
        plt.close()

    # 4. Generate Report
    report = f"""# Comparison Report
    
## Final Metrics (Step {df_base['step'].max()})

### Baseline
- Test Acc (2-hop): {df_base.iloc[-1].get('val_acc', 'N/A'):.4f}
- OOD Acc (3-hop): {df_base.iloc[-1].get('ood_acc', 'N/A'):.4f}

### Bottleneck
- Test Acc (2-hop): {df_bottle.iloc[-1].get('val_acc', 'N/A'):.4f}
- OOD Acc (3-hop): {df_bottle.iloc[-1].get('ood_acc', 'N/A'):.4f}

## Findings
[This section to be filled manually after inspection]
"""
    with open(os.path.join(output_dir, 'comparison_report.md'), 'w') as f:
        f.write(report)
        
    print(f"Comparison generated in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run1', type=str, required=True, help='Path to baseline run directory')
    parser.add_argument('--run2', type=str, required=True, help='Path to bottleneck run directory')
    parser.add_argument('--output_dir', type=str, default='outputs/comparison')
    args = parser.parse_args()
    
    generate_comparison(args.run1, args.run2, args.output_dir)
