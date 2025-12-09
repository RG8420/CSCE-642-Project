import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = "hyperparameter_analysis_results.csv"

# 1. Load Data
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_path} not found.")
    raise

# 2. Select Metrics
# Grouping variables for easier interpretation in the heatmap
COLUMNS = [
    'gamma', 'lam', 'vf_coef', 'ent_coef',
    'final_mean_reward', 'final_policy_loss', 'final_value_loss'
]
df_analysis = df[COLUMNS].copy()

# 3. Calculate Correlation
correlation_matrix = df_analysis.corr(method='pearson')

# 3. Plot Heatmap with increased font size
plt.figure(figsize=(12, 10))
ax = sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt=".2f", 
    linewidths=.5, 
    # Adjust cbar_kws to include a label
    cbar_kws={'label': 'Pearson Correlation Coefficient'} 
)

# Set the font size for the x and y axis tick labels (set to 16 for better fit)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)

# Get the colorbar object to change its label size (set to 18 as requested)
cbar = ax.collections[0].colorbar
cbar.set_label('Pearson Correlation Coefficient', fontsize=18)
cbar.ax.tick_params(labelsize=14) # Also adjust tick labels on the colorbar

# plt.title('Correlation Heatmap: Hyperparameters vs. Performance Metrics', fontsize=20)
plt.tight_layout()

# Save the plot
plot_filename = "hyperparameter_correlation_heatmap.png"
plt.savefig(plot_filename, dpi=600)
print(f"Plot saved as {plot_filename}")

# Print the relevant correlations for a quick summary
print("\n--- Correlation of Hyperparameters with Performance Metrics ---")
print(correlation_matrix.loc[['gamma', 'lam', 'vf_coef', 'ent_coef'], 
                             ['final_mean_reward', 'final_policy_loss', 'final_value_loss']])