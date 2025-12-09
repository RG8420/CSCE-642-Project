import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the model files (assuming .csv extension, which is standard for SB3 logs)
model_files = {
    "PPO": "PPO_Test_Results/PPO_test_results.csv",
    "A2C": "PPO_Test_Results/A2C_test_results.csv",
    "SAC": "PPO_Test_Results/SAC_test_results.csv",
    "TD3": "PPO_Test_Results/TD3_test_results.csv",
}

all_results = []
loaded_models = []

# Column mapping check for common output formats (user's code vs SB3 defaults)
# The user's provided code used 'iter' and 'reward' for columns.
# A common alternative is 'Episode' and 'Cumulative_Reward'.

for model_name, file_path in model_files.items():
    df = pd.read_csv(file_path)
    # Determine column names based on available columns
    iteration_col = 'iter'
    reward_col = 'reward'
    
    data_to_plot = df[[iteration_col, reward_col]].copy()
    # Standardize column names for concatenation and plotting
    data_to_plot.rename(columns={iteration_col: "Iteration", reward_col: "Reward"}, inplace=True)
    data_to_plot["Model"] = model_name
    all_results.append(data_to_plot)
    loaded_models.append(model_name)
    
# Concatenate all loaded dataframes
df_combined = pd.concat(all_results, ignore_index=True)
print(f"\nSuccessfully loaded and combined data for models: {', '.join(loaded_models)}.")

# Plotting using Matplotlib
plt.figure(figsize=(12, 6))

for model_name, group in df_combined.groupby('Model'):
    # Ensure the plot remains ordered by iteration
    group_sorted = group.sort_values(by='Iteration') 
    plt.plot(group_sorted['Iteration'], group_sorted['Reward'], label=model_name)

plt.xlabel('Episode', fontsize=16)
plt.ylabel('Cumulative Reward', fontsize=16)
# plt.title('RL Model Performance Comparison During Testing')
plt.legend(title='Model')
plt.grid(False)
plt.tight_layout()
# plt.show()

# Save the plot
plot_filename = "model_performance_comparison.png"
plt.savefig(plot_filename, dpi=600)
print(f"Plot saved as {plot_filename}")
    