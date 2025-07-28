import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('BMW_Updated.csv')

# Group and summarize
grouped = df.groupby(['Model', 'Region', 'Year'])['Sales_Volume'].sum().reset_index()

# Get unique models (you can slice this to top 5 if needed)
models = grouped['Model'].unique()

# Loop through each model
# for model in models:
#     model_data = grouped[grouped['Model'] == model]
    
#     # --- Grouped Bar Plot ---
#     plt.figure(figsize=(14, 6))
#     plt.subplot(1, 2, 1)
#     sns.barplot(data=model_data, x='Year', y='Sales_Volume', hue='Region')
#     plt.title(f"{model} - Sales Volume by Region and Year")
#     plt.xlabel("Year")
#     plt.ylabel("Sales Volume")
#     plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True)

    # --- Heatmap ---
    plt.subplot(1, 2, 2)
    pivot = model_data.pivot(index='Region', columns='Year', values='Sales_Volume')
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
    plt.title(f"{model} - Sales Heatmap")
    plt.xlabel("Year")
    plt.ylabel("Region")

    plt.tight_layout()
    plt.show()
