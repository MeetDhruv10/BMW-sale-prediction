import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv("BMW_Updated.csv")

# Set plot style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Distribution of Car Prices
plt.figure()
sns.histplot(df["Price_USD"], bins=20, kde=True, color='teal')
plt.title("Distribution of Car Prices")
plt.xlabel("Price in USD")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Grouping total sales by year, region, and classification
grouped = df.groupby(['Year', 'Region', 'Sales_Classification'])['Sales_Volume'].sum().reset_index()

# Pivot for stacked bar chart by region
pivot_df = df.groupby(['Year', 'Region', 'Sales_Classification'])['Sales_Volume'].sum().unstack().fillna(0)

# Plot stacked sales classification bars per region
for region in pivot_df.index.get_level_values('Region').unique():
    region_data = pivot_df.loc[pivot_df.index.get_level_values('Region') == region]
    region_data = region_data.droplevel('Region')

    region_data.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab20c')
    plt.title(f"Stacked Sales Classification in {region}")
    plt.xlabel("Year")
    plt.ylabel("Sales Volume")
    plt.legend(title="Sales Classification")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Normalize for percentage-based stacked bars
percent_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

# Plot % stacked bar per region
for region in percent_df.index.get_level_values('Region').unique():
    region_data = percent_df.loc[percent_df.index.get_level_values('Region') == region]
    region_data = region_data.droplevel('Region')

    region_data.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='Set2')
    plt.title(f"Sales Classification (% share) in {region}")
    plt.xlabel("Year")
    plt.ylabel("Percentage")
    plt.legend(title="Sales Classification")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Sales trend line chart (total across regions)
trend_df = df.groupby(['Year', 'Sales_Classification'])['Sales_Volume'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_df, x='Year', y='Sales_Volume', hue='Sales_Classification', marker='o')
plt.title("Sales Classification Trend Over Years (All Regions)")
plt.xlabel("Year")
plt.ylabel("Total Sales Volume")
plt.grid(True)
plt.tight_layout()
plt.show()

# Heatmap of average price by region and model tier
pivot_table = df.pivot_table(values='Price_USD', index='Region', columns='Model_Tier', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title('Average Price of Model Tiers by Region')
plt.xlabel('Model Tier')
plt.ylabel('Region')
plt.tight_layout()
plt.show()

# Pie chart: Total sales by model tier
tier_sales = df.groupby('Model_Tier')['Sales_Volume'].sum()

plt.figure(figsize=(6, 6))
plt.pie(tier_sales, labels=tier_sales.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Sales Volume Share by Model Tier')
plt.tight_layout()
plt.show()

# Fuel Type distribution bar chart per region
plt.figure(figsize=(16, 8))  # Wider + more height
fuel_region_counts = df.groupby(['Region', 'Fuel_Type']).size().reset_index(name='Count')
sns.barplot(data=fuel_region_counts, x='Region', y='Count', hue='Fuel_Type', palette='coolwarm')

plt.title("Fuel Type Distribution Across Regions", fontsize=16)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Number of Cars Sold", fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=11)

# Add value labels to each bar
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9)
plt.tight_layout()
plt.show()

# Pie chart for each region's fuel type breakdown
regions = df['Region'].dropna().unique()
for region in regions:
    region_data = df[df['Region'] == region]
    fuel_counts = region_data['Fuel_Type'].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(fuel_counts, labels=fuel_counts.index, autopct=lambda p: f'{int(p * sum(fuel_counts) / 100)}\n({p:.1f}%)', startangle=90, colors=plt.cm.Pastel1.colors)
    plt.title(f"Fuel Type Distribution in {region}")
    plt.tight_layout()
    plt.show()

# Binning engine size
df['Engine_Bin'] = pd.cut(df['Engine_Size_L'], bins=[0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                          labels=['<2L', '2-3L', '3-4L', '4-5L', '5-6L'])

# Heatmap of average price by Engine Bin and Model Tier
pivot = df.pivot_table(values='Price_USD', index='Engine_Bin', columns='Model_Tier', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="cividis")
plt.title("Average Price by Engine Size and Model Tier")
plt.xlabel("Model Tier")
plt.ylabel("Engine Size Range")
plt.tight_layout()
plt.show()

# ---- Individual Model Plots ----
grouped = df.groupby(['Model', 'Region', 'Year'])['Sales_Volume'].sum().reset_index()
models = grouped['Model'].unique()

# Uncomment to activate per-model graphs
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

# 7. Car Age vs Price
plt.figure()
sns.lineplot(x='Car_Age', y='Price_USD', data=df, ci=None, marker='o')
plt.title("Car Age vs Price")
plt.xlabel("Car Age (Years)")
plt.ylabel("Price in USD")
plt.tight_layout()
plt.show()

# 8. Average Price by Region
plt.figure()
sns.barplot(x='Region', y='Price_USD', data=df, estimator=np.mean, palette='magma')
plt.title("Average Price by Region")
plt.xlabel("Region")
plt.ylabel("Average Price (USD)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# 9. Transmission Type Distribution
plt.figure()
sns.countplot(x='Transmission', data=df, palette='autumn')
plt.title("Transmission Type Distribution")
plt.xlabel("Transmission")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 10. Model Popularity Score by BMW Model
plt.figure()
sns.barplot(x='Model_Popularity_Score', y='Model', data=df.sort_values('Model_Popularity_Score', ascending=False), palette='Blues_r')
plt.title("Model Popularity Score by BMW Model")
plt.xlabel("Popularity Score")
plt.ylabel("Model")
plt.tight_layout()
plt.show()

# 11. Mileage vs Price
plt.figure()
sns.scatterplot(x='Mileage_KM', y='Price_USD', hue='Fuel_Type', data=df, palette='cool')
plt.title("Mileage vs Price")
plt.xlabel("Mileage (KM)")
plt.ylabel("Price in USD")
plt.tight_layout()
plt.show()

# 12. Sales Volume by Efficiency Group
plt.figure()
sns.barplot(x='Efficiency_Group', y='Sales_Volume', data=df, estimator=np.mean, palette='YlGnBu')
plt.title("Sales Volume by Efficiency Group")
plt.xlabel("Efficiency Group")
plt.ylabel("Average Sales Volume")
plt.tight_layout()
plt.show()
