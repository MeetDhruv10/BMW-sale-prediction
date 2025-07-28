import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
df = pd.read_csv("BMW.csv")

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

grouped = df.groupby(['Year', 'Region', 'Sales_Classification'])['Sales_Volume'].sum().reset_index()

# Pivot to reshape the data
pivot_df = df.groupby(['Year', 'Region', 'Sales_Classification'])['Sales_Volume'].sum().unstack().fillna(0)

# Plot a stacked bar chart for each region
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

# Convert to percentages
percent_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

# Plot percentage stacked bars
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

# Sales Classification Trend Over Years (All Regions)
trend_df = df.groupby(['Year', 'Sales_Classification'])['Sales_Volume'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=trend_df, x='Year', y='Sales_Volume', hue='Sales_Classification', marker='o')
plt.title("Sales Classification Trend Over Years (All Regions)")
plt.xlabel("Year")
plt.ylabel("Total Sales Volume")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Price Distribution by Model Tier
pivot_table = df.pivot_table(values='Price_USD', index='Region', columns='Model_Tier', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title('Average Price of Model Tiers by Region')
plt.xlabel('Model Tier')
plt.ylabel('Region')
plt.tight_layout()
plt.show()


tier_sales = df.groupby('Model_Tier')['Sales_Volume'].sum()

plt.figure(figsize=(6, 6))
plt.pie(tier_sales, labels=tier_sales.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Sales Volume Share by Model Tier')
plt.tight_layout()
plt.show()

# 4. Average Mileage per Year by Fuel Type
plt.figure(figsize=(16, 8))  # Wider + more height
fuel_region_counts = df.groupby(['Region', 'Fuel_Type']).size().reset_index(name='Count')
sns.barplot(data=fuel_region_counts, x='Region', y='Count', hue='Fuel_Type', palette='coolwarm')

plt.title("Fuel Type Distribution Across Regions", fontsize=16)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Number of Cars Sold", fontsize=12)

# Fix label overlap
plt.xticks(rotation=45, ha='right', fontsize=11)
# Add count labels on top of each bar (optional enhancement)
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%.0f', label_type='edge', fontsize=9)
plt.tight_layout()
plt.show()
regions = df['Region'].dropna().unique()
for region in regions:
    region_data = df[df['Region'] == region]
    fuel_counts = region_data['Fuel_Type'].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(fuel_counts, labels=fuel_counts.index, autopct=lambda p: f'{int(p * sum(fuel_counts) / 100)}\n({p:.1f}%)', startangle=90, colors=plt.cm.Pastel1.colors)
    plt.title(f"Fuel Type Distribution in {region}")
    plt.tight_layout()
    plt.show()



# Bin engine sizes if not already binned
df['Engine_Bin'] = pd.cut(df['Engine_Size_L'], bins=[0, 2.0, 3.0, 4.0, 5.0, 6.0], 
                          labels=['<2L', '2-3L', '3-4L', '4-5L', '5-6L'])

# Create the pivot table: average price by Engine Bin and Model Tier
pivot = df.pivot_table(values='Price_USD', index='Engine_Bin', columns='Model_Tier', aggfunc='mean')

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="cividis")
plt.title("Average Price by Engine Size and Model Tier")
plt.xlabel("Model Tier")
plt.ylabel("Engine Size Range")
plt.tight_layout()
plt.show()


# 6. Count of Sales Classification
plt.figure()
sns.countplot(x='Sales_Classification', data=df, palette='Set3')
plt.title("Count of Sales Classification")
plt.xlabel("Sales Classification")
plt.ylabel("Number of Cars")
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
