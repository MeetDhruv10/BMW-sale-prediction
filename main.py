import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load Dataset
# =========================
df = pd.read_csv("BMW_Updated.csv")
sns.set_style("whitegrid")  # consistent style


# =========================
# 1. Sales Heatmap (per model)
# =========================
grouped = df.groupby(['Model', 'Region', 'Year'])['Sales_Volume'].sum().reset_index()

for model in grouped['Model'].unique():
    model_data = grouped[grouped['Model'] == model]
    pivot = model_data.pivot(index='Region', columns='Year', values='Sales_Volume')

    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=0.5)
    plt.title(f"{model} - Sales Heatmap", fontsize=16)
    plt.xlabel("Year")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.show()


# =========================
# 2. Engine Size Distribution by Efficiency Group
# =========================
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Efficiency_Group", y="Engine_Size_L", palette="Set3")
plt.title("Engine Size Distribution by Efficiency Group", fontsize=14)
plt.xlabel("Efficiency Group")
plt.ylabel("Engine Size (L)")
plt.tight_layout()
plt.show()


# =========================
# 3. Fuel Type Distribution by Region
# =========================
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Region', hue='Fuel_Type', palette="pastel")
plt.title("Fuel Type Distribution by Region", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.legend(title='Fuel Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# =========================
# 4a. Pie Chart - Car Colors
# =========================
if 'Color' in df.columns:
    color_counts = df['Color'].value_counts()
    color_map = {
        'Red': 'red', 'Black': 'black', 'White': 'white',
        'Blue': 'blue', 'Silver': 'silver', 'Gray': 'gray', 'Green': 'green'
    }
    pie_colors = [color_map.get(c, 'lightgray') for c in color_counts.index]

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        color_counts,
        labels=color_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=pie_colors,
        wedgeprops={'edgecolor': 'black', 'linewidth': 2}
    )
    for autotext in autotexts:
        autotext.set_color('orange')
        autotext.set_fontsize(12)
    plt.title("Distribution of Car Colors", fontsize=14)
    plt.tight_layout()
    plt.show()


# =========================
# 4b. Pie Chart - Fuel Efficiency
# =========================
if 'Efficiency_Group' in df.columns:
    efficiency_counts = df['Efficiency_Group'].value_counts()
    color_mapping = {
        'Efficient': '#A3C4F3',       # pastel blue
        'Fuel Hungry': '#F4A3A3',     # pastel red
        'Moderate': '#FFF5A3',        # pastel yellow
        'Very Efficient': '#A3F4B1'   # pastel green
    }
    pie_colors = [color_mapping.get(c, 'lightgray') for c in efficiency_counts.index]

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        efficiency_counts,
        labels=efficiency_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=pie_colors,
        wedgeprops={'edgecolor': 'black', 'linewidth': 2}
    )
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(12)
    plt.title("Fuel Efficiency Category Shares", fontsize=14)
    plt.tight_layout()
    plt.show()


# =========================
# 5. Average Car Age by Region
# =========================
avg_age = df.groupby("Region")["Car_Age"].mean().reset_index().sort_values("Region")

plt.figure(figsize=(10, 6))
sns.lineplot(data=avg_age, x="Region", y="Car_Age", marker="o", color="skyblue")
plt.title("Average Car Age by Region", fontsize=14)
plt.xlabel("Region")
plt.ylabel("Average Car Age (Years)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# =========================
# 6. Mean vs Median Price by Fuel (Dumbbell Plot)
# =========================
price_stats = df.groupby("Fuel_Type")["Price_USD"].agg(["mean", "median"]).reset_index()
price_stats = price_stats.sort_values("mean")

fuel_colors = {'Petrol': 'blue', 'Diesel': 'red', 'Hybrid': 'yellow', 'Electric': 'green'}

plt.figure(figsize=(10, 6))
for i, row in price_stats.iterrows():
    color = fuel_colors.get(row["Fuel_Type"], "gray")
    plt.plot([row["median"], row["mean"]], [row["Fuel_Type"], row["Fuel_Type"]],
             color=color, linewidth=4)
    plt.scatter(row["median"], row["Fuel_Type"], color=color, s=80, label=f"{row['Fuel_Type']} Median")
    plt.scatter(row["mean"], row["Fuel_Type"], color=color, s=80, marker='D', label=f"{row['Fuel_Type']} Mean")

plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlabel("Price (USD)")
plt.ylabel("Fuel Type")
plt.title("Mean vs Median Price by Fuel (Dumbbell)", fontsize=14)

# Remove duplicate legend entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()


# =========================
# 7. Sales Volume Trends with COVID-19 Impact Highlighted
# =========================
sales_per_year = df.groupby('Year')['Sales_Volume'].sum().reset_index()

plt.figure(figsize=(12, 7))
plt.plot(sales_per_year['Year'], sales_per_year['Sales_Volume'], marker='o', linestyle='-')

# Highlight COVID period
plt.axvspan(2019, 2021, color='red', alpha=0.1, label='COVID Impact Period')

plt.annotate('COVID-19 Impact',
             xy=(2020, sales_per_year.loc[sales_per_year['Year'] == 2020, 'Sales_Volume'].iloc[0]),
             xytext=(2020, sales_per_year['Sales_Volume'].max()*0.9),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ha='center')

plt.annotate('Pre-COVID Dip',
             xy=(2018, sales_per_year.loc[sales_per_year['Year'] == 2018, 'Sales_Volume'].iloc[0]),
             xytext=(2017, sales_per_year['Sales_Volume'].max()*0.75),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ha='center')

plt.annotate('Recovery',
             xy=(2022, sales_per_year.loc[sales_per_year['Year'] == 2022, 'Sales_Volume'].iloc[0]),
             xytext=(2023, sales_per_year['Sales_Volume'].max()*0.9),
             arrowprops=dict(facecolor='black', shrink=0.05),
             ha='center')

plt.title('Sales Volume Trends with COVID-19 Impact Highlighted')
plt.xlabel('Year')
plt.ylabel('Total Sales Volume')
plt.grid(True)
plt.xticks(sales_per_year['Year'], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()


# =========================
# 8. Top 5 Models by Region (Grouped Bar Chart)
# =========================
top_models = df.groupby("Model")["Sales_Volume"].sum().nlargest(5).index
df_top = df[df["Model"].isin(top_models)]

grouped = df_top.groupby(["Region", "Model"])["Sales_Volume"].sum().reset_index()
pivot_df = grouped.pivot(index="Region", columns="Model", values="Sales_Volume")

fig, ax = plt.subplots(figsize=(12, 6))
pivot_df.plot(kind="bar", ax=ax)

ax.set_xlabel("Region")
ax.set_ylabel("Total Sales Volume")
ax.set_title("Total Sales Volume of Top 5 Models by Region")

plt.xticks(rotation=45, ha="right")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
