import pandas as pd
import numpy as np

# Step 1: Read the CSV file
df = pd.read_csv('BMW.csv')

# Step 2: Calculate Car Age
df['Car_Age'] = 2025 - df['Year']

# Step 3: Categorize Efficiency Group based on Engine Size
def fuel_efficiency_group(engine):
    if engine <= 1.6:
        return 'Very Efficient'
    elif engine <= 2.5:
        return 'Efficient'
    elif engine <= 4.0:
        return 'Moderate'
    else:
        return 'Fuel Hungry'

df['Efficiency_Group'] = df['Engine_Size_L'].apply(fuel_efficiency_group)

# Step 4: Classify Model as 'Luxury', 'Standard', or 'Unknown'
def classify_model_tier(model):
    flagship_models = ['i8', '7 Series', 'X6', 'M4', 'M5']
    luxury_models = ['3 Series', '5 Series', 'i3', 'M3', 'X1', 'X3', 'X5']
    
    try:
        model = str(model)
        if model in flagship_models:
            return 'Luxury+'
        elif model in luxury_models:
            return 'Luxury'
        else:
            return 'Others'
    except:
        return 'Others'

df['Model_Tier'] = df['Model'].apply(classify_model_tier)



# Step 5: Calculate Mileage per Year
df['Mileage_per_Year'] = df['Mileage_KM'] / df['Car_Age']

# Step 6: Calculate Model Popularity Score
model_counts = df['Model'].value_counts()
df['Model_Popularity_Score'] = df['Model'].map(model_counts)

# Optional: Step 7 - Region + Model Popularity (if needed for visuals)
sales_by_region_model = df.groupby(['Region', 'Model']).size().reset_index(name='Popularity')

# Step 8: Reorder columns   
ordered_cols = [
    'Model', 'Year', 'Car_Age', 'Region', 'Color',
    'Engine_Size_L', 'Fuel_Type', 'Transmission',
    'Mileage_KM', 'Mileage_per_Year',
    'Price_USD', 'Sales_Volume', 'Model_Popularity_Score',
    'Efficiency_Group', 'Model_Tier', 'Sales_Classification'
]

df = df[ordered_cols]

# Step 8: Save to new CSV
df.to_csv('BMW.csv', index=False)
print('Data cleaned and saved to BMW.csv')
