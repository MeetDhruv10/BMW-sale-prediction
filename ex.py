import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your CSV file
df = pd.read_csv("BMW.csv")

# Step 1: Bin the Engine_Size_L into categories for grouping
plt.figure()
sns.countplot(x='Sales_Classification', data=df, palette='Set3')
plt.title("Count of Sales Classification")
plt.xlabel("Sales Classification")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.show()