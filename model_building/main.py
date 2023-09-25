from immo_regression import cleaning, model
import pandas as pd

# Importing the dataset
df = pd.read_csv('/Users/andre/Documents/GitHub/immo-analysis-project/data/dataset_immo_.csv')

# Calling the functions
cleaned_df = cleaning(df)
final_model = model(cleaned_df)