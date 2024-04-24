import pandas as pd

# Load the dataset
path='C:\Users\YUVRAJ\Desktop\LLMs\Crop Prediction\data\df1.csv'
df = pd.read_csv(path)

# Combine 'Crop', 'Crop1', and 'Crop2' columns into a single combined column
df['Combined_Crop'] = df[['Crop', 'Crop1', 'Crop2']].apply(lambda x: ','.join(x.dropna()), axis=1)

# Drop the original 'Crop', 'Crop1', and 'Crop2' columns
df = df.drop(columns=['Crop', 'Crop1', 'Crop2'])

# Display the modified DataFrame
print(df.head())
