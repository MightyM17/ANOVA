import pandas as pd

file_path = 'dataset.csv'
df = pd.read_csv(file_path)

df['Sr. No'] = df['Sr. No'].fillna(method='ffill')
#df['Questions'] = df['Questions'].fillna('')
grouped_df = df.groupby('Sr. No')


required_columns = ['Data'] + [f'Unnamed: {i}' for i in range(3, 10)]
# Dictionary to store the processed DataFrames, keyed by 'Sr. No'
processed_groups = {}

for name, group in grouped_df:
    missing_columns = [col for col in required_columns if col not in group.columns]
    if missing_columns:
        print(f"Missing columns in group {name}: {missing_columns}")
        continue
    
    filtered_group = group[required_columns]
    filtered_group.fillna('', inplace=True)
    processed_groups[name] = filtered_group

df.drop(columns=required_columns, inplace=True)
new_df = df.groupby('Sr. No').first()

for i in range(1, len(processed_groups) + 1):
    d = processed_groups[i]
    new_columns = d.iloc[0]
    d.columns = new_columns
    d = d.drop(d.index[0])
    d.reset_index(drop=True, inplace=True)
    processed_groups[i] = d
    
new_df = new_df.assign(Data=processed_groups.values())
print(new_df)


