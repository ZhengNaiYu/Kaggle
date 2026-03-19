import pandas as pd

# only read the first 5 rows to check the structure of the dataset
df_head = pd.read_csv('data/Jane Street Preprocessed.csv', nrows=5)
print(df_head)
print(df_head.shape)
print(df_head.dtypes)

# print all columns
print("\n===== All Columns =====")
for i, col in enumerate(df_head.columns, 1):
    print(f"{i}. {col}")
print(f"\nTotal {len(df_head.columns)} columns.")

# # if the dataset is too large, read in chunks
# for chunk in pd.read_csv('data/Jane Street Preprocessed.csv', chunksize=10000):
#     # process the chunk (e.g., print its shape)
#     print(chunk.shape)