import pandas as pd
import numpy as np

#  df = pd.read_csv('./periodic_table.csv')
#  df = df[['Symbol','Metal', 'Nonmetal', 'Metalloid']]
#  df = df.replace({'yes': True})
#  df = df.replace({np.nan : False})
#  df.to_csv('metal_info.csv', index=False)
df = pd.read_csv('metal_info.csv')
print(df['Symbol'][df['Metal']].to_list())

