import pandas as pd

concrete = pd.read_excel('./Concrete_Data.xls')

# Recasting the column names for uniformity with other datasets
cols = concrete.columns
newcols = [str(i+1) for i in range(len(cols)-1)] + ['target']
concrete.columns = newcols
concrete = concrete.reset_index()

concrete.to_csv('./dataset.csv', index=False)
