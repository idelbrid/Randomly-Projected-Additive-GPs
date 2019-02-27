import pandas as pd

housing = pd.read_csv('./housing.data', header=None, sep='\s+')
housing.columns = ['CRIM',
                 'ZN',
                 'INDUS',
                 'CHAS',
                 'NOX',
                 'RM',
                 'AGE',
                 'DIS',
                 'RAD',
                 'TAX',
                 'PTRATIO',
                 'B',
                 'LSTAT',
                 'target']  # last column is target (a.k.a. MEDV)

# note: all attributes are real-valued, except for CHAS, which is 1 if
#   the tract bounds the Charles River, 0 otherwise. Since this one
#   feature is binary, rather than "categorical", it's OK.

housing = housing.reset_index()  # give it an explicit index column

housing.to_csv('./dataset.csv', index=False, )  # save (without extra index)