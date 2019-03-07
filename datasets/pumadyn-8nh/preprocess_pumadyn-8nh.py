import pandas as pd

puma = pd.read_csv('./Dataset.data', header=None, sep='\s+')
puma.columns = [str(i + 1) for i in range(8)] + ['target']
puma = puma.reset_index()

puma.to_csv('./dataset.csv', index=False, )  # save (without extra index)