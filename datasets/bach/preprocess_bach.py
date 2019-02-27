from scipy.io import loadmat
import pandas as pd

d = loadmat('./bach_synth_r_200.mat')
df = pd.DataFrame(data=d['X'])
df.columns=[str(i+1) for i in range(8)]
df = df.reset_index()
df['target'] = d['y']

df.to_csv('./dataset.csv', index=None)
