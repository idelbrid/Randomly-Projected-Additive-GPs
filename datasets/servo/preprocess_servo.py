import pandas as pd

servo = pd.read_csv('./servo.data', header=None)
servo.columns = ['1', '2', '3', '4', 'target']  # last column is target
# preprocess features b/c they are categorical
# It seems like it should be one-hot encoded, but this is how the
#     other paper did it, so I want to be consistent.
replace_dict = {a: i for i, a in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
servo = servo.replace(replace_dict)

servo = servo.reset_index()  # give it an explicit index column

servo.to_csv('./dataset.csv', index=False, )  # save (without extra index)