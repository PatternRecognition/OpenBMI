import pandas as pd

df = pd.read_csv('C:\\Users\\dk\\Downloads\\run-Jun05_11-37-22_DESKTOP-NMLMMUUfold_3_g_0.7-tag-Eval_Acc.csv')

print(df)


from matplotlib import pyplot as plt




plt.bar(df.Step , df['Value'])

plt.show()