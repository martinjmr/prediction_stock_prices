import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv('../data/X_train.csv')
y = pd.read_csv('../data/y_train.csv')

print(X.shape)
print(y.shape)