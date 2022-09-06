# Sigmoid-neuron-for-classification-problem
Gender classification using Sigmoid neuron

# IMPORTING LIBRARIES
```python
import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense
import seaborn as sns
import matplotlib as plt
import numpy as np
```

# Loading Dataset
```python
data = pd.read_csv('/content/gender_classification_v7.csv', delimiter=',')
```

# Creating correlation Graph
```python
corr = data.corr()
sns.heatmap(corr, 
xticklabels=corr.columns.values,
yticklabels=corr.columns.values)
```
