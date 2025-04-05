import autogluon
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

data1 = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')

train_data = TabularDataset(data=data1)
test_data = TabularDataset(data=data2)

predictor = TabularPredictor(label='Survived').fit(train_data)