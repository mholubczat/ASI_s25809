import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

data1 = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')

train_data = TabularDataset(data=data1)
test_data = TabularDataset(data=data2)

predictor = (TabularPredictor(label='Survived', path='models', eval_metric='roc_auc')
             .fit(train_data, presets='medium', time_limit=1800))
predictor.save('models')

predictions=predictor.predict(test_data)
print(predictions)

leaderboard = predictor.leaderboard()
print(leaderboard)

print(predictor.evaluate(train_data))

model_name=predictor.model_best
model_info=predictor.info()['model_info']
hyperparams=model_info[model_name]['hyperparameters']
print(hyperparams)