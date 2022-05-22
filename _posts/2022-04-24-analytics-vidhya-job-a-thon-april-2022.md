---
title: Predict rental car demand
author: omkar_patil
date: 2022-04-24
categories: [solution, code]
tags: [prediction, time series analysis, forcasting]
math: true
mermaid: true
---

# Problem Statement :
    
ABC is a car rental company based out of Bangalore. It rents cars for both in and out stations at affordable prices. The users can rent different types of cars like Sedans, Hatchbacks, SUVs and MUVs, Minivans and so on.

In recent times, the demand for cars is on the rise. As a result, the company would like to tackle the problem of supply and demand. The ultimate goal of the company is to strike the balance between the supply and demand inorder to meet the user expectations. 

The company has collected the details of each rental. Based on the past data, the company would like to forecast the demand of car rentals on an hourly basis. 


```python
import pandas as pd
from matplotlib import pyplot

```


```python
training_data = pd.read_csv('/kaggle/input/jobathon-april-2022/train_E1GspfA.csv')
training_data.shape
```

Training dataset contains 18247 data points, each having 3 features.
 - date
 - hour
 - demand

Data points are spread over 3 dimensions


```python
training_data.head()
```


```python
training_data.info()
```

The training data has no null values, every data point has value for every feature

The _date_ feature has object as its data type, which needs to be converted to DateTime


```python
training_data['date'] = pd.to_datetime(training_data['date'])
```

Training data contains data point in the time range :


```python
print(f"starting date : {str(training_data['date'].dt.date.min())}")
print(f"end date : {str(training_data['date'].dt.date.max())}")
```

Instead of having _hour_ as separate frature/column, _date_ and _hour_ can be combined to form a timestamp


```python
def dataPreprocessing(dataFrame):
    dataFrame['date'] = pd.to_datetime(dataFrame['date']) + dataFrame['hour'].astype('timedelta64[h]')
    dataFrame.drop(columns=['hour'], axis=1, inplace=True)
    return dataFrame
```

Dropping the non required column : _hour_


```python
training_data = dataPreprocessing(training_data)
training_data.head()
```

# Exploratory Data Analysis


```python
import plotly.express as px
```


```python
fig = px.line(training_data, x='date', y='demand')

fig.update_xaxes(rangeslider_visible=True)
fig.show()
```

# Splitting training data in tain and validation set


```python
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

```


```python
training_data.rename(columns={'date': 'ds', 'demand': 'y'}, inplace=True)
train_data = training_data.sample(frac=0.8, random_state=10)

validation_data = training_data.drop(train_data.index)

print(f'training data size : {train_data.shape}')
print(f'validation data size : {validation_data.shape}')

train_data = train_data.reset_index()
validation_data = validation_data.reset_index()
```

# Prediction Models

importing required libraries


```python
from sklearn.metrics import mean_absolute_error
from fbprophet import Prophet
```

fitting the model on the training data


```python
model = Prophet()
model.fit(train_data)
```

Performing prediction on the validation dataset


```python
prediction = model.predict(pd.DataFrame({'ds':validation_data['ds']}))
y_actual = validation_data['y']
y_predicted = prediction['yhat']
y_predicted = y_predicted.astype(int)
mean_absolute_error(y_actual, y_predicted)
```

Plotting results of predictions on validation dataset


```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=validation_data['ds'], y=y_actual, name="actual targets"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=validation_data['ds'], y=y_predicted, name="predicted targets"),
    secondary_y=True,
)

fig.update_layout(
    title_text="Actual vs Predicted Targets"
)

fig.update_xaxes(title_text="Timeline")
fig.update_yaxes(title_text="<b>actual</b> targets", secondary_y=False)
fig.update_yaxes(title_text="<b>predicted</b> targets", secondary_y=True)

fig.show()
```

# Predictions on test dataset


```python
test_data = pd.read_csv('/kaggle/input/jobathon-april-2022/test_6QvDdzb.csv')
print(f'test dataset size : {test_data.shape}')
testing_data = dataPreprocessing(test_data.copy())
testing_data.head()
```


```python
test_prediction = model.predict(pd.DataFrame({'ds':testing_data['date']}))
```


```python
test_prediction = test_prediction['yhat']
test_prediction = test_prediction.astype(int)
test_data['demand'] = test_prediction
test_data.head()
# test_data.to_csv('./kaggle/output/submission.csv', index=False)
```

Source code : <a href="https://www.kaggle.com/code/phileinsophos/job-a-thon-april2022" style="color: blue;">Kaggle</a>
