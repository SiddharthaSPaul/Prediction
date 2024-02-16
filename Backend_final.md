# Import libraries and data


```python
# To print multiple output in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
```


```python
# Import all required libraries
import pandas as pd # Data manipulation and analysis library
from statsmodels.tsa.seasonal import seasonal_decompose # Seasonal decomposition to check seasonality
from sklearn.feature_selection import RFE # RFE (Recursive Feature Elimination) is for feature selection
from sklearn.ensemble import RandomForestRegressor # Random forest modelling
import numpy as np # For arrays and mathematical operations
from statsmodels.tsa.stattools import adfuller # Dickey-fuller testto check stationarity of data
from sklearn.metrics import mean_squared_error # For evaluating the model
from sklearn.preprocessing import LabelEncoder # To encode categorical integer features
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # Statistical data visualization
import scipy.stats as stats # Statistical analysis
import pylab # For plotting
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import warnings # To handle warnings
warnings.filterwarnings("ignore") # Ignore all warings
from statsmodels.tsa.statespace.sarimax import SARIMAX # To do SARIMAX
from sklearn.model_selection import train_test_split # To split into train and test data set
from sklearn.preprocessing import StandardScaler # For RNN: Recursive neural network
from keras.models import Sequential # For RNN
from keras.layers import LSTM, Dense, Dropout # For RNN
from keras.optimizers import Adam # For RNN
```

    /var/folders/yh/pl2cz6pd3rz655m2p297hdn40000gp/T/ipykernel_65959/2863653633.py:15: UserWarning: 
    The dash_core_components package is deprecated. Please replace
    `import dash_core_components as dcc` with `from dash import dcc`
      import dash_core_components as dcc
    /var/folders/yh/pl2cz6pd3rz655m2p297hdn40000gp/T/ipykernel_65959/2863653633.py:16: UserWarning: 
    The dash_html_components package is deprecated. Please replace
    `import dash_html_components as html` with `from dash import html`
      import dash_html_components as html



```python
# Ignore all warings
warnings.filterwarnings("ignore")
```


```python
# Import data
# Import data
file_path = '/Users/deepakvarier/Downloads/hackathon_data'
date_format = "%d/%m/%y"
df = pd.read_csv(file_path+'/train.csv', sep = ',', parse_dates = ['week'], date_parser = lambda x: pd.to_datetime(x, format = date_format))
```

# Data cleaning


```python
# Characteristics of data
df.head()
df.shape
df.info()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-17</td>
      <td>8091</td>
      <td>216418</td>
      <td>99.0375</td>
      <td>111.8625</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-17</td>
      <td>8091</td>
      <td>216419</td>
      <td>99.0375</td>
      <td>99.0375</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-17</td>
      <td>8091</td>
      <td>216425</td>
      <td>133.9500</td>
      <td>133.9500</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-17</td>
      <td>8091</td>
      <td>216233</td>
      <td>133.9500</td>
      <td>133.9500</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-17</td>
      <td>8091</td>
      <td>217390</td>
      <td>141.0750</td>
      <td>141.0750</td>
      <td>0</td>
      <td>0</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>






    (150150, 9)



    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150150 entries, 0 to 150149
    Data columns (total 9 columns):
     #   Column           Non-Null Count   Dtype         
    ---  ------           --------------   -----         
     0   record_ID        150150 non-null  int64         
     1   week             150150 non-null  datetime64[ns]
     2   store_id         150150 non-null  int64         
     3   sku_id           150150 non-null  int64         
     4   total_price      150149 non-null  float64       
     5   base_price       150150 non-null  float64       
     6   is_featured_sku  150150 non-null  int64         
     7   is_display_sku   150150 non-null  int64         
     8   units_sold       150150 non-null  int64         
    dtypes: datetime64[ns](1), float64(2), int64(6)
    memory usage: 10.3 MB



```python
df_max_week = df['week'].max()
```


```python
df_min_week = df['week'].min()
```


```python
# Check null values in the data
df.isnull().sum()
```




    record_ID          0
    week               0
    store_id           0
    sku_id             0
    total_price        1
    base_price         0
    is_featured_sku    0
    is_display_sku     0
    units_sold         0
    dtype: int64




```python
# Since total no. of rows = 150150 and the null value is only in 1 row, therefore, we will remove the null row
# Calculate the total number of rows
total_rows = len(df)
# Calculate the number of rows with missing values
na_rows = df.isna().any(axis=1).sum()
if na_rows < total_rows * 0.01:
    df.dropna(inplace=True)
else:
    # Fill missing values with the average of store_id and sku_id combination
    df.fillna(df.groupby(['store_id', 'sku_id']).transform('mean'), inplace=True)
df.isnull().sum()
```




    record_ID          0
    week               0
    store_id           0
    sku_id             0
    total_price        0
    base_price         0
    is_featured_sku    0
    is_display_sku     0
    units_sold         0
    dtype: int64




```python
# Checking whether there are rows where the total_price or units_sold <=0
df.shape
df['total_price'].loc[df['total_price']<=0].count()
df['units_sold'].loc[df['units_sold']<=0].count()
```




    (150149, 9)






    0






    0




```python
# Delete rows with negative rows
con1 = df['units_sold']<=0
con2 = df['total_price']<=0
df = df[~(con1 & con2)]
df.shape
```




    (150149, 9)




```python
# Dropping duplicates if any
df.shape
df = df.drop_duplicates(['week', 'store_id', 'sku_id'])
df.shape
```




    (150149, 9)






    (150149, 9)




```python
# Sort dataframe by date column in chronological order
df = df.sort_values(by='week', ascending=False)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>150149</th>
      <td>212644</td>
      <td>2013-07-09</td>
      <td>9984</td>
      <td>679023</td>
      <td>234.4125</td>
      <td>234.4125</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
    </tr>
    <tr>
      <th>149427</th>
      <td>211610</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>378934</td>
      <td>213.0375</td>
      <td>213.0375</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>162</td>
    </tr>
    <tr>
      <th>149376</th>
      <td>211531</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216419</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>137</td>
    </tr>
    <tr>
      <th>149377</th>
      <td>211532</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>300021</td>
      <td>109.0125</td>
      <td>109.0125</td>
      <td>0</td>
      <td>0</td>
      <td>108</td>
    </tr>
  </tbody>
</table>
</div>



# Data Selection (Partly)


```python
# Function to create data frame for the selected store_id and sku_id
def create_dataframe(sku_id, df):
    # Filter the data for the specified store_id and sku_id
    filtered_data = df[(df['sku_id'] == sku_id)]

    # If no data is found for the specified sku_id, return None
    if filtered_data.empty:
        print("No data found for the specified sku_id.")
        return None

    return filtered_data
```


```python
# Get user input for sku_id
sku_id = int(input("Enter sku_id: "))
store_id = int(input("Enter store_id: "))

#sku_id=216425
```

    Enter sku_id: 216418
    Enter store_id: 8091



```python
type(sku_id)
```




    int




```python
# Call the function with user inputs to create dataframe of selected store_id and sku_id
df_selected = create_dataframe(sku_id,df)
if df_selected is not None:
    df_selected.head()
    df_selected.shape
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>162</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>74</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>






    (8840, 9)




```python
#df_selected = df_selected.drop(columns=['record_ID', 'store_id'])
```


```python
df_selected.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>162</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>74</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Group by sku_id and week and perform aggregation
#df_selected = df.groupby(['sku_id','week']).agg({
#    'total_price': 'mean',
#    'base_price': 'mean',
#    'is_featured_sku': 'max',
#    'is_display_sku': 'max',
#    'units_sold': 'sum'
#}).reset_index()

# Print the aggregated DataFrame
#print(df_selected)
```

# Data Pre-processing


```python
# Pre-processing the data
def preprocess_data(df):
    # Convert 'week' column to datetime type and extract seasonality features
    df['week'] = pd.to_datetime(df['week'])
    df['month'] = df['week'].dt.month
    df['year'] = df['week'].dt.year
    df['day_of_week'] = df['week'].dt.dayofweek
    df['day_of_month'] = df['week'].dt.day
    df['discount'] = df['base_price'] - df['total_price']
    # Encode categorical variables 'is_featured_sku' and 'is_display_sku'
    label_encoder = LabelEncoder()
    df['is_featured_sku'] = label_encoder.fit_transform(df['is_featured_sku'])
    df['is_display_sku'] = label_encoder.fit_transform(df['is_display_sku'])
    
    return df
```


```python
# Call the function to pre-process the data
df_processed = preprocess_data(df_selected)
```


```python
df_processed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>162</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>74</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df_processed.drop(['week'], inplace=True, axis = 1)
```


```python
df_processed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>162</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>140</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>74</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>63</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check if the data is stationary
result = adfuller(df_processed['units_sold'].dropna())
# Print the test statistic and p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

    ADF Statistic: -12.236053035856896
    p-value: 1.0275640013347197e-22



```python
# Since the p-value is below 0.05,
# the data can be assumed to be stationary hence we can proceed with the data without any transformation.
```


```python
df_processed.shape
```




    (8840, 14)




```python
df_processed['units_sold'].skew()
```




    2.3969726636205153




```python
# units sold is highly positively skewed since skewness > 1
```


```python
df_processed.units_sold.hist()
```




    <Axes: >




    
![png](output_34_1.png)
    



```python
sns.kdeplot(df_processed.units_sold)
```




    <Axes: xlabel='units_sold', ylabel='Density'>




    
![png](output_35_1.png)
    



```python
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_processed['units_sold'])
plt.show()
```




    <Figure size 1600x500 with 0 Axes>






    <Axes: >






    <Axes: xlabel='units_sold', ylabel='Density'>




    
![png](output_36_3.png)
    



```python
# Q-Q plot
stats.probplot(df_processed.units_sold, plot = pylab)
```




    ((array([-3.78002203, -3.55315481, -3.42852287, ...,  3.42852287,
              3.55315481,  3.78002203]),
      array([   1,    1,    1, ...,  825,  991, 1099])),
     (65.42171805422386, 88.92386877828054, 0.9095122752806173))




    
![png](output_37_1.png)
    



```python
# Tail of the data
df_processed.loc[df_processed['store_id']==store_id].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4620</th>
      <td>6553</td>
      <td>2011-02-14</td>
      <td>8091</td>
      <td>216418</td>
      <td>106.8750</td>
      <td>106.8750</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>2</td>
      <td>2011</td>
      <td>0</td>
      <td>14</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3465</th>
      <td>4913</td>
      <td>2011-02-07</td>
      <td>8091</td>
      <td>216418</td>
      <td>98.3250</td>
      <td>98.3250</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>2</td>
      <td>2011</td>
      <td>0</td>
      <td>7</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2310</th>
      <td>3277</td>
      <td>2011-01-31</td>
      <td>8091</td>
      <td>216418</td>
      <td>96.9000</td>
      <td>96.9000</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>31</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1155</th>
      <td>1641</td>
      <td>2011-01-24</td>
      <td>8091</td>
      <td>216418</td>
      <td>99.0375</td>
      <td>111.8625</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>24</td>
      <td>12.825</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-17</td>
      <td>8091</td>
      <td>216418</td>
      <td>99.0375</td>
      <td>111.8625</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>12.825</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Logarithmic transformation of data
df_processed['units_sold'] = np.log(df_processed['units_sold'])
```


```python
# Tail of the data
df_processed.loc[df_processed['store_id']==store_id].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4620</th>
      <td>6553</td>
      <td>2011-02-14</td>
      <td>8091</td>
      <td>216418</td>
      <td>106.8750</td>
      <td>106.8750</td>
      <td>0</td>
      <td>0</td>
      <td>3.178054</td>
      <td>2</td>
      <td>2011</td>
      <td>0</td>
      <td>14</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3465</th>
      <td>4913</td>
      <td>2011-02-07</td>
      <td>8091</td>
      <td>216418</td>
      <td>98.3250</td>
      <td>98.3250</td>
      <td>0</td>
      <td>0</td>
      <td>2.833213</td>
      <td>2</td>
      <td>2011</td>
      <td>0</td>
      <td>7</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>2310</th>
      <td>3277</td>
      <td>2011-01-31</td>
      <td>8091</td>
      <td>216418</td>
      <td>96.9000</td>
      <td>96.9000</td>
      <td>0</td>
      <td>0</td>
      <td>2.302585</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>31</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>1155</th>
      <td>1641</td>
      <td>2011-01-24</td>
      <td>8091</td>
      <td>216418</td>
      <td>99.0375</td>
      <td>111.8625</td>
      <td>0</td>
      <td>0</td>
      <td>3.526361</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>24</td>
      <td>12.825</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-17</td>
      <td>8091</td>
      <td>216418</td>
      <td>99.0375</td>
      <td>111.8625</td>
      <td>0</td>
      <td>0</td>
      <td>2.995732</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>12.825</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_processed['units_sold'].skew()
```




    -0.4681500384226393




```python
plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_processed['units_sold'])
plt.subplot(1,2,2)
stats.probplot(df_processed['units_sold'], plot = pylab)
plt.show()
```




    <Figure size 1600x500 with 0 Axes>






    <Axes: >






    <Axes: xlabel='units_sold', ylabel='Density'>






    <Axes: >






    ((array([-3.78002203, -3.55315481, -3.42852287, ...,  3.42852287,
              3.55315481,  3.78002203]),
      array([0.        , 0.        , 0.        , ..., 6.71538339, 6.89871453,
             7.00215595])),
     (0.8272750079295538, 4.179689663970701, 0.9931534920717102))




    
![png](output_42_5.png)
    



```python
# Finding the boundary values
UL = df_processed['units_sold'].mean() + 3*df_processed['units_sold'].std()
LL = df_processed['units_sold'].mean() - 3*df_processed['units_sold'].std()
UL
LL
```




    6.677765810601505






    1.6816135173398958




```python
df_processed.shape
```




    (8840, 14)




```python
df_processed['units_sold'].loc[df_processed['units_sold']<LL].count()
```




    45




```python
df_processed['units_sold'].loc[df_processed['units_sold']>UL].count()
```




    4




```python
# Removing outliers
condition1 = df_processed['units_sold']>UL
condition2 = df_processed['units_sold']<LL
df_processed = df_processed[~(condition1 & condition2)]
```

# Understanding the components of the data


```python
# Seasonal decompose
df_processed.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>5.087596</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>4.941642</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>4.304065</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>3.806662</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>4.143135</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Pre-processing for seasonal decompose
df_seasonal_decompose = df_processed
df_seasonal_decompose['week'] = pd.to_datetime(df_seasonal_decompose['week'])
df_seasonal_decompose = df_seasonal_decompose.set_index('week')
#store_id=8091
df_seasonal_decompose = df_seasonal_decompose[df_seasonal_decompose['store_id'] == store_id]
```


```python
# Seasonal decomposition
result_seasonal_decompose = seasonal_decompose(df_seasonal_decompose['units_sold'], model='additive', period=52)  # Assuming weekly seasonality
```


```python
# Plot the decomposition
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(df_seasonal_decompose['units_sold'], label='Original')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(result_seasonal_decompose.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(result_seasonal_decompose.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(result_seasonal_decompose.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```




    <Figure size 1200x800 with 0 Axes>






    <Axes: >






    [<matplotlib.lines.Line2D at 0x287060150>]






    <matplotlib.legend.Legend at 0x28708cc50>






    <Axes: >






    [<matplotlib.lines.Line2D at 0x2870dbad0>]






    <matplotlib.legend.Legend at 0x2871b8d90>






    <Axes: >






    [<matplotlib.lines.Line2D at 0x287192f50>]






    <matplotlib.legend.Legend at 0x287190d90>






    <Axes: >






    [<matplotlib.lines.Line2D at 0x287140110>]






    <matplotlib.legend.Legend at 0x287156750>




    
![png](output_52_13.png)
    



```python
# Calculate metrics
#trend_mean = result.trend.mean()  # Mean of the trend component
#seasonal_mean = result.seasonal.mean()  # Mean of the seasonal component
#residual_std = result.resid.std()  # Standard deviation of the residual component

# Print insights
#print("Insights from Seasonal Decomposition:")
#print(f"Mean of Trend Component: {trend_mean}")
#print(f"Mean of Seasonal Component: {seasonal_mean}")
#print(f"Standard Deviation of Residual Component: {residual_std}")
```

# Random Forest


```python
# Calculate the number of rows for testing
test_size = int(len(df_processed)*0.2)
end_point = len(df_processed)
x = end_point - test_size
```


```python
df_processed.shape
test_size
end_point
x
```




    (8840, 14)






    1768






    8840






    7072




```python
# Split into train and test
df_processed_train = df_processed.iloc[:x - 1]
df_processed_test = df_processed.iloc[x:]
```


```python
max_training_week = df_processed_train['week'].max()
min_training_week = df_processed_train['week'].min()
```


```python
# Check shape of test and train
df_processed_train.shape
df_processed_test.shape
```




    (7071, 14)






    (1768, 14)




```python
# Processed data
df_processed_train.head()
df_processed_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>5.087596</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>4.941642</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>4.304065</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>3.806662</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>4.143135</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29255</th>
      <td>41405</td>
      <td>2011-07-11</td>
      <td>9112</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>119.7000</td>
      <td>0</td>
      <td>1</td>
      <td>5.023881</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
    </tr>
    <tr>
      <th>29297</th>
      <td>41474</td>
      <td>2011-07-11</td>
      <td>9164</td>
      <td>216418</td>
      <td>116.1375</td>
      <td>106.1625</td>
      <td>0</td>
      <td>1</td>
      <td>4.418841</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
    </tr>
    <tr>
      <th>29283</th>
      <td>41451</td>
      <td>2011-07-11</td>
      <td>9147</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>118.9875</td>
      <td>0</td>
      <td>0</td>
      <td>4.317488</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>29236</th>
      <td>41384</td>
      <td>2011-07-11</td>
      <td>9092</td>
      <td>216418</td>
      <td>69.1125</td>
      <td>69.1125</td>
      <td>0</td>
      <td>0</td>
      <td>3.784190</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>29181</th>
      <td>41308</td>
      <td>2011-07-11</td>
      <td>8991</td>
      <td>216418</td>
      <td>69.8250</td>
      <td>69.8250</td>
      <td>0</td>
      <td>0</td>
      <td>4.077537</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test = df_processed_test.loc[:, df_processed_test.columns != 'units_sold']
y_test = df_processed_test[['units_sold']]
X_train = df_processed_train.loc[:, df_processed_train.columns != 'units_sold']
y_train = df_processed_train[['units_sold']]
```


```python
X_test.head()
y_test.head()
X_train.head()
y_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29255</th>
      <td>41405</td>
      <td>2011-07-11</td>
      <td>9112</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>119.7000</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
    </tr>
    <tr>
      <th>29297</th>
      <td>41474</td>
      <td>2011-07-11</td>
      <td>9164</td>
      <td>216418</td>
      <td>116.1375</td>
      <td>106.1625</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
    </tr>
    <tr>
      <th>29283</th>
      <td>41451</td>
      <td>2011-07-11</td>
      <td>9147</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>118.9875</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>29236</th>
      <td>41384</td>
      <td>2011-07-11</td>
      <td>9092</td>
      <td>216418</td>
      <td>69.1125</td>
      <td>69.1125</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>29181</th>
      <td>41308</td>
      <td>2011-07-11</td>
      <td>8991</td>
      <td>216418</td>
      <td>69.8250</td>
      <td>69.8250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29255</th>
      <td>5.023881</td>
    </tr>
    <tr>
      <th>29297</th>
      <td>4.418841</td>
    </tr>
    <tr>
      <th>29283</th>
      <td>4.317488</td>
    </tr>
    <tr>
      <th>29236</th>
      <td>3.784190</td>
    </tr>
    <tr>
      <th>29181</th>
      <td>4.077537</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>5.087596</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>4.941642</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>4.304065</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>3.806662</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>4.143135</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
```


```python
X_test_sarimax = X_test
y_test_sarimax = y_test
X_train_sarimax = X_train
y_train_sarimax = y_train
```


```python
X_test.head()
y_test.head()
X_train.head()
y_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41405</td>
      <td>2011-07-11</td>
      <td>9112</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>119.7000</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41474</td>
      <td>2011-07-11</td>
      <td>9164</td>
      <td>216418</td>
      <td>116.1375</td>
      <td>106.1625</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41451</td>
      <td>2011-07-11</td>
      <td>9147</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>118.9875</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41384</td>
      <td>2011-07-11</td>
      <td>9092</td>
      <td>216418</td>
      <td>69.1125</td>
      <td>69.1125</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41308</td>
      <td>2011-07-11</td>
      <td>8991</td>
      <td>216418</td>
      <td>69.8250</td>
      <td>69.8250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.023881</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.418841</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.317488</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.784190</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.077537</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.087596</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.941642</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.304065</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.806662</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.143135</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(y_test)
```




    pandas.core.frame.DataFrame




```python
type(X_test)
```




    pandas.core.frame.DataFrame




```python
X_test.set_index('week', inplace=True)
X_train.set_index('week', inplace=True)
```


```python
def train_random_forest(X_train, y_train):
    # Creating a Random Forest regressor
    #rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Training the model
    #rf_regressor.fit(X_train, y_train)

    # Making predictions on the testing set
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor = RFE(estimator = rf_regressor, n_features_to_select=7)
    fit = rf_regressor.fit(X_train, y_train)
    y_pred = fit.predict(X_test)
    selected_features = X_train.columns[rf_regressor.support_]
    print("Selected Features:",selected_features)
    
    return y_pred, fit
```


```python
y_pred, fit = train_random_forest(X_train,y_train)
```

    Selected Features: Index(['record_ID', 'store_id', 'total_price', 'base_price', 'is_display_sku',
           'month', 'day_of_month'],
          dtype='object')



```python
y_pred
```




    array([5.11313863, 4.77454691, 4.04309527, ..., 5.32315111, 4.88205047,
           3.28997654])



# Evaluate Random Forest Model


```python
#Evaluate accuracy using MAPE
y_true = np.array(y_test['units_sold'])
sumvalue=np.sum(y_true)
mape=np.sum(np.abs((y_true - y_pred)))/sumvalue*100
accuracy=100-mape
print('Accuracy:', round(accuracy,2),'%.')
```

    Accuracy: 91.95 %.



```python
# Find RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:",rmse)
print("MSE:",mse)
```

    RMSE: 0.45974194406315844
    MSE: 0.2113626551309723



```python
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual units_sold')
    plt.ylabel('Predicted units_sold')
    plt.title('Actual vs. Predicted units_sold')
    plt.show()
```


```python
y_test1 = y_test.values.flatten()
```


```python
y_test1
```




    array([5.02388052, 4.41884061, 4.31748811, ..., 4.47733681, 5.42934563,
           2.99573227])




```python
actual_values_rf = np.exp(y_test1)
predicted_values_rf = np.exp(y_pred)
actual_values_rf = pd.DataFrame(actual_values_rf, columns=['actual_values_rf'])
predicted_values_rf = pd.DataFrame(predicted_values_rf, columns = ['predicted_values_rf'])
```


```python
actual_values_rf
predicted_values_rf
X_test.reset_index(inplace = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>152.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>83.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>65.0</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>88.0</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>228.0</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 1 columns</p>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>166.191149</td>
    </tr>
    <tr>
      <th>1</th>
      <td>118.456631</td>
    </tr>
    <tr>
      <th>2</th>
      <td>57.002508</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48.250691</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62.508541</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>129.156057</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>36.100967</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>205.028935</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>131.900846</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>26.842234</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 1 columns</p>
</div>




```python
merged_rf_df = pd.concat([X_test, actual_values_rf, predicted_values_rf], axis = 1)
```


```python
merged_rf_df['predicted_values_rf'] = merged_rf_df['predicted_values_rf'].round(0)
merged_rf_df['actual_values_rf'] = merged_rf_df['actual_values_rf'].round(0)
```


```python
merged_rf_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>record_ID</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-07-11</td>
      <td>41405</td>
      <td>9112</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>119.7000</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
      <td>152.0</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-07-11</td>
      <td>41474</td>
      <td>9164</td>
      <td>216418</td>
      <td>116.1375</td>
      <td>106.1625</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
      <td>83.0</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-07-11</td>
      <td>41451</td>
      <td>9147</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>118.9875</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>75.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-07-11</td>
      <td>41384</td>
      <td>9092</td>
      <td>216418</td>
      <td>69.1125</td>
      <td>69.1125</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>44.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-07-11</td>
      <td>41308</td>
      <td>8991</td>
      <td>216418</td>
      <td>69.8250</td>
      <td>69.8250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>59.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>2011-01-17</td>
      <td>1062</td>
      <td>9578</td>
      <td>216418</td>
      <td>97.6125</td>
      <td>98.3250</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.7125</td>
      <td>65.0</td>
      <td>129.0</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>2011-01-17</td>
      <td>1040</td>
      <td>9532</td>
      <td>216418</td>
      <td>89.0625</td>
      <td>89.0625</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>24.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>2011-01-17</td>
      <td>1082</td>
      <td>9672</td>
      <td>216418</td>
      <td>98.3250</td>
      <td>98.3250</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>88.0</td>
      <td>205.0</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>2011-01-17</td>
      <td>1102</td>
      <td>9611</td>
      <td>216418</td>
      <td>98.3250</td>
      <td>98.3250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>228.0</td>
      <td>132.0</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>2011-01-17</td>
      <td>1</td>
      <td>8091</td>
      <td>216418</td>
      <td>99.0375</td>
      <td>111.8625</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>12.8250</td>
      <td>20.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 15 columns</p>
</div>




```python
merged_rf_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1768 entries, 0 to 1767
    Data columns (total 15 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   week                 1768 non-null   datetime64[ns]
     1   record_ID            1768 non-null   int64         
     2   store_id             1768 non-null   int64         
     3   sku_id               1768 non-null   int64         
     4   total_price          1768 non-null   float64       
     5   base_price           1768 non-null   float64       
     6   is_featured_sku      1768 non-null   int64         
     7   is_display_sku       1768 non-null   int64         
     8   month                1768 non-null   int32         
     9   year                 1768 non-null   int32         
     10  day_of_week          1768 non-null   int32         
     11  day_of_month         1768 non-null   int32         
     12  discount             1768 non-null   float64       
     13  actual_values_rf     1768 non-null   float64       
     14  predicted_values_rf  1768 non-null   float64       
    dtypes: datetime64[ns](1), float64(5), int32(4), int64(5)
    memory usage: 179.7 KB



```python
merged_rf_df['week'] = merged_rf_df.apply(lambda row: '-'.join([str(row['year']), str(row['month']), str(row['day_of_month'])]), axis=1)
```


```python
merged_rf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>record_ID</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-7-11</td>
      <td>41405</td>
      <td>9112</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>119.7000</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
      <td>152.0</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-7-11</td>
      <td>41474</td>
      <td>9164</td>
      <td>216418</td>
      <td>116.1375</td>
      <td>106.1625</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
      <td>83.0</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-7-11</td>
      <td>41451</td>
      <td>9147</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>118.9875</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>75.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-7-11</td>
      <td>41384</td>
      <td>9092</td>
      <td>216418</td>
      <td>69.1125</td>
      <td>69.1125</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>44.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-7-11</td>
      <td>41308</td>
      <td>8991</td>
      <td>216418</td>
      <td>69.8250</td>
      <td>69.8250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>59.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_rf_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1768 entries, 0 to 1767
    Data columns (total 15 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   week                 1768 non-null   object 
     1   record_ID            1768 non-null   int64  
     2   store_id             1768 non-null   int64  
     3   sku_id               1768 non-null   int64  
     4   total_price          1768 non-null   float64
     5   base_price           1768 non-null   float64
     6   is_featured_sku      1768 non-null   int64  
     7   is_display_sku       1768 non-null   int64  
     8   month                1768 non-null   int32  
     9   year                 1768 non-null   int32  
     10  day_of_week          1768 non-null   int32  
     11  day_of_month         1768 non-null   int32  
     12  discount             1768 non-null   float64
     13  actual_values_rf     1768 non-null   float64
     14  predicted_values_rf  1768 non-null   float64
    dtypes: float64(5), int32(4), int64(5), object(1)
    memory usage: 179.7+ KB



```python
merged_rf_df['week'] = pd.to_datetime(merged_rf_df['week'])
```


```python
merged_rf_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1768 entries, 0 to 1767
    Data columns (total 15 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   week                 1768 non-null   datetime64[ns]
     1   record_ID            1768 non-null   int64         
     2   store_id             1768 non-null   int64         
     3   sku_id               1768 non-null   int64         
     4   total_price          1768 non-null   float64       
     5   base_price           1768 non-null   float64       
     6   is_featured_sku      1768 non-null   int64         
     7   is_display_sku       1768 non-null   int64         
     8   month                1768 non-null   int32         
     9   year                 1768 non-null   int32         
     10  day_of_week          1768 non-null   int32         
     11  day_of_month         1768 non-null   int32         
     12  discount             1768 non-null   float64       
     13  actual_values_rf     1768 non-null   float64       
     14  predicted_values_rf  1768 non-null   float64       
    dtypes: datetime64[ns](1), float64(5), int32(4), int64(5)
    memory usage: 179.7 KB



```python
merged_rf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>record_ID</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-07-11</td>
      <td>41405</td>
      <td>9112</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>119.7000</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
      <td>152.0</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-07-11</td>
      <td>41474</td>
      <td>9164</td>
      <td>216418</td>
      <td>116.1375</td>
      <td>106.1625</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
      <td>83.0</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-07-11</td>
      <td>41451</td>
      <td>9147</td>
      <td>216418</td>
      <td>118.9875</td>
      <td>118.9875</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>75.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-07-11</td>
      <td>41384</td>
      <td>9092</td>
      <td>216418</td>
      <td>69.1125</td>
      <td>69.1125</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>44.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-07-11</td>
      <td>41308</td>
      <td>8991</td>
      <td>216418</td>
      <td>69.8250</td>
      <td>69.8250</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>59.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns_to_drop_rf = ['record_ID', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku', 'month', 'year', 'day_of_week', 'day_of_month', 'discount']
```


```python
merged_rf_df = merged_rf_df.drop(columns=columns_to_drop_rf)
```


```python
merged_rf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-07-11</td>
      <td>9112</td>
      <td>216418</td>
      <td>152.0</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-07-11</td>
      <td>9164</td>
      <td>216418</td>
      <td>83.0</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-07-11</td>
      <td>9147</td>
      <td>216418</td>
      <td>75.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-07-11</td>
      <td>9092</td>
      <td>216418</td>
      <td>44.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-07-11</td>
      <td>8991</td>
      <td>216418</td>
      <td>59.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# merged_rf_df.set_index('week', inplace=True)
```


```python
merged_rf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-07-11</td>
      <td>9112</td>
      <td>216418</td>
      <td>152.0</td>
      <td>166.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-07-11</td>
      <td>9164</td>
      <td>216418</td>
      <td>83.0</td>
      <td>118.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-07-11</td>
      <td>9147</td>
      <td>216418</td>
      <td>75.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-07-11</td>
      <td>9092</td>
      <td>216418</td>
      <td>44.0</td>
      <td>48.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-07-11</td>
      <td>8991</td>
      <td>216418</td>
      <td>59.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
type(merged_rf_df)
```




    pandas.core.frame.DataFrame




```python
type(store_id)
```




    int




```python
merged_rf_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1768 entries, 0 to 1767
    Data columns (total 5 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   week                 1768 non-null   datetime64[ns]
     1   store_id             1768 non-null   int64         
     2   sku_id               1768 non-null   int64         
     3   actual_values_rf     1768 non-null   float64       
     4   predicted_values_rf  1768 non-null   float64       
    dtypes: datetime64[ns](1), float64(2), int64(2)
    memory usage: 69.2 KB



```python
condition1_rf = merged_rf_df['sku_id']  == sku_id
condition2_rf = merged_rf_df['store_id'] == store_id
```


```python
merged_rf_df = merged_rf_df[(condition1_rf.values) & (condition2_rf.values)]
merged_rf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>2011-07-11</td>
      <td>8091</td>
      <td>216418</td>
      <td>18.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2011-07-04</td>
      <td>8091</td>
      <td>216418</td>
      <td>6.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>2011-06-27</td>
      <td>8091</td>
      <td>216418</td>
      <td>17.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2011-06-20</td>
      <td>8091</td>
      <td>216418</td>
      <td>19.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>287</th>
      <td>2011-06-13</td>
      <td>8091</td>
      <td>216418</td>
      <td>33.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
comp_rf_df = merged_rf_df
```


```python
comp_rf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>2011-07-11</td>
      <td>8091</td>
      <td>216418</td>
      <td>18.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2011-07-04</td>
      <td>8091</td>
      <td>216418</td>
      <td>6.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>2011-06-27</td>
      <td>8091</td>
      <td>216418</td>
      <td>17.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2011-06-20</td>
      <td>8091</td>
      <td>216418</td>
      <td>19.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>287</th>
      <td>2011-06-13</td>
      <td>8091</td>
      <td>216418</td>
      <td>33.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create a figure and axis objects
fig, ax = plt.subplots()

# Plot two separate lines for each column
ax.plot(comp_rf_df['week'], comp_rf_df['actual_values_rf'], label='actual_values_rf')
ax.plot(comp_rf_df['week'], comp_rf_df['predicted_values_rf'], label='predicted_values_rf')

# Set labels and title
# Set labels and title
ax.set_xlabel('Weeks')
ax.set_ylabel('Demand')
ax.set_title('Actutal Values Vs Predicted values')

# Add legend
ax.legend()

# Store the plot in a variable
comp = fig

# Show the plot
plt.show()
```




    [<matplotlib.lines.Line2D at 0x287250810>]






    [<matplotlib.lines.Line2D at 0x28833f3d0>]






    Text(0.5, 0, 'Weeks')






    Text(0, 0.5, 'Demand')






    Text(0.5, 1.0, 'Actutal Values Vs Predicted values')






    <matplotlib.legend.Legend at 0x287294c50>




    
![png](output_102_6.png)
    



```python
comp
```




    
![png](output_103_0.png)
    



# Recurrent Neural network Model


```python
# Starting RNN (Recurrent Neural Network)
df_nrr = df_processed
df_nrr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>record_ID</th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>211530</td>
      <td>2013-07-09</td>
      <td>9112</td>
      <td>216418</td>
      <td>110.4375</td>
      <td>110.4375</td>
      <td>0</td>
      <td>0</td>
      <td>5.087596</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>211599</td>
      <td>2013-07-09</td>
      <td>9164</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>4.941642</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>211576</td>
      <td>2013-07-09</td>
      <td>9147</td>
      <td>216418</td>
      <td>109.7250</td>
      <td>109.7250</td>
      <td>0</td>
      <td>0</td>
      <td>4.304065</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>211509</td>
      <td>2013-07-09</td>
      <td>9092</td>
      <td>216418</td>
      <td>86.9250</td>
      <td>86.9250</td>
      <td>0</td>
      <td>0</td>
      <td>3.806662</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>211433</td>
      <td>2013-07-09</td>
      <td>8991</td>
      <td>216418</td>
      <td>87.6375</td>
      <td>87.6375</td>
      <td>0</td>
      <td>0</td>
      <td>4.143135</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop unnecessary columns
df_nrr = df_nrr.drop(columns=['record_ID', 'week'])  # Drop unnecessary columns
```


```python
# Normalize numerical features
scaler = StandardScaler()
df_nrr[['total_price', 'base_price']] = scaler.fit_transform(df_nrr[['total_price', 'base_price']])
```


```python
df_nrr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>units_sold</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>9112</td>
      <td>216418</td>
      <td>1.298610</td>
      <td>1.239105</td>
      <td>0</td>
      <td>0</td>
      <td>5.087596</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>9164</td>
      <td>216418</td>
      <td>1.248473</td>
      <td>1.183047</td>
      <td>0</td>
      <td>0</td>
      <td>4.941642</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>9147</td>
      <td>216418</td>
      <td>1.248473</td>
      <td>1.183047</td>
      <td>0</td>
      <td>0</td>
      <td>4.304065</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>9092</td>
      <td>216418</td>
      <td>-0.355896</td>
      <td>-0.610792</td>
      <td>0</td>
      <td>0</td>
      <td>3.806662</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>8991</td>
      <td>216418</td>
      <td>-0.305759</td>
      <td>-0.554734</td>
      <td>0</td>
      <td>0</td>
      <td>4.143135</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split data into features (X) and target (y)
X_nrr = df_nrr.drop(columns=['units_sold'])
y_nrr = df_nrr['units_sold']
```


```python
# Split data into training and testing sets
X_nrr_train, X_nrr_test, y_nrr_train, y_nrr_test = train_test_split(X_nrr, y_nrr, test_size=0.2, random_state=42)
```


```python
# Reshape input data for LSTM
X_nrr_train = np.array(X_nrr_train).reshape(X_nrr_train.shape[0], X_nrr_train.shape[1], 1)
X_nrr_test = np.array(X_nrr_test).reshape(X_nrr_test.shape[0], X_nrr_test.shape[1], 1)
```


```python
y_nrr.head()
```




    149375    5.087596
    149417    4.941642
    149403    4.304065
    149356    3.806662
    149301    4.143135
    Name: units_sold, dtype: float64




```python
#X_nrr.head()
```


```python
# Define the RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_nrr_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
```


```python
# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
```

    WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.



```python
# Train the model
model.fit(X_nrr_train, y_nrr_train, epochs=10, batch_size=32, validation_data=(X_nrr_test, y_nrr_test))
```

    Epoch 1/10
    221/221 [==============================] - 5s 13ms/step - loss: 1.4095 - val_loss: 0.7395
    Epoch 2/10
    221/221 [==============================] - 1s 7ms/step - loss: 0.7051 - val_loss: 0.5882
    Epoch 3/10
    221/221 [==============================] - 1s 6ms/step - loss: 0.6315 - val_loss: 0.5710
    Epoch 4/10
    221/221 [==============================] - 1s 7ms/step - loss: 0.6013 - val_loss: 0.5782
    Epoch 5/10
    221/221 [==============================] - 1s 7ms/step - loss: 0.6192 - val_loss: 0.5625
    Epoch 6/10
    221/221 [==============================] - 2s 7ms/step - loss: 0.6002 - val_loss: 0.5609
    Epoch 7/10
    221/221 [==============================] - 2s 7ms/step - loss: 0.5976 - val_loss: 0.5718
    Epoch 8/10
    221/221 [==============================] - 1s 6ms/step - loss: 0.5943 - val_loss: 0.5565
    Epoch 9/10
    221/221 [==============================] - 1s 7ms/step - loss: 0.5901 - val_loss: 0.5508
    Epoch 10/10
    221/221 [==============================] - 1s 6ms/step - loss: 0.5769 - val_loss: 0.5494





    <keras.src.callbacks.History at 0x288326110>



# Evaluate Recurrent Neural Network Model


```python
# Evaluate the model
y_nrr_pred = model.predict(X_nrr_test).flatten()
rmse_nrr = np.sqrt(mean_squared_error(y_nrr_test, y_nrr_pred))
mape_nrr = np.mean(np.abs((y_nrr_test - y_nrr_pred) / y_nrr_test)) * 100
loss_nrr = model.evaluate(X_nrr_test, y_nrr_test)

print("Test Loss:", loss_nrr)
print("Root Mean Squared Error (RMSE):", rmse_nrr)
print("Mean Absolute Percentage Error (MAPE):", mape_nrr)
```

    56/56 [==============================] - 1s 2ms/step
    56/56 [==============================] - 0s 2ms/step - loss: 0.5494
    Test Loss: 0.5494303107261658
    Root Mean Squared Error (RMSE): 0.7412356324232751
    Mean Absolute Percentage Error (MAPE): inf



```python
plot_predictions(y_nrr_test, y_nrr_pred)
```


    
![png](output_119_0.png)
    



```python
rmse, rmse_nrr
```




    (0.45974194406315844, 0.7412356324232751)




```python
# Find RMSE
mse_nrr = mean_squared_error(y_nrr_test, y_nrr_pred)
rmse_nrr = np.sqrt(mse_nrr)
print("RMSE:",rmse_nrr)
print("MSE:",mse_nrr)
```

    RMSE: 0.7412356324232751
    MSE: 0.5494302627739325



```python
#Evaluate accuracy using MAPE
y_nrr_true = np.array(y_nrr_test)
sumvalue=np.sum(y_nrr_true)
mape_nrr=np.sum(np.abs((y_nrr_true - y_nrr_pred)))/sumvalue*100
accuracy_nrr=100-mape_nrr
print('Accuracy:', round(accuracy_nrr,2),'%.')
```

    Accuracy: 86.53 %.



```python
y_nrr_pred
```




    array([4.3798623, 3.8713446, 3.7493665, ..., 4.435361 , 3.8918242,
           3.9191353], dtype=float32)




```python
y_nrr_true
```




    array([4.92725369, 3.55534806, 4.00733319, ..., 4.18965474, 4.20469262,
           3.33220451])




```python
actual_values_nrr = np.exp(y_nrr_true)
predicted_values_nrr = np.exp(y_nrr_pred)
comp_nrr = pd.DataFrame(data=[actual_values_nrr,predicted_values_nrr]).T
comp_nrr.columns=['y_nrr_test','y_nrr_pred']
comp_nrr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_nrr_test</th>
      <th>y_nrr_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>138.0</td>
      <td>79.827042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.0</td>
      <td>48.006889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>42.494156</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94.0</td>
      <td>35.908718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>157.0</td>
      <td>51.544071</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>114.0</td>
      <td>86.155014</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>62.0</td>
      <td>37.066128</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>66.0</td>
      <td>84.382576</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>67.0</td>
      <td>49.000195</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>28.0</td>
      <td>50.356884</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 2 columns</p>
</div>




```python
actual_values_nrr = pd.DataFrame(actual_values_nrr, columns=['actual_values_nrr'])
predicted_values_nrr = pd.DataFrame(predicted_values_nrr, columns = ['predicted_values_nrr'])
```


```python
X_nrr.head()
y_nrr.head()
X_nrr.shape
y_nrr.shape
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>149375</th>
      <td>9112</td>
      <td>216418</td>
      <td>1.298610</td>
      <td>1.239105</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149417</th>
      <td>9164</td>
      <td>216418</td>
      <td>1.248473</td>
      <td>1.183047</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149403</th>
      <td>9147</td>
      <td>216418</td>
      <td>1.248473</td>
      <td>1.183047</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149356</th>
      <td>9092</td>
      <td>216418</td>
      <td>-0.355896</td>
      <td>-0.610792</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>149301</th>
      <td>8991</td>
      <td>216418</td>
      <td>-0.305759</td>
      <td>-0.554734</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2013</td>
      <td>1</td>
      <td>9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>






    149375    5.087596
    149417    4.941642
    149403    4.304065
    149356    3.806662
    149301    4.143135
    Name: units_sold, dtype: float64






    (8840, 11)






    (8840,)




```python
# Calculate the number of rows to keep (20% of total rows)
num_rows_to_keep_nrr = int(len(X_nrr) * 0.2)

# Slice the DataFrame to keep the last 20% of the data
X_nrr_df_test = X_nrr[-num_rows_to_keep_nrr:]
y_nrr_df_test  = y_nrr[-num_rows_to_keep_nrr:]
```


```python
X_nrr_df_test.reset_index(inplace = True)
```


```python
X_nrr_df_test.head()
actual_values_nrr
predicted_values_nrr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29255</td>
      <td>9112</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.967852</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29297</td>
      <td>9164</td>
      <td>216418</td>
      <td>1.699702</td>
      <td>0.902760</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29283</td>
      <td>9147</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.911794</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29236</td>
      <td>9092</td>
      <td>216418</td>
      <td>-1.609309</td>
      <td>-2.012228</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29181</td>
      <td>8991</td>
      <td>216418</td>
      <td>-1.559173</td>
      <td>-1.956171</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_values_nrr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>138.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>94.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>157.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>114.0</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>62.0</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>66.0</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>67.0</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>28.0</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 1 columns</p>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_values_nrr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>79.827042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.006889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42.494156</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.908718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>51.544071</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>86.155014</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>37.066128</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>84.382576</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>49.000195</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>50.356884</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 1 columns</p>
</div>




```python
X_nrr_df_test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29255</td>
      <td>9112</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.967852</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29297</td>
      <td>9164</td>
      <td>216418</td>
      <td>1.699702</td>
      <td>0.902760</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29283</td>
      <td>9147</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.911794</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29236</td>
      <td>9092</td>
      <td>216418</td>
      <td>-1.609309</td>
      <td>-2.012228</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29181</td>
      <td>8991</td>
      <td>216418</td>
      <td>-1.559173</td>
      <td>-1.956171</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_nrr_df = pd.concat([X_nrr_df_test, actual_values_nrr, predicted_values_nrr], axis = 1)
```


```python
merged_nrr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
      <th>actual_values_nrr</th>
      <th>predicted_values_nrr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29255</td>
      <td>9112</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.967852</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
      <td>138.0</td>
      <td>79.827042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29297</td>
      <td>9164</td>
      <td>216418</td>
      <td>1.699702</td>
      <td>0.902760</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
      <td>35.0</td>
      <td>48.006889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29283</td>
      <td>9147</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.911794</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>55.0</td>
      <td>42.494156</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29236</td>
      <td>9092</td>
      <td>216418</td>
      <td>-1.609309</td>
      <td>-2.012228</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>94.0</td>
      <td>35.908718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29181</td>
      <td>8991</td>
      <td>216418</td>
      <td>-1.559173</td>
      <td>-1.956171</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>157.0</td>
      <td>51.544071</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>753</td>
      <td>9578</td>
      <td>216418</td>
      <td>0.396152</td>
      <td>0.286128</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.7125</td>
      <td>114.0</td>
      <td>86.155014</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>734</td>
      <td>9532</td>
      <td>216418</td>
      <td>-0.205486</td>
      <td>-0.442619</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>62.0</td>
      <td>37.066128</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>763</td>
      <td>9672</td>
      <td>216418</td>
      <td>0.446289</td>
      <td>0.286128</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>66.0</td>
      <td>84.382576</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>776</td>
      <td>9611</td>
      <td>216418</td>
      <td>0.446289</td>
      <td>0.286128</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>67.0</td>
      <td>49.000195</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>0</td>
      <td>8091</td>
      <td>216418</td>
      <td>0.496425</td>
      <td>1.351220</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>12.8250</td>
      <td>28.0</td>
      <td>50.356884</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 14 columns</p>
</div>




```python
merged_nrr_df['predicted_values_nrr'] = merged_nrr_df['predicted_values_nrr'].round(0)
merged_nrr_df['actual_values_nrr'] = merged_nrr_df['actual_values_nrr'].round(0)
```


```python
merged_nrr_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1768 entries, 0 to 1767
    Data columns (total 14 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   index                 1768 non-null   int64  
     1   store_id              1768 non-null   int64  
     2   sku_id                1768 non-null   int64  
     3   total_price           1768 non-null   float64
     4   base_price            1768 non-null   float64
     5   is_featured_sku       1768 non-null   int64  
     6   is_display_sku        1768 non-null   int64  
     7   month                 1768 non-null   int32  
     8   year                  1768 non-null   int32  
     9   day_of_week           1768 non-null   int32  
     10  day_of_month          1768 non-null   int32  
     11  discount              1768 non-null   float64
     12  actual_values_nrr     1768 non-null   float64
     13  predicted_values_nrr  1768 non-null   float32
    dtypes: float32(1), float64(4), int32(4), int64(5)
    memory usage: 159.0 KB



```python
merged_nrr_df['week'] = merged_nrr_df.apply(lambda row: '-'.join([str(row['year']), str(row['month']), str(row['day_of_month'])]), axis=1)
```


```python
merged_nrr_df['week'] = merged_nrr_df.apply(lambda row: '-'.join([str(row['year']), str(row['month']), str(row['day_of_month'])]), axis=1)
```


```python
merged_nrr_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
      <th>actual_values_nrr</th>
      <th>predicted_values_nrr</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29255</td>
      <td>9112</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.967852</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
      <td>138.0</td>
      <td>80.0</td>
      <td>2011-7-11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29297</td>
      <td>9164</td>
      <td>216418</td>
      <td>1.699702</td>
      <td>0.902760</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
      <td>35.0</td>
      <td>48.0</td>
      <td>2011-7-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29283</td>
      <td>9147</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.911794</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>55.0</td>
      <td>42.0</td>
      <td>2011-7-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29236</td>
      <td>9092</td>
      <td>216418</td>
      <td>-1.609309</td>
      <td>-2.012228</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>94.0</td>
      <td>36.0</td>
      <td>2011-7-11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29181</td>
      <td>8991</td>
      <td>216418</td>
      <td>-1.559173</td>
      <td>-1.956171</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>157.0</td>
      <td>52.0</td>
      <td>2011-7-11</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1763</th>
      <td>753</td>
      <td>9578</td>
      <td>216418</td>
      <td>0.396152</td>
      <td>0.286128</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.7125</td>
      <td>114.0</td>
      <td>86.0</td>
      <td>2011-1-17</td>
    </tr>
    <tr>
      <th>1764</th>
      <td>734</td>
      <td>9532</td>
      <td>216418</td>
      <td>-0.205486</td>
      <td>-0.442619</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>62.0</td>
      <td>37.0</td>
      <td>2011-1-17</td>
    </tr>
    <tr>
      <th>1765</th>
      <td>763</td>
      <td>9672</td>
      <td>216418</td>
      <td>0.446289</td>
      <td>0.286128</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>66.0</td>
      <td>84.0</td>
      <td>2011-1-17</td>
    </tr>
    <tr>
      <th>1766</th>
      <td>776</td>
      <td>9611</td>
      <td>216418</td>
      <td>0.446289</td>
      <td>0.286128</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>0.0000</td>
      <td>67.0</td>
      <td>49.0</td>
      <td>2011-1-17</td>
    </tr>
    <tr>
      <th>1767</th>
      <td>0</td>
      <td>8091</td>
      <td>216418</td>
      <td>0.496425</td>
      <td>1.351220</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2011</td>
      <td>0</td>
      <td>17</td>
      <td>12.8250</td>
      <td>28.0</td>
      <td>50.0</td>
      <td>2011-1-17</td>
    </tr>
  </tbody>
</table>
<p>1768 rows × 15 columns</p>
</div>




```python
merged_nrr_df['week'] = pd.to_datetime(merged_nrr_df['week'])
```


```python
merged_nrr_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>total_price</th>
      <th>base_price</th>
      <th>is_featured_sku</th>
      <th>is_display_sku</th>
      <th>month</th>
      <th>year</th>
      <th>day_of_week</th>
      <th>day_of_month</th>
      <th>discount</th>
      <th>actual_values_nrr</th>
      <th>predicted_values_nrr</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29255</td>
      <td>9112</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.967852</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.7125</td>
      <td>138.0</td>
      <td>80.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>29297</td>
      <td>9164</td>
      <td>216418</td>
      <td>1.699702</td>
      <td>0.902760</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>-9.9750</td>
      <td>35.0</td>
      <td>48.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>29283</td>
      <td>9147</td>
      <td>216418</td>
      <td>1.900248</td>
      <td>1.911794</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>55.0</td>
      <td>42.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29236</td>
      <td>9092</td>
      <td>216418</td>
      <td>-1.609309</td>
      <td>-2.012228</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>94.0</td>
      <td>36.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>29181</td>
      <td>8991</td>
      <td>216418</td>
      <td>-1.559173</td>
      <td>-1.956171</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2011</td>
      <td>0</td>
      <td>11</td>
      <td>0.0000</td>
      <td>157.0</td>
      <td>52.0</td>
      <td>2011-07-11</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns_to_drop_nrr = ['index', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku', 'month', 'year', 'day_of_week', 'day_of_month', 'discount']
```


```python
merged_nrr_df = merged_nrr_df.drop(columns=columns_to_drop_nrr)
```


```python
merged_nrr_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_nrr</th>
      <th>predicted_values_nrr</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9112</td>
      <td>216418</td>
      <td>138.0</td>
      <td>80.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9164</td>
      <td>216418</td>
      <td>35.0</td>
      <td>48.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9147</td>
      <td>216418</td>
      <td>55.0</td>
      <td>42.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9092</td>
      <td>216418</td>
      <td>94.0</td>
      <td>36.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8991</td>
      <td>216418</td>
      <td>157.0</td>
      <td>52.0</td>
      <td>2011-07-11</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged_nrr_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1768 entries, 0 to 1767
    Data columns (total 5 columns):
     #   Column                Non-Null Count  Dtype         
    ---  ------                --------------  -----         
     0   store_id              1768 non-null   int64         
     1   sku_id                1768 non-null   int64         
     2   actual_values_nrr     1768 non-null   float64       
     3   predicted_values_nrr  1768 non-null   float32       
     4   week                  1768 non-null   datetime64[ns]
    dtypes: datetime64[ns](1), float32(1), float64(1), int64(2)
    memory usage: 62.3 KB



```python
condition1_nrr = merged_nrr_df['sku_id']  == sku_id
condition2_nrr = merged_nrr_df['store_id'] == store_id
```


```python
merged_nrr_df = merged_nrr_df[(condition1_nrr.values) & (condition2_nrr.values)]
merged_nrr_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_nrr</th>
      <th>predicted_values_nrr</th>
      <th>week</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>8091</td>
      <td>216418</td>
      <td>24.0</td>
      <td>103.0</td>
      <td>2011-07-11</td>
    </tr>
    <tr>
      <th>83</th>
      <td>8091</td>
      <td>216418</td>
      <td>119.0</td>
      <td>38.0</td>
      <td>2011-07-04</td>
    </tr>
    <tr>
      <th>156</th>
      <td>8091</td>
      <td>216418</td>
      <td>182.0</td>
      <td>85.0</td>
      <td>2011-06-27</td>
    </tr>
    <tr>
      <th>212</th>
      <td>8091</td>
      <td>216418</td>
      <td>130.0</td>
      <td>83.0</td>
      <td>2011-06-20</td>
    </tr>
    <tr>
      <th>287</th>
      <td>8091</td>
      <td>216418</td>
      <td>374.0</td>
      <td>111.0</td>
      <td>2011-06-13</td>
    </tr>
  </tbody>
</table>
</div>




```python
comp_nrr_df = merged_nrr_df
```


```python
# Create a figure and axis objects
fig, ax = plt.subplots()

# Plot two separate lines for each column
ax.plot(comp_nrr_df['week'], comp_nrr_df['actual_values_nrr'], label='actual_values_nrr')
ax.plot(comp_nrr_df['week'], comp_nrr_df['predicted_values_nrr'], label='predicted_values_nrr')

# Set labels and title
# Set labels and title
ax.set_xlabel('Weeks')
ax.set_ylabel('Demand')
ax.set_title('Actutal Values Vs Predicted values')

# Add legend
ax.legend()

# Store the plot in a variable
comp_nrr = fig

# Show the plot
plt.show()
```




    [<matplotlib.lines.Line2D at 0x10672fbd0>]






    [<matplotlib.lines.Line2D at 0x28914f9d0>]






    Text(0.5, 0, 'Weeks')






    Text(0, 0.5, 'Demand')






    Text(0.5, 1.0, 'Actutal Values Vs Predicted values')






    <matplotlib.legend.Legend at 0x28b2016d0>




    
![png](output_148_6.png)
    



```python
comp_nrr
```




    
![png](output_149_0.png)
    




```python
rmse, rmse_nrr
```




    (0.45974194406315844, 0.7412356324232751)



# Comparison of both the models - Random Forest Vs Recurrent Neural Network


```python
# Create a DataFrame for evaluation
rmse_rf = round(rmse,2)
rmse_nrr = round(rmse_nrr,2)
accuracy_rf = str(round(accuracy,2)) + "%"
accuracy_nrr = str(round(accuracy_nrr,2)) + "%"

data_evaluation = {
    'Param':['RMSE', 'Accuracy'],
    'Random Forest': [rmse_rf, accuracy_rf],
    'Recursive Neural Network': [rmse_nrr,accuracy_nrr ],
}

index = ['0','1']
df_evaluation = pd.DataFrame(data_evaluation, index=index)

# Display the DataFrame
print(df_evaluation)
```

          Param Random Forest Recursive Neural Network
    0      RMSE          0.46                     0.74
    1  Accuracy        91.95%                   86.53%


# Re-order Point


```python
# Calculation of safety stock factor
def calculate_safety_factor(desired_service_level, standard_deviation):
    # Calculation Z-score corresponding to the desired service level
    z_score = stats.norm.ppf(desired_service_level)
    
    #Calculate safety factor
    safety_factor = z_score * standard_deviation
    
    return safety_factor
```


```python
# Get desired service level
#desired_service_level = float(input("Enter desired service level (ex: 0.95 for 95%): "))
desired_service_level = 0.9
```


```python
# Calculation of standard_deviation
filtered_df = df_processed[(df['store_id'] == store_id) & (df_processed['sku_id'] == sku_id)]
standard_deviation = filtered_df['units_sold'].std()
```


```python
# Calculation of re-order point
def calculate_reorder_point (demand_forecast, lead_time, safety_factor):
    average_demand = np.mean(demand_forecast)
    demand_std = np.std(demand_forecast)
    safety_stock = safety_factor * demand_std
    safety_stock = safety_stock.round()
    reorder_point = average_demand * lead_time + safety_stock
    reorder_point = reorder_point.round()
    return reorder_point, safety_stock
```


```python
# Get user input for sku_id
#lead_time = int(input("Enter lead time in weeks: "))
lead_time = 2
```


```python
comp_rf_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>week</th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>2011-07-11</td>
      <td>8091</td>
      <td>216418</td>
      <td>18.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2011-07-04</td>
      <td>8091</td>
      <td>216418</td>
      <td>6.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>156</th>
      <td>2011-06-27</td>
      <td>8091</td>
      <td>216418</td>
      <td>17.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>212</th>
      <td>2011-06-20</td>
      <td>8091</td>
      <td>216418</td>
      <td>19.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>287</th>
      <td>2011-06-13</td>
      <td>8091</td>
      <td>216418</td>
      <td>33.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Selection of model based on RMSE
if rmse_rf <= rmse_nrr:
    demand_forecast = pd.concat([X_test, actual_values_rf, predicted_values_rf], axis = 1)
    demand_forecast['predicted_values_rf'] = demand_forecast['predicted_values_rf'].round(0)
    demand_forecast = demand_forecast[(condition1_rf.values) & (condition2_rf.values)]
    demand_forecast = demand_forecast['predicted_values_rf']
    demand_forecast = demand_forecast.values
    selected_model = "Random Forest"
else:
    demand_forecast = pd.concat([X_nrr_df_test, actual_values_nrr, predicted_values_nrr], axis = 1)
    demand_forecast['predicted_values_nrr'] = demand_forecast['predicted_values_nrr'].round(0)
    demand_forecast = demand_forecast[(condition1_nrr.values) & (condition2_nrr.values)]
    demand_forecast = demand_forecast['predicted_values_nrr']
    demand_forecast = demand_forecast.values
    selected_model = "Recursive Neural Network"

```


```python
safety_factor = calculate_safety_factor(desired_service_level, standard_deviation)
```


```python
reorder_point = calculate_reorder_point (demand_forecast, lead_time, safety_factor)
```


```python
type(reorder_point)
```




    tuple




```python
reorder_point
```




    (58.0, 3.0)



# Frontend code


```python
data_dict = {'record_ID': 'Unique ID for each week store sku combination',     
             'week': 'Starting date of the week',     
             'store_id': 'Unique id for each store - Can be unique customers (EBs) for TBH',     
             'sku_id': 'Unique ID for each product - Can be the type of steel',     
             'total_price': 'Sales price of 1 product',     
             'base_price': 'Base price for 1 product',     
             'is_features__sku': 'Was part of the featured item of the week - Can be the product for which we sent out a marketing communication',     
             'is_display_sku': 'Product was displayed prominently - Can be the product which is highlighted on showcase page',     
             'units_sold': 'No. of units sold of the product at the store in the given week'}

input_dict = {'sku_id': 'Select the product for which you want the forecast and the re-order point',
             'store_id': 'Select the store for which you want the forecast and the re-order point',
             'lead_time': 'Time between realizing that the sku needs to be ordered for the store till it actually arrives at the store',
             'service_level': 'Service level that you desire to maintain for your customer'}

data_clean_steps_dict = {1: "Changed the format of the variables to the desired format",
                         2: "If the number of nan/missing values in the data is < 1% then deleted the rows else filled the rows with the mean for the given combination of sku_id and store_id",     
                         3: "Removed inconsistency in data (like record_id stored in total_price)",     
                         4: "Removed duplicate data (if any)",     
                         5: "If the total_price, base_price or units_sold had values <=0 then removed it if such cases are <=1% else replaced by the mean for the given combination of sku_id and store_id"}

unique_sku_ids = df['sku_id'].unique()
unique_store_ids = df['store_id'].unique()
comp_nrr_df = comp_nrr_df[['week'] + [col for col in comp_nrr_df.columns if col != 'week']]
#comp_rounded = comp.round(2)
```


```python
comp_rf_df_drop = comp_rf_df
comp_nrr_df_drop = comp_nrr_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-07-11</th>
      <td>8091</td>
      <td>216418</td>
      <td>18.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>2011-07-04</th>
      <td>8091</td>
      <td>216418</td>
      <td>6.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>2011-06-27</th>
      <td>8091</td>
      <td>216418</td>
      <td>17.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2011-06-20</th>
      <td>8091</td>
      <td>216418</td>
      <td>19.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>2011-06-13</th>
      <td>8091</td>
      <td>216418</td>
      <td>33.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>store_id</th>
      <th>sku_id</th>
      <th>actual_values_nrr</th>
      <th>predicted_values_nrr</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-07-11</th>
      <td>8091</td>
      <td>216418</td>
      <td>24.0</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>2011-07-04</th>
      <td>8091</td>
      <td>216418</td>
      <td>119.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2011-06-27</th>
      <td>8091</td>
      <td>216418</td>
      <td>182.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>2011-06-20</th>
      <td>8091</td>
      <td>216418</td>
      <td>130.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>2011-06-13</th>
      <td>8091</td>
      <td>216418</td>
      <td>374.0</td>
      <td>111.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
comp_rf_df_drop.set_index('week',inplace=True)
comp_nrr_df_drop.set_index('week',inplace=True)

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    /var/folders/yh/pl2cz6pd3rz655m2p297hdn40000gp/T/ipykernel_65959/2074298600.py in ?()
    ----> 1 comp_rf_df_drop.set_index('week',inplace=True)
          2 comp_nrr_df_drop.set_index('week',inplace=True)


    ~/anaconda3/lib/python3.11/site-packages/pandas/core/frame.py in ?(self, keys, drop, append, inplace, verify_integrity)
       5855                     if not found:
       5856                         missing.append(col)
       5857 
       5858         if missing:
    -> 5859             raise KeyError(f"None of {missing} are in the columns")
       5860 
       5861         if inplace:
       5862             frame = self


    KeyError: "None of ['week'] are in the columns"



```python
comp_rf_df_drop.head()
comp_nrr_df_drop.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_values_rf</th>
      <th>predicted_values_rf</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-07-11</th>
      <td>18.0</td>
      <td>21.0</td>
    </tr>
    <tr>
      <th>2011-07-04</th>
      <td>6.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>2011-06-27</th>
      <td>17.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2011-06-20</th>
      <td>19.0</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>2011-06-13</th>
      <td>33.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_values_nrr</th>
      <th>predicted_values_nrr</th>
    </tr>
    <tr>
      <th>week</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2011-07-11</th>
      <td>24.0</td>
      <td>103.0</td>
    </tr>
    <tr>
      <th>2011-07-04</th>
      <td>119.0</td>
      <td>38.0</td>
    </tr>
    <tr>
      <th>2011-06-27</th>
      <td>182.0</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>2011-06-20</th>
      <td>130.0</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>2011-06-13</th>
      <td>374.0</td>
      <td>111.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
drop_columns = ['store_id','sku_id']
```


```python
comp_rf_df_drop = comp_rf_df_drop.drop(columns = drop_columns)

```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[202], line 1
    ----> 1 comp_rf_df_drop = comp_rf_df_drop.drop(columns = drop_columns)
          2 comp_nrr_df_drop = comp_nrr_df_drop.drop(columns = drop_columns)


    File ~/anaconda3/lib/python3.11/site-packages/pandas/core/frame.py:5258, in DataFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       5110 def drop(
       5111     self,
       5112     labels: IndexLabel = None,
       (...)
       5119     errors: IgnoreRaise = "raise",
       5120 ) -> DataFrame | None:
       5121     """
       5122     Drop specified labels from rows or columns.
       5123 
       (...)
       5256             weight  1.0     0.8
       5257     """
    -> 5258     return super().drop(
       5259         labels=labels,
       5260         axis=axis,
       5261         index=index,
       5262         columns=columns,
       5263         level=level,
       5264         inplace=inplace,
       5265         errors=errors,
       5266     )


    File ~/anaconda3/lib/python3.11/site-packages/pandas/core/generic.py:4549, in NDFrame.drop(self, labels, axis, index, columns, level, inplace, errors)
       4547 for axis, labels in axes.items():
       4548     if labels is not None:
    -> 4549         obj = obj._drop_axis(labels, axis, level=level, errors=errors)
       4551 if inplace:
       4552     self._update_inplace(obj)


    File ~/anaconda3/lib/python3.11/site-packages/pandas/core/generic.py:4591, in NDFrame._drop_axis(self, labels, axis, level, errors, only_slice)
       4589         new_axis = axis.drop(labels, level=level, errors=errors)
       4590     else:
    -> 4591         new_axis = axis.drop(labels, errors=errors)
       4592     indexer = axis.get_indexer(new_axis)
       4594 # Case for non-unique axis
       4595 else:


    File ~/anaconda3/lib/python3.11/site-packages/pandas/core/indexes/base.py:6699, in Index.drop(self, labels, errors)
       6697 if mask.any():
       6698     if errors != "ignore":
    -> 6699         raise KeyError(f"{list(labels[mask])} not found in axis")
       6700     indexer = indexer[~mask]
       6701 return self.delete(indexer)


    KeyError: "['store_id', 'sku_id'] not found in axis"



```python
comp_nrr_df_drop = comp_nrr_df_drop.drop(columns = drop_columns)
```


```python
# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout =     html.Div(children=[
    html.H1(children='Demand Foreasting and Inventory Management'),
    html.H2(children='Predicting units sold for given SKU and Store'),
    
    
    html.Div([
    html.Div('Weekly Data Table'),
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody([
            html.Tr([
                html.Td(df.iloc[i][col]) for col in df.columns
            ]) for i in range(3)
        ])
    ])
]),
    
    html.Div([
        html.H3('Column Description:'),
        html.Ul([html.Li(f"{key}: {value}") for key, value in data_dict.items()])
    ]),
   
    html.H3('Time Period for consideration:'),
    html.Div(f'Maximum week period: {df_max_week}'),
    html.Div(f'Minimum week period: {df_min_week}'),
    
    html.Div(f'Maximum week period in training data set: {max_training_week}'),
    html.Div(f'Minimum week period in training data set: {min_training_week}'),

    
    html.Div([
        html.H3('Inputs required from customer:'),
        html.Ul([html.Li(f"{key}: {value}") for key, value in input_dict.items()])
    ]),
    
    html.Div([
        html.H3('Target Variable'),
    ]),
    html.Div(children='''
        Units Sold
    '''),
    html.H3('Input Parameters :'),
    
    html.Div(f'SKU ID: {sku_id}'),
    html.Div(f'Store ID: {store_id}'),
    
    
    #html.Table(
     #   [
      #      html.Tr(
       #         [html.Th(col, style={'border': '1px solid black', 'background-color': 'lightgray', 'padding': '8px'}) for col in comp_rf_df.columns]
        #    )
        #] +
        #[
         #   html.Tr(
          #      [
           #         html.Td(comp_rf_df.iloc[i][col], style={'border': '1px solid black', 'padding': '8px'}) 
            #        for col in comp_rf_df.columns
             #   ],
              #  style={'border': '1px solid black'}
            #)
            #for i in range(10)
        #],
        #style={'border-collapse': 'collapse'}
    #),
    html.Div([
    html.H4('Predicted Values of Random Forest'),
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in comp_rf_df.columns])),
        html.Tbody([
            html.Tr([
                html.Td(comp_rf_df.iloc[i][col]) for col in comp_rf_df.columns
            ]) for i in range(10)
        ])
    ])
]),
    dcc.Graph(
        id='line-plot',
        figure={
            'data': [
                {'x': list(range(1, len(comp_rf_df_drop)+1)), 'y': comp_rf_df_drop['actual_values_rf'], 'type': 'line', 'name': 'Test Data'},
                {'x': list(range(1, len(comp_rf_df_drop)+1)), 'y': comp_rf_df_drop['predicted_values_rf'], 'type': 'line', 'name': 'Predicted'}
            ],
            'layout': {
                'title': 'Test Data vs Predicted values for Random Forest',
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': 'Values'}
            }
        }
    ),
    
    html.Div([
    html.H4('Predicted Values of Recursive Neural Network'),
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in comp_nrr_df.columns])),
        html.Tbody([
            html.Tr([
                html.Td(comp_nrr_df.iloc[i][col]) for col in comp_nrr_df.columns
            ]) for i in range(10)
        ])
    ])
]),
    dcc.Graph(
        id='line-plot',
        figure={
            'data': [
                {'x': list(range(1, len(comp_nrr_df_drop)+1)), 'y': comp_nrr_df_drop['actual_values_nrr'], 'type': 'line', 'name': 'Test Data'},
                {'x': list(range(1, len(comp_nrr_df_drop)+1)), 'y': comp_nrr_df_drop['predicted_values_nrr'], 'type': 'line', 'name': 'Predicted'}
            ],
            'layout': {
                'title': 'Test Data vs Predicted values for Recursive Neural Network',
                'xaxis': {'title': 'Index'},
                'yaxis': {'title': 'Values'}
            }
        }
    ),
    
    html.H3('Ran 2 models to compare accuracy of forecast:'),
    
    html.Div([
    html.Div('Evaluation of the models'),
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df_evaluation.columns])),
        html.Tbody([
            html.Tr([
                html.Td(df_evaluation.iloc[i][col]) for col in df_evaluation.columns
            ]) for i in range(2)
        ])
    ]),
        html.H3(f'Selected Model is: {selected_model} due to lower RMSE value'),
        
        
        
    html.Div([
            html.H2('Optimal Reorder point and safety stock based on the selected model'),
            html.Div([
                html.Div(f'Reorder point: {reorder_point[0]}'),  # Display first tuple value
                html.Br(),  # Line break
                html.Div(f'Safety Stock: {reorder_point[1]}')   # Display second tuple value
            ])
        ]),
]),
    
])






    
if __name__ == '__main__':
    app.run_server(debug=True)
```

    ---------------------------------------------------------------------------
    DuplicateIdError                          Traceback (most recent call last)
    File ~/anaconda3/lib/python3.11/site-packages/flask/app.py:1818, in Flask.full_dispatch_request(self=<Flask '__main__'>)
       1816 try:
       1817     request_started.send(self)
    -> 1818     rv = self.preprocess_request()
            self = <Flask '__main__'>
       1819     if rv is None:
       1820         rv = self.dispatch_request()
    
    File ~/anaconda3/lib/python3.11/site-packages/flask/app.py:2309, in Flask.preprocess_request(self=<Flask '__main__'>)
       2307 if name in self.before_request_funcs:
       2308     for before_func in self.before_request_funcs[name]:
    -> 2309         rv = self.ensure_sync(before_func)()
            before_func = <bound method Dash._setup_server of <dash.dash.Dash object at 0x2a10cf2d0>>
            self = <Flask '__main__'>
       2311         if rv is not None:
       2312             return rv
    
    File ~/anaconda3/lib/python3.11/site-packages/dash/dash.py:1343, in Dash._setup_server(self=<dash.dash.Dash object>)
       1340 if not self.layout and self.use_pages:
       1341     self.layout = page_container
    -> 1343 _validate.validate_layout(self.layout, self._layout_value())
            self = <dash.dash.Dash object at 0x2a10cf2d0>
            _validate = <module 'dash._validate' from '/Users/deepakvarier/anaconda3/lib/python3.11/site-packages/dash/_validate.py'>
       1345 self._generate_scripts_html()
       1346 self._generate_css_dist_html()
    
    File ~/anaconda3/lib/python3.11/site-packages/dash/_validate.py:416, in validate_layout(
        layout=Div([H1('Demand Foreasting and Inventory Managem...8.0'), Br(None), Div('Safety Stock: 3.0')])])])]),
        layout_value=Div([H1('Demand Foreasting and Inventory Managem...8.0'), Br(None), Div('Safety Stock: 3.0')])])])])
    )
        414 component_id = stringify_id(getattr(component, "id", None))
        415 if component_id and component_id in component_ids:
    --> 416     raise exceptions.DuplicateIdError(
            exceptions = <module 'dash.exceptions' from '/Users/deepakvarier/anaconda3/lib/python3.11/site-packages/dash/exceptions.py'>
        417         f"""
        418         Duplicate component id found in the initial layout: `{component_id}`
        419         """
        420     )
        421 component_ids.add(component_id)
    
    DuplicateIdError: Duplicate component id found in the initial layout: `line-plot`
    




<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8050/"
    frameborder="0"
    allowfullscreen

></iframe>




```python

```
