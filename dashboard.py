#!/usr/bin/env python
# coding: utf-8

# In[142]:


# To print multiple output in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[174]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
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
import warnings # To handle warnings
warnings.filterwarnings("ignore") # Ignore all warings
from statsmodels.tsa.statespace.sarimax import SARIMAX # To do SARIMAX
from sklearn.model_selection import train_test_split # To split into train and test data set
from sklearn.preprocessing import StandardScaler # For RNN: Recursive neural network
from keras.models import Sequential # For RNN
from keras.layers import LSTM, Dense, Dropout # For RNN
from keras.optimizers import Adam # For RNN


# In[150]:


# Step 1: Data Collection and Preprocessing
df_x_y = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 11, 12, 13, 14]
})
data = pd.read_csv('/Users/deepakvarier/Downloads/hackathon_data/train.csv')
data = data.dropna(subset=['total_price'])
data.isnull().sum()
df = data

df_city = pd.DataFrame({
    'Name': ['John', 'Alice', 'Bob', 'Emily'],
    'Age': [30, 25, 40, 35],
    'City': ['New York', 'Los Angeles', 'Chicago', 'San Francisco']
})

par_data = pd.DataFrame({
    'Param': ['Minimum time period', 'Maximum time period', 'Number of rows'],
    'Value': ['17-01-2011', '09-07-2013', '150150'],
})

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


target_variable = 0
# Sample data
x_data = ['Category A', 'Category B', 'Category C']
y_data = [20, 14, 23]
data.head()
par_data.head()


# In[145]:


# Ignore all warings
warnings.filterwarnings("ignore")


# In[146]:


# Characteristics of data
df.head()
df.shape
df.info()


# In[151]:


# Check null values in the data
df.isnull().sum()


# In[152]:


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


# In[153]:


# Checking whether there are rows where the total_price or units_sold <=0
df.shape
df['total_price'].loc[df['total_price']<=0].count()
df['units_sold'].loc[df['units_sold']<=0].count()


# In[154]:


data['sku_id'].nunique()


# In[155]:


data['store_id'].nunique()


# In[156]:


# Get user input for sku_id
storeid = int(input("Enter store_id: "))


# In[157]:


# Delete rows with negative rows
con1 = df['units_sold']<=0
con2 = df['total_price']<=0
df = df[~(con1 & con2)]
df.shape


# In[158]:


# Dropping duplicates if any
df.shape
df = df.drop_duplicates(['week', 'store_id', 'sku_id'])
df.shape


# In[159]:


# Sort dataframe by date column in chronological order
df = df.sort_values(by='week', ascending=False)
df.head()


# In[160]:


# Function to create data frame for the selected store_id and sku_id
def create_dataframe(sku_id, df):
    # Filter the data for the specified store_id and sku_id
    filtered_data = df[(df['sku_id'] == sku_id)]

    # If no data is found for the specified sku_id, return None
    if filtered_data.empty:
        print("No data found for the specified sku_id.")
        return None

    return filtered_data


# In[161]:


# Get user input for sku_id
sku_id = int(input("Enter sku_id: "))


# In[162]:


# Call the function with user inputs to create dataframe of selected store_id and sku_id
df_selected = create_dataframe(sku_id,df)
if df_selected is not None:
    df_selected.head()
    df_selected.shape


# In[163]:


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


# In[164]:


# Call the function to pre-process the data
df_processed = preprocess_data(df_selected)


# In[165]:


# Check if the data is stationary
result = adfuller(df_processed['units_sold'].dropna())
# Print the test statistic and p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[166]:


df_processed['units_sold'].skew()


# In[181]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_processed['units_sold'])
plt.subplot(1,2,2)
stats.probplot(df_processed['units_sold'], plot = pylab)
plt.show()


# In[182]:


# Finding the boundary values
UL = df_processed['units_sold'].mean() + 3*df_processed['units_sold'].std()
LL = df_processed['units_sold'].mean() - 3*df_processed['units_sold'].std()
UL
LL


# In[31]:


def create_dataframe(store_id, df):
    # Filter the data for the specified store_id and sku_id
    filtered_data = df[(df['store_id'] == store_id)]

    # If no data is found for the specified sku_id, return None
    if filtered_data.empty:
        print("No data found for the specified store_id.")
        return None

    return filtered_data


# In[28]:


desired_service_level = float(input("Enter desired service level (ex: 0.95 for 95%): "))


# In[ ]:


df_processed.units_sold.hist()


# In[ ]:


sns.kdeplot(df_processed.units_sold)


# In[ ]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_processed['units_sold'])
plt.show()


# In[ ]:


# Q-Q plot
stats.probplot(df_processed.units_sold, plot = pylab)


# In[ ]:


# Tail of the data
df_processed.loc[df_processed['store_id']==8091].tail()


# In[ ]:


# Logarithmic transformation of data
df_processed['units_sold'] = np.log(df_processed['units_sold'])


# In[ ]:


# Tail of the data
df_processed.loc[df_processed['store_id']==8091].tail()


# In[ ]:


df_processed['units_sold'].skew()


# In[ ]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_processed['units_sold'])
plt.subplot(1,2,2)
stats.probplot(df_processed['units_sold'], plot = pylab)
plt.show()


# In[ ]:


# Finding the boundary values
UL = df_processed['units_sold'].mean() + 3*df_processed['units_sold'].std()
LL = df_processed['units_sold'].mean() - 3*df_processed['units_sold'].std()
UL
LL


# In[ ]:


df_processed['units_sold'].loc[df_processed['units_sold']<LL].count()


# In[ ]:


df_processed['units_sold'].loc[df_processed['units_sold']>UL].count()


# In[ ]:


# Removing outliers
condition1 = df_processed['units_sold']>UL
condition2 = df_processed['units_sold']<LL
df_processed = df_processed[~(condition1 & condition2)]


# In[ ]:


# Calculate the number of rows for testing
test_size = int(len(df_processed)*0.2)
end_point = len(df_processed)
x = end_point - test_size


# In[ ]:


# Split into train and test
df_processed_train = df_processed.iloc[:x - 1]
df_processed_test = df_processed.iloc[x:]


# In[ ]:


# Check shape of test and train
df_processed_train.shape
df_processed_test.shape


# In[ ]:


# Processed data
df_processed_train.head()
df_processed_test.head()


# In[ ]:


X_test = df_processed_test.loc[:, df_processed_test.columns != 'units_sold']
y_test = df_processed_test[['units_sold']]
X_train = df_processed_train.loc[:, df_processed_train.columns != 'units_sold']
y_train = df_processed_train[['units_sold']]


# In[ ]:


X_test.head()
y_test.head()
X_train.head()
y_train.head()


# In[183]:


# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout =     html.Div(children=[
    html.H1(children='Demand Foreasting and Inventory Management'),
    html.H2(children='Predicting units sold for given SKU and Store'),
    
    #html.Label('Filter by Store:'),
    #dcc.Dropdown(
     #   id='store-filter',
      #  options=[{'label': store, 'value': store} for store in data['store_id'].unique()],
       # value=None
    #),
    html.Div([
    html.Div('Weekly Data Table'),
    html.Table([
        html.Thead(html.Tr([html.Th(col) for col in data.columns])),
        html.Tbody([
            html.Tr([
                html.Td(data.iloc[i][col]) for col in data.columns
            ]) for i in range(3)
        ])
    ])
]),
    html.Div([
        html.H3('Column Description:'),
        html.Ul([html.Li(f"{key}: {value}") for key, value in data_dict.items()])
    ]),
   
    html.H3('Time Period for consideration:'),
    html.Div(children='''
        Minimum time period of data - 17-01-2011
    '''),
    html.Div(children='''
        Maximum time period of data - 09-07-2013
    '''),
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
    
    html.Div([
        html.Label('SKU ID:'),
        dcc.Input(id='input1', type='number', value=0),
    ], style={'display': 'inline-block', 'width': '20%', 'margin-right': '10px'}),
    
    html.Div([
        html.Label('Store ID:'),
        dcc.Input(id='input2', type='number', value=0),
    ], style={'display': 'inline-block', 'width': '20%', 'margin-right': '10px'}),
    
    
    html.Div([
        html.Label('Lead Time:'),
        dcc.Input(id='input3', type='number', value=0),
    ], style={'display': 'inline-block', 'width': '20%', 'margin-right': '10px'}),
    
    html.Div([
        html.Label('Service Levels:'),
        dcc.Input(id='input4', type='number', value=0),
    ], style={'display': 'inline-block', 'width': '20%', 'margin-right': '10px'}),
    
     html.Div(style={'height': '20px'}),
    
    html.Div([
        html.Button('Submit', id='submit-button', n_clicks=0)
    ]),
    
    html.Div(id='output'),
    
    html.Div([
        html.H3('Data Cleaning Steps:'),
        html.Ul([html.Li(f"{value}") for key, value in data_clean_steps_dict.items()])
    ]),
    
    dcc.Graph(id='bar-chart'),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                go.Scatter(
                    x=df_x_y['x'],
                    y=df_x_y['y'],
                    mode='lines+markers',
                    name='Line chart'
                )
            ],
            'layout': {
                'title': 'Line chart visualization'
            }
        }
    ),
    html.H5('Histogram :'),

    dcc.Graph(id='histogram'),
    
    html.H1('Bar Graph Example'),
        dcc.Graph(
            id='bar-graph',
        figure={
            'data': [
                go.Bar(
                    x=x_data,
                    y=y_data,
                    marker_color='rgb(55, 83, 109)',
                    opacity=0.7
                )
            ],
            'layout': go.Layout(
                title='Bar Graph',
                xaxis={'title': 'Categories'},
                yaxis={'title': 'Values'},
                margin={'l': 40, 'b': 40, 't': 40, 'r': 40},
                hovermode='closest'
            )
        }
    ),
])

# Define callback to update output
@app.callback(
    Output('output', 'children'),
    [Input('submit-button', 'n_clicks')],
    [Input('input1', 'value'),
     Input('input2', 'value'),
     Input('input3', 'value'),
     Input('input4', 'value')]
)
def update_output(n_clicks, input1, input2, input3, input4):
    if n_clicks > 0:
        if n_clicks > 0:
            target_variable = {
                'Parameter 1': input1,
                'Parameter 2': input2,
                'Parameter 3': input3,
                'Parameter 4': input4
            }
            result1 = create_dataframe(input2,data)
            
            return f'Stored inputs: {target_variable}'

def update_histogram(selected_category):
    filtered_df = df_processed[df_processed['store_id'] == input2]
    fig = go.Figure(data=[go.Histogram(x=filtered_df['units_sold'])])
    fig.update_layout(title=f'Histogram for Category {selected_category}')
    
    return fig
    
if __name__ == '__main__':
    app.run_server(debug=True)


# In[36]:





# In[17]:


storeid


# In[19]:


storeid = int(storeid)


# In[20]:


type(storeid)


# In[ ]:


# Call the function with user inputs to create dataframe of selected store_id and sku_id
df_selected = create_dataframe(storeid,data)
if df_selected is not None:
    df_selected.head()
    df_selected.shape


# In[37]:





# In[ ]:




