#!/usr/bin/env python
# coding: utf-8

# # Import libraries and data

# In[1]:


# To print multiple output in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


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


# In[3]:


# Ignore all warings
warnings.filterwarnings("ignore")


# In[4]:


# Import data
# Import data
file_path = '/Users/deepakvarier/Downloads/hackathon_data'
date_format = "%d/%m/%y"
df = pd.read_csv(file_path+'/train.csv', sep = ',', parse_dates = ['week'], date_parser = lambda x: pd.to_datetime(x, format = date_format))


# # Data cleaning

# In[5]:


# Characteristics of data
df.head()
df.shape
df.info()


# In[157]:


df_max_week = df['week'].max()


# In[158]:


df_min_week = df['week'].min()


# In[8]:


# Check null values in the data
df.isnull().sum()


# In[9]:


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


# In[10]:


# Checking whether there are rows where the total_price or units_sold <=0
df.shape
df['total_price'].loc[df['total_price']<=0].count()
df['units_sold'].loc[df['units_sold']<=0].count()


# In[11]:


# Delete rows with negative rows
con1 = df['units_sold']<=0
con2 = df['total_price']<=0
df = df[~(con1 & con2)]
df.shape


# In[12]:


# Dropping duplicates if any
df.shape
df = df.drop_duplicates(['week', 'store_id', 'sku_id'])
df.shape


# In[13]:


# Sort dataframe by date column in chronological order
df = df.sort_values(by='week', ascending=False)
df.head()


# # Data Selection (Partly)

# In[14]:


# Function to create data frame for the selected store_id and sku_id
def create_dataframe(sku_id, df):
    # Filter the data for the specified store_id and sku_id
    filtered_data = df[(df['sku_id'] == sku_id)]

    # If no data is found for the specified sku_id, return None
    if filtered_data.empty:
        print("No data found for the specified sku_id.")
        return None

    return filtered_data


# In[15]:


# Get user input for sku_id
sku_id = int(input("Enter sku_id: "))
store_id = int(input("Enter store_id: "))

#sku_id=216425


# In[16]:


type(sku_id)


# In[17]:


# Call the function with user inputs to create dataframe of selected store_id and sku_id
df_selected = create_dataframe(sku_id,df)
if df_selected is not None:
    df_selected.head()
    df_selected.shape


# In[18]:


#df_selected = df_selected.drop(columns=['record_ID', 'store_id'])


# In[19]:


df_selected.head()


# In[20]:


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


# # Data Pre-processing

# In[21]:


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


# In[22]:


# Call the function to pre-process the data
df_processed = preprocess_data(df_selected)


# In[23]:


df_processed.head()


# In[24]:


#df_processed.drop(['week'], inplace=True, axis = 1)


# In[25]:


df_processed.head()


# In[26]:


# Check if the data is stationary
result = adfuller(df_processed['units_sold'].dropna())
# Print the test statistic and p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[27]:


# Since the p-value is below 0.05,
# the data can be assumed to be stationary hence we can proceed with the data without any transformation.


# In[28]:


df_processed.shape


# In[29]:


df_processed['units_sold'].skew()


# In[30]:


# units sold is highly positively skewed since skewness > 1


# In[31]:


df_processed.units_sold.hist()


# In[32]:


sns.kdeplot(df_processed.units_sold)


# In[33]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_processed['units_sold'])
plt.show()


# In[34]:


# Q-Q plot
stats.probplot(df_processed.units_sold, plot = pylab)


# In[35]:


# Tail of the data
df_processed.loc[df_processed['store_id']==store_id].tail()


# In[36]:


# Logarithmic transformation of data
df_processed['units_sold'] = np.log(df_processed['units_sold'])


# In[37]:


# Tail of the data
df_processed.loc[df_processed['store_id']==store_id].tail()


# In[38]:


df_processed['units_sold'].skew()


# In[39]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df_processed['units_sold'])
plt.subplot(1,2,2)
stats.probplot(df_processed['units_sold'], plot = pylab)
plt.show()


# In[40]:


# Finding the boundary values
UL = df_processed['units_sold'].mean() + 3*df_processed['units_sold'].std()
LL = df_processed['units_sold'].mean() - 3*df_processed['units_sold'].std()
UL
LL


# In[41]:


df_processed.shape


# In[42]:


df_processed['units_sold'].loc[df_processed['units_sold']<LL].count()


# In[43]:


df_processed['units_sold'].loc[df_processed['units_sold']>UL].count()


# In[44]:


# Removing outliers
condition1 = df_processed['units_sold']>UL
condition2 = df_processed['units_sold']<LL
df_processed = df_processed[~(condition1 & condition2)]


# # Understanding the components of the data

# In[45]:


# Seasonal decompose
df_processed.head()


# In[46]:


# Pre-processing for seasonal decompose
df_seasonal_decompose = df_processed
df_seasonal_decompose['week'] = pd.to_datetime(df_seasonal_decompose['week'])
df_seasonal_decompose = df_seasonal_decompose.set_index('week')
#store_id=8091
df_seasonal_decompose = df_seasonal_decompose[df_seasonal_decompose['store_id'] == store_id]


# In[47]:


# Seasonal decomposition
result_seasonal_decompose = seasonal_decompose(df_seasonal_decompose['units_sold'], model='additive', period=52)  # Assuming weekly seasonality


# In[48]:


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


# In[49]:


# Calculate metrics
#trend_mean = result.trend.mean()  # Mean of the trend component
#seasonal_mean = result.seasonal.mean()  # Mean of the seasonal component
#residual_std = result.resid.std()  # Standard deviation of the residual component

# Print insights
#print("Insights from Seasonal Decomposition:")
#print(f"Mean of Trend Component: {trend_mean}")
#print(f"Mean of Seasonal Component: {seasonal_mean}")
#print(f"Standard Deviation of Residual Component: {residual_std}")


# # Random Forest

# In[50]:


# Calculate the number of rows for testing
test_size = int(len(df_processed)*0.2)
end_point = len(df_processed)
x = end_point - test_size


# In[51]:


df_processed.shape
test_size
end_point
x


# In[52]:


# Split into train and test
df_processed_train = df_processed.iloc[:x - 1]
df_processed_test = df_processed.iloc[x:]


# In[161]:


max_training_week = df_processed_train['week'].max()
min_training_week = df_processed_train['week'].min()


# In[53]:


# Check shape of test and train
df_processed_train.shape
df_processed_test.shape


# In[54]:


# Processed data
df_processed_train.head()
df_processed_test.head()


# In[55]:


X_test = df_processed_test.loc[:, df_processed_test.columns != 'units_sold']
y_test = df_processed_test[['units_sold']]
X_train = df_processed_train.loc[:, df_processed_train.columns != 'units_sold']
y_train = df_processed_train[['units_sold']]


# In[56]:


X_test.head()
y_test.head()
X_train.head()
y_train.head()


# In[57]:


X_test.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)


# In[58]:


X_test_sarimax = X_test
y_test_sarimax = y_test
X_train_sarimax = X_train
y_train_sarimax = y_train


# In[59]:


X_test.head()
y_test.head()
X_train.head()
y_train.head()


# In[60]:


type(y_test)


# In[61]:


type(X_test)


# In[62]:


X_test.set_index('week', inplace=True)
X_train.set_index('week', inplace=True)


# In[63]:


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


# In[64]:


y_pred, fit = train_random_forest(X_train,y_train)


# In[65]:


y_pred


# # Evaluate Random Forest Model

# In[66]:


#Evaluate accuracy using MAPE
y_true = np.array(y_test['units_sold'])
sumvalue=np.sum(y_true)
mape=np.sum(np.abs((y_true - y_pred)))/sumvalue*100
accuracy=100-mape
print('Accuracy:', round(accuracy,2),'%.')


# In[67]:


# Find RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:",rmse)
print("MSE:",mse)


# In[68]:


def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel('Actual units_sold')
    plt.ylabel('Predicted units_sold')
    plt.title('Actual vs. Predicted units_sold')
    plt.show()


# In[69]:


y_test1 = y_test.values.flatten()


# In[70]:


y_test1


# In[71]:


actual_values_rf = np.exp(y_test1)
predicted_values_rf = np.exp(y_pred)
actual_values_rf = pd.DataFrame(actual_values_rf, columns=['actual_values_rf'])
predicted_values_rf = pd.DataFrame(predicted_values_rf, columns = ['predicted_values_rf'])


# In[72]:


actual_values_rf
predicted_values_rf
X_test.reset_index(inplace = True)


# In[73]:


merged_rf_df = pd.concat([X_test, actual_values_rf, predicted_values_rf], axis = 1)


# In[164]:


merged_rf_df['predicted_values_rf'] = merged_rf_df['predicted_values_rf'].round(0)
merged_rf_df['actual_values_rf'] = merged_rf_df['actual_values_rf'].round(0)


# In[75]:


merged_rf_df


# In[76]:


merged_rf_df.info()


# In[77]:


merged_rf_df['week'] = merged_rf_df.apply(lambda row: '-'.join([str(row['year']), str(row['month']), str(row['day_of_month'])]), axis=1)


# In[78]:


merged_rf_df.head()


# In[79]:


merged_rf_df.info()


# In[80]:


merged_rf_df['week'] = pd.to_datetime(merged_rf_df['week'])


# In[81]:


merged_rf_df.info()


# In[82]:


merged_rf_df.head()


# In[83]:


columns_to_drop_rf = ['record_ID', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku', 'month', 'year', 'day_of_week', 'day_of_month', 'discount']


# In[84]:


merged_rf_df = merged_rf_df.drop(columns=columns_to_drop_rf)


# In[85]:


merged_rf_df.head()


# In[86]:


# merged_rf_df.set_index('week', inplace=True)


# In[87]:


merged_rf_df.head()


# In[88]:


type(merged_rf_df)


# In[89]:


type(store_id)


# In[90]:


merged_rf_df.info()


# In[91]:


condition1_rf = merged_rf_df['sku_id']  == sku_id
condition2_rf = merged_rf_df['store_id'] == store_id


# In[92]:


merged_rf_df = merged_rf_df[(condition1_rf.values) & (condition2_rf.values)]
merged_rf_df.head()


# In[93]:


comp_rf_df = merged_rf_df


# In[94]:


comp_rf_df.head()


# In[95]:


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


# In[96]:


comp


# # Recurrent Neural network Model

# In[97]:


# Starting RNN (Recurrent Neural Network)
df_nrr = df_processed
df_nrr.head()


# In[98]:


# Drop unnecessary columns
df_nrr = df_nrr.drop(columns=['record_ID', 'week'])  # Drop unnecessary columns


# In[99]:


# Normalize numerical features
scaler = StandardScaler()
df_nrr[['total_price', 'base_price']] = scaler.fit_transform(df_nrr[['total_price', 'base_price']])


# In[100]:


df_nrr.head()


# In[101]:


# Split data into features (X) and target (y)
X_nrr = df_nrr.drop(columns=['units_sold'])
y_nrr = df_nrr['units_sold']


# In[102]:


# Split data into training and testing sets
X_nrr_train, X_nrr_test, y_nrr_train, y_nrr_test = train_test_split(X_nrr, y_nrr, test_size=0.2, random_state=42)


# In[103]:


# Reshape input data for LSTM
X_nrr_train = np.array(X_nrr_train).reshape(X_nrr_train.shape[0], X_nrr_train.shape[1], 1)
X_nrr_test = np.array(X_nrr_test).reshape(X_nrr_test.shape[0], X_nrr_test.shape[1], 1)


# In[104]:


y_nrr.head()


# In[105]:


#X_nrr.head()


# In[106]:


# Define the RNN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_nrr_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))


# In[107]:


# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')


# In[108]:


# Train the model
model.fit(X_nrr_train, y_nrr_train, epochs=100, batch_size=32, validation_data=(X_nrr_test, y_nrr_test))


# # Evaluate Recurrent Neural Network Model

# In[109]:


# Evaluate the model
y_nrr_pred = model.predict(X_nrr_test).flatten()
rmse_nrr = np.sqrt(mean_squared_error(y_nrr_test, y_nrr_pred))
mape_nrr = np.mean(np.abs((y_nrr_test - y_nrr_pred) / y_nrr_test)) * 100
loss_nrr = model.evaluate(X_nrr_test, y_nrr_test)

print("Test Loss:", loss_nrr)
print("Root Mean Squared Error (RMSE):", rmse_nrr)
print("Mean Absolute Percentage Error (MAPE):", mape_nrr)


# In[110]:


plot_predictions(y_nrr_test, y_nrr_pred)


# In[111]:


rmse, rmse_nrr


# In[112]:


# Find RMSE
mse_nrr = mean_squared_error(y_nrr_test, y_nrr_pred)
rmse_nrr = np.sqrt(mse_nrr)
print("RMSE:",rmse_nrr)
print("MSE:",mse_nrr)


# In[113]:


#Evaluate accuracy using MAPE
y_nrr_true = np.array(y_nrr_test)
sumvalue=np.sum(y_nrr_true)
mape_nrr=np.sum(np.abs((y_nrr_true - y_nrr_pred)))/sumvalue*100
accuracy_nrr=100-mape_nrr
print('Accuracy:', round(accuracy_nrr,2),'%.')


# In[114]:


y_nrr_pred


# In[115]:


y_nrr_true


# In[116]:


actual_values_nrr = np.exp(y_nrr_true)
predicted_values_nrr = np.exp(y_nrr_pred)
comp_nrr = pd.DataFrame(data=[actual_values_nrr,predicted_values_nrr]).T
comp_nrr.columns=['y_nrr_test','y_nrr_pred']
comp_nrr


# In[117]:


actual_values_nrr = pd.DataFrame(actual_values_nrr, columns=['actual_values_nrr'])
predicted_values_nrr = pd.DataFrame(predicted_values_nrr, columns = ['predicted_values_nrr'])


# In[118]:


X_nrr.head()
y_nrr.head()
X_nrr.shape
y_nrr.shape


# In[119]:


# Calculate the number of rows to keep (20% of total rows)
num_rows_to_keep_nrr = int(len(X_nrr) * 0.2)

# Slice the DataFrame to keep the last 20% of the data
X_nrr_df_test = X_nrr[-num_rows_to_keep_nrr:]
y_nrr_df_test  = y_nrr[-num_rows_to_keep_nrr:]


# In[120]:


X_nrr_df_test.reset_index(inplace = True)


# In[121]:


X_nrr_df_test.head()
actual_values_nrr
predicted_values_nrr


# In[122]:


X_nrr_df_test.head()


# In[123]:


merged_nrr_df = pd.concat([X_nrr_df_test, actual_values_nrr, predicted_values_nrr], axis = 1)


# In[124]:


merged_nrr_df


# In[165]:


merged_nrr_df['predicted_values_nrr'] = merged_nrr_df['predicted_values_nrr'].round(0)
merged_nrr_df['actual_values_nrr'] = merged_nrr_df['actual_values_nrr'].round(0)


# In[126]:


merged_nrr_df.info()


# In[127]:


merged_nrr_df['week'] = merged_nrr_df.apply(lambda row: '-'.join([str(row['year']), str(row['month']), str(row['day_of_month'])]), axis=1)


# In[128]:


merged_nrr_df['week'] = merged_nrr_df.apply(lambda row: '-'.join([str(row['year']), str(row['month']), str(row['day_of_month'])]), axis=1)


# In[129]:


merged_nrr_df


# In[130]:


merged_nrr_df['week'] = pd.to_datetime(merged_nrr_df['week'])


# In[131]:


merged_nrr_df.head()


# In[132]:


columns_to_drop_nrr = ['index', 'total_price', 'base_price', 'is_featured_sku', 'is_display_sku', 'month', 'year', 'day_of_week', 'day_of_month', 'discount']


# In[133]:


merged_nrr_df = merged_nrr_df.drop(columns=columns_to_drop_nrr)


# In[134]:


merged_nrr_df.head()


# In[135]:


merged_nrr_df.info()


# In[136]:


condition1_nrr = merged_nrr_df['sku_id']  == sku_id
condition2_nrr = merged_nrr_df['store_id'] == store_id


# In[137]:


merged_nrr_df = merged_nrr_df[(condition1_nrr.values) & (condition2_nrr.values)]
merged_nrr_df.head()


# In[138]:


comp_nrr_df = merged_nrr_df


# In[139]:


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


# In[140]:


comp_nrr


# In[141]:


rmse, rmse_nrr


# # Comparison of both the models - Random Forest Vs Recurrent Neural Network

# In[142]:


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


# # Re-order Point

# In[143]:


# Calculation of safety stock factor
def calculate_safety_factor(desired_service_level, standard_deviation):
    # Calculation Z-score corresponding to the desired service level
    z_score = stats.norm.ppf(desired_service_level)
    
    #Calculate safety factor
    safety_factor = z_score * standard_deviation
    
    return safety_factor


# In[144]:


# Get desired service level
#desired_service_level = float(input("Enter desired service level (ex: 0.95 for 95%): "))
desired_service_level = 0.95


# In[145]:


# Calculation of standard_deviation
filtered_df = df_processed[(df['store_id'] == store_id) & (df_processed['sku_id'] == sku_id)]
standard_deviation = filtered_df['units_sold'].std()


# In[168]:


# Calculation of re-order point
def calculate_reorder_point (demand_forecast, lead_time, safety_factor):
    average_demand = np.mean(demand_forecast)
    demand_std = np.std(demand_forecast)
    safety_stock = safety_factor * demand_std
    safety_stock = safety_stock.round()
    reorder_point = average_demand * lead_time + safety_stock
    reorder_point = reorder_point.round()
    return reorder_point, safety_stock


# In[147]:


# Get user input for sku_id
#lead_time = int(input("Enter lead time in weeks: "))
lead_time = 2


# In[148]:


comp_rf_df.head()


# In[175]:


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


# In[150]:


safety_factor = calculate_safety_factor(desired_service_level, standard_deviation)


# In[151]:


reorder_point = calculate_reorder_point (demand_forecast, lead_time, safety_factor)


# In[170]:


type(reorder_point)


# In[169]:


reorder_point


# # Frontend code

# In[230]:


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
             'store_id': 'Select the store for which you want the forecast and the re-order point'}

assumption_dict = {'lead_time': 'Lead time is 2 weeks',
             'service_level': 'Desired service level is 95%'}

data_clean_steps_dict = {1: "Changed the format of the variables to the desired format",
                         2: "If the number of nan/missing values in the data is < 1% then deleted the rows else filled the rows with the mean for the given combination of sku_id and store_id",     
                         3: "Removed inconsistency in data (like record_id stored in total_price)",     
                         4: "Removed duplicate data (if any)",     
                         5: "If the total_price, base_price or units_sold had values <=0 then removed it if such cases are <=1% else replaced by the mean for the given combination of sku_id and store_id"}

unique_sku_ids = df['sku_id'].unique()
unique_store_ids = df['store_id'].unique()
comp_nrr_df = comp_nrr_df[['week'] + [col for col in comp_nrr_df.columns if col != 'week']]
#comp_rounded = comp.round(2)


# In[196]:


comp_rf_df_drop = comp_rf_df
comp_nrr_df_drop = comp_nrr_df


# In[197]:


comp_rf_df_drop.set_index('week',inplace=True)
comp_nrr_df_drop.set_index('week',inplace=True)


# In[205]:


comp_rf_df_drop.head()
comp_nrr_df_drop.head()


# In[199]:


drop_columns = ['store_id','sku_id']


# In[202]:


comp_rf_df_drop = comp_rf_df_drop.drop(columns = drop_columns)


# In[204]:


comp_nrr_df_drop = comp_nrr_df_drop.drop(columns = drop_columns)


# In[217]:


# Define styles for SKU ID and Store ID lines
sku_id_style = {'color': 'blue', 'font-size': '16px', 'font-weight': 'bold'}  # Example styles for SKU ID
store_id_style = {'color': 'green', 'font-size': '16px', 'font-weight': 'bold'}  # Example styles for Store ID


# In[231]:


# Create Dash app
app = dash.Dash(__name__)

# Define layout
app.layout =     html.Div(children=[
    html.H1(children='Demand Foreasting and Inventory Management'),
    html.H2(children='Predicting units sold for given SKU and Store'),
    
    html.Div([
    html.H4('sales data format', style={'textAlign': 'center', 'marginBottom': '20px'}),  # Center the heading and add bottom margin
    html.Table(
        children=[
            html.Thead(
                html.Tr([html.Th(col, style={'backgroundColor': 'lightblue', 'textAlign': 'center', 'padding': '10px'}) for col in df.columns])  # Apply background color, center align text, and add padding to header cells
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df.iloc[i][col], style={'textAlign': 'center', 'padding': '8px'}) for col in df.columns  # Center align text and add padding to body cells
                ]) for i in range(5)  # Display only first 5 rows for demonstration
            ])
        ],
        style={'width': '100%', 'borderCollapse': 'collapse'}  # Set table width to 100% and collapse borders
    )
]),
    
    
     
    html.Div([
        html.H3('Target Variable'),
    ]),
    html.Div([
    html.Div(f'Units Sold', style=sku_id_style),  # Apply style to SKU ID line
]),
    
    html.H3('Input Parameters :'),
    
    html.Div([
    html.Div(f'SKU ID: {sku_id}', style=store_id_style),  # Apply style to SKU ID line
    html.Div(f'Store ID: {store_id}', style=store_id_style),  # Apply style to Store ID line
]),
    
    
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
    html.H4('Predicted Values of Random Forest', style={'textAlign': 'center', 'marginBottom': '20px'}),  # Center the heading and add bottom margin
    html.Table(
        children=[
            html.Thead(
                html.Tr([html.Th(col, style={'backgroundColor': 'lightblue', 'textAlign': 'center', 'padding': '10px'}) for col in comp_rf_df.columns])  # Apply background color, center align text, and add padding to header cells
            ),
            html.Tbody([
                html.Tr([
                    html.Td(comp_rf_df.iloc[i][col], style={'textAlign': 'center', 'padding': '8px'}) for col in comp_rf_df.columns  # Center align text and add padding to body cells
                ]) for i in range(5)  # Display only first 5 rows for demonstration
            ])
        ],
        style={'width': '100%', 'borderCollapse': 'collapse'}  # Set table width to 100% and collapse borders
    )
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
    html.H4('Predicted Values of Recursive Neural Network', style={'textAlign': 'center', 'marginBottom': '20px'}),  # Center the heading and add bottom margin
    html.Table(
        children=[
            html.Thead(
                html.Tr([html.Th(col, style={'backgroundColor': 'lightblue', 'textAlign': 'center', 'padding': '10px'}) for col in comp_nrr_df.columns])  # Apply background color, center align text, and add padding to header cells
            ),
            html.Tbody([
                html.Tr([
                    html.Td(comp_nrr_df.iloc[i][col], style={'textAlign': 'center', 'padding': '8px'}) for col in comp_nrr_df.columns  # Center align text and add padding to body cells
                ]) for i in range(5)  # Display only first 5 rows for demonstration
            ])
        ],
        style={'width': '100%', 'borderCollapse': 'collapse'}  # Set table width to 100% and collapse borders
    )
]),
    dcc.Graph(
        id='nrr-line-plot',
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
    html.Div([
    html.H2('Evaluation Metrics', style={'textAlign': 'center', 'color': 'navy', 'fontFamily': 'Arial, sans-serif', 'marginTop': '20px', 'marginBottom': '20px'}),  # Center the heading and apply additional styling
    html.Table(
        children=[
            html.Thead(
                html.Tr([html.Th(col, style={'backgroundColor': 'lightblue', 'textAlign': 'center', 'padding': '10px', 'border': '1px solid black'}) for col in df_evaluation.columns])  # Apply background color, center align text, add padding, and border to header cells
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df_evaluation.iloc[i][col], style={'textAlign': 'center', 'padding': '8px', 'border': '1px solid black'}) for col in df_evaluation.columns  # Center align text, add padding, and border to body cells
                ]) for i in range(len(df_evaluation))  # Loop through all rows of DataFrame
            ])
        ],
        style={'width': '80%', 'margin': '0 auto', 'borderCollapse': 'collapse', 'border': '1px solid black'}  # Set table width to 80%, center the table, collapse borders, and add border to the table
    )
]),
        html.H3(f'Selected Model is: {selected_model} due to lower RMSE value'),
        
        
        
    html.Div([
    html.H2('Optimal Reorder Point and Safety Stock Based on the Selected Model', style={'textAlign': 'center', 'color': 'navy', 'fontFamily': 'Arial, sans-serif', 'marginTop': '20px', 'marginBottom': '20px'}),  # Center the heading and apply additional styling
    html.Div([
        html.Div(f'Reorder Point: {reorder_point[0]}', style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px'}),  # Apply style to reorder point line
        html.Div(f'Safety Stock: {reorder_point[1]}', style={'fontSize': '18px', 'fontWeight': 'bold'}),  # Apply style to safety stock line
    ], style={'border': '2px solid #ccc', 'borderRadius': '10px', 'padding': '20px'})  # Apply border, border radius, and padding to the div
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
        html.H3('Assumptions:'),
        html.Ul([html.Li(f"{key}: {value}") for key, value in assumption_dict.items()])
    ]),
   
]),
    
])


    
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




