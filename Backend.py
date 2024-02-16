{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8097987a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# To print multiple output in a cell\n",
    "from IPython.core.interactiveshell import InteractiveShell\n",
    "InteractiveShell.ast_node_interactivity = 'all'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ece7fc6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import all required libraries\n",
    "import pandas as pd # Data manipulation and analysis library\n",
    "from sklearn.feature_selection import RFE # RFE (Recursive Feature Elimination) is for feature selection\n",
    "from sklearn.ensemble import RandomForestRegressor # Random forest modelling\n",
    "import numpy as np # For arrays and mathematical operations\n",
    "from statsmodels.tsa.stattools import adfuller # Dickey-fuller testto check stationarity of data\n",
    "from sklearn.metrics import mean_squared_error # For evaluating the model\n",
    "from sklearn.preprocessing import LabelEncoder # To encode categorical integer features\n",
    "import matplotlib.pyplot as plt # For plotting\n",
    "import seaborn as sns # Statistical data visualization\n",
    "import scipy.stats as stats # Statistical analysis\n",
    "import pylab # For plotting\n",
    "import warnings # To handle warnings\n",
    "warnings.filterwarnings(\"ignore\") # Ignore all warings\n",
    "from statsmodels.tsa.statespace.sarimax import SARIMAX # To do SARIMAX\n",
    "from sklearn.model_selection import train_test_split # To split into train and test data set\n",
    "from sklearn.preprocessing import StandardScaler # For RNN: Recursive neural network\n",
    "from keras.models import Sequential # For RNN\n",
    "from keras.layers import LSTM, Dense, Dropout # For RNN\n",
    "from keras.optimizers import Adam # For RNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cba96e44",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ignore all warings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "58afcf9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import data\n",
    "file_path = '/Users/deepakvarier/Downloads/hackathon_data'\n",
    "date_format = \"%d/%m/%y\"\n",
    "df = pd.read_csv(file_path+'/train.csv', sep = ',', parse_dates = ['week'], date_parser = lambda x: pd.to_datetime(x, format = date_format))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "601fee48",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>2011-01-17</td>\n",
       "      <td>8091</td>\n",
       "      <td>216418</td>\n",
       "      <td>99.0375</td>\n",
       "      <td>111.8625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>2011-01-17</td>\n",
       "      <td>8091</td>\n",
       "      <td>216419</td>\n",
       "      <td>99.0375</td>\n",
       "      <td>99.0375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2011-01-17</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>19</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>2011-01-17</td>\n",
       "      <td>8091</td>\n",
       "      <td>216233</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>2011-01-17</td>\n",
       "      <td>8091</td>\n",
       "      <td>217390</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>52</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "0          1 2011-01-17      8091  216418      99.0375    111.8625   \n",
       "1          2 2011-01-17      8091  216419      99.0375     99.0375   \n",
       "2          3 2011-01-17      8091  216425     133.9500    133.9500   \n",
       "3          4 2011-01-17      8091  216233     133.9500    133.9500   \n",
       "4          5 2011-01-17      8091  217390     141.0750    141.0750   \n",
       "\n",
       "   is_featured_sku  is_display_sku  units_sold  \n",
       "0                0               0          20  \n",
       "1                0               0          28  \n",
       "2                0               0          19  \n",
       "3                0               0          44  \n",
       "4                0               0          52  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(150150, 9)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 150150 entries, 0 to 150149\n",
      "Data columns (total 9 columns):\n",
      " #   Column           Non-Null Count   Dtype         \n",
      "---  ------           --------------   -----         \n",
      " 0   record_ID        150150 non-null  int64         \n",
      " 1   week             150150 non-null  datetime64[ns]\n",
      " 2   store_id         150150 non-null  int64         \n",
      " 3   sku_id           150150 non-null  int64         \n",
      " 4   total_price      150149 non-null  float64       \n",
      " 5   base_price       150150 non-null  float64       \n",
      " 6   is_featured_sku  150150 non-null  int64         \n",
      " 7   is_display_sku   150150 non-null  int64         \n",
      " 8   units_sold       150150 non-null  int64         \n",
      "dtypes: datetime64[ns](1), float64(2), int64(6)\n",
      "memory usage: 10.3 MB\n"
     ]
    }
   ],
   "source": [
    "# Characteristics of data\n",
    "df.head()\n",
    "df.shape\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3aab5372",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Timestamp('2013-07-09 00:00:00')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['week'].max()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "22df1928",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Timestamp('2011-01-17 00:00:00')"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['week'].min()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6b70cc8a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "record_ID          0\n",
       "week               0\n",
       "store_id           0\n",
       "sku_id             0\n",
       "total_price        1\n",
       "base_price         0\n",
       "is_featured_sku    0\n",
       "is_display_sku     0\n",
       "units_sold         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check null values in the data\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "dd303165",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "record_ID          0\n",
       "week               0\n",
       "store_id           0\n",
       "sku_id             0\n",
       "total_price        0\n",
       "base_price         0\n",
       "is_featured_sku    0\n",
       "is_display_sku     0\n",
       "units_sold         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Since total no. of rows = 150150 and the null value is only in 1 row, therefore, we will remove the null row\n",
    "# Calculate the total number of rows\n",
    "total_rows = len(df)\n",
    "# Calculate the number of rows with missing values\n",
    "na_rows = df.isna().any(axis=1).sum()\n",
    "if na_rows < total_rows * 0.01:\n",
    "    df.dropna(inplace=True)\n",
    "else:\n",
    "    # Fill missing values with the average of store_id and sku_id combination\n",
    "    df.fillna(df.groupby(['store_id', 'sku_id']).transform('mean'), inplace=True)\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e191446c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(150149, 9)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Checking whether there are rows where the total_price or units_sold <=0\n",
    "df.shape\n",
    "df['total_price'].loc[df['total_price']<=0].count()\n",
    "df['units_sold'].loc[df['units_sold']<=0].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ecc4acda",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(150149, 9)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Delete rows with negative rows\n",
    "con1 = df['units_sold']<=0\n",
    "con2 = df['total_price']<=0\n",
    "df = df[~(con1 & con2)]\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2054cbe3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(150149, 9)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(150149, 9)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Dropping duplicates if any\n",
    "df.shape\n",
    "df = df.drop_duplicates(['week', 'store_id', 'sku_id'])\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "07a067c9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>150149</th>\n",
       "      <td>212644</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9984</td>\n",
       "      <td>679023</td>\n",
       "      <td>234.4125</td>\n",
       "      <td>234.4125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149427</th>\n",
       "      <td>211610</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>378934</td>\n",
       "      <td>213.0375</td>\n",
       "      <td>213.0375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>16</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149375</th>\n",
       "      <td>211530</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216418</td>\n",
       "      <td>110.4375</td>\n",
       "      <td>110.4375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>162</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149376</th>\n",
       "      <td>211531</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216419</td>\n",
       "      <td>109.7250</td>\n",
       "      <td>109.7250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>137</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149377</th>\n",
       "      <td>211532</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>300021</td>\n",
       "      <td>109.0125</td>\n",
       "      <td>109.0125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>108</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "150149     212644 2013-07-09      9984  679023     234.4125    234.4125   \n",
       "149427     211610 2013-07-09      9164  378934     213.0375    213.0375   \n",
       "149375     211530 2013-07-09      9112  216418     110.4375    110.4375   \n",
       "149376     211531 2013-07-09      9112  216419     109.7250    109.7250   \n",
       "149377     211532 2013-07-09      9112  300021     109.0125    109.0125   \n",
       "\n",
       "        is_featured_sku  is_display_sku  units_sold  \n",
       "150149                0               0          15  \n",
       "149427                0               0          16  \n",
       "149375                0               0         162  \n",
       "149376                0               0         137  \n",
       "149377                0               0         108  "
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Sort dataframe by date column in chronological order\n",
    "df = df.sort_values(by='week', ascending=False)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2b75708f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to create data frame for the selected store_id and sku_id\n",
    "def create_dataframe(sku_id, df):\n",
    "    # Filter the data for the specified store_id and sku_id\n",
    "    filtered_data = df[(df['sku_id'] == sku_id)]\n",
    "\n",
    "    # If no data is found for the specified sku_id, return None\n",
    "    if filtered_data.empty:\n",
    "        print(\"No data found for the specified sku_id.\")\n",
    "        return None\n",
    "\n",
    "    return filtered_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "093f17cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get user input for sku_id\n",
    "#sku_id = int(input(\"Enter sku_id: \"))\n",
    "sku_id=216425"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "c61a03ba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>61</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "149378     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "149358     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "149420     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "149394     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "149406     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "        is_featured_sku  is_display_sku  units_sold  \n",
       "149378                0               0          72  \n",
       "149358                0               0          18  \n",
       "149420                0               0          44  \n",
       "149394                0               0          13  \n",
       "149406                0               0          61  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(8580, 9)"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Call the function with user inputs to create dataframe of selected store_id and sku_id\n",
    "df_selected = create_dataframe(sku_id,df)\n",
    "if df_selected is not None:\n",
    "    df_selected.head()\n",
    "    df_selected.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "57a9dd9b",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df_selected = df_selected.drop(columns=['record_ID', 'store_id'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "f394e619",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>72</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>61</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "149378     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "149358     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "149420     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "149394     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "149406     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "        is_featured_sku  is_display_sku  units_sold  \n",
       "149378                0               0          72  \n",
       "149358                0               0          18  \n",
       "149420                0               0          44  \n",
       "149394                0               0          13  \n",
       "149406                0               0          61  "
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_selected.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "9625818b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Group by sku_id and week and perform aggregation\n",
    "#df_selected = df.groupby(['sku_id','week']).agg({\n",
    "#    'total_price': 'mean',\n",
    "#    'base_price': 'mean',\n",
    "#    'is_featured_sku': 'max',\n",
    "#    'is_display_sku': 'max',\n",
    "#    'units_sold': 'sum'\n",
    "#}).reset_index()\n",
    "\n",
    "# Print the aggregated DataFrame\n",
    "#print(df_selected)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "554a2094",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pre-processing the data\n",
    "def preprocess_data(df):\n",
    "    # Convert 'week' column to datetime type and extract seasonality features\n",
    "    df['week'] = pd.to_datetime(df['week'])\n",
    "    df['month'] = df['week'].dt.month\n",
    "    df['year'] = df['week'].dt.year\n",
    "    df['day_of_week'] = df['week'].dt.dayofweek\n",
    "    df['day_of_month'] = df['week'].dt.day\n",
    "    df['discount'] = df['base_price'] - df['total_price']\n",
    "    # Encode categorical variables 'is_featured_sku' and 'is_display_sku'\n",
    "    label_encoder = LabelEncoder()\n",
    "    df['is_featured_sku'] = label_encoder.fit_transform(df['is_featured_sku'])\n",
    "    df['is_display_sku'] = label_encoder.fit_transform(df['is_display_sku'])\n",
    "    \n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "468c0d0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Call the function to pre-process the data\n",
    "df_processed = preprocess_data(df_selected)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "cf8f1c30",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>72</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>61</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "149378     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "149358     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "149420     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "149394     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "149406     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "        is_featured_sku  is_display_sku  units_sold  month  year  day_of_week  \\\n",
       "149378                0               0          72      7  2013            1   \n",
       "149358                0               0          18      7  2013            1   \n",
       "149420                0               0          44      7  2013            1   \n",
       "149394                0               0          13      7  2013            1   \n",
       "149406                0               0          61      7  2013            1   \n",
       "\n",
       "        day_of_month  discount  \n",
       "149378             9       0.0  \n",
       "149358             9       0.0  \n",
       "149420             9       0.0  \n",
       "149394             9       0.0  \n",
       "149406             9       0.0  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4d5eac87",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df_processed.drop(['week'], inplace=True, axis = 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "125fefb4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>72</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>44</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>13</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>61</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "149378     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "149358     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "149420     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "149394     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "149406     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "        is_featured_sku  is_display_sku  units_sold  month  year  day_of_week  \\\n",
       "149378                0               0          72      7  2013            1   \n",
       "149358                0               0          18      7  2013            1   \n",
       "149420                0               0          44      7  2013            1   \n",
       "149394                0               0          13      7  2013            1   \n",
       "149406                0               0          61      7  2013            1   \n",
       "\n",
       "        day_of_month  discount  \n",
       "149378             9       0.0  \n",
       "149358             9       0.0  \n",
       "149420             9       0.0  \n",
       "149394             9       0.0  \n",
       "149406             9       0.0  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "c69940ac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ADF Statistic: -7.585355344646755\n",
      "p-value: 2.6155227314869616e-11\n"
     ]
    }
   ],
   "source": [
    "# Check if the data is stationary\n",
    "result = adfuller(df_processed['units_sold'].dropna())\n",
    "# Print the test statistic and p-value\n",
    "print('ADF Statistic:', result[0])\n",
    "print('p-value:', result[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "842bcd25",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Since the p-value is below 0.05,\n",
    "# the data can be assumed to be stationary hence we can proceed with the data without any transformation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7e12d333",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8580, 14)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "eba71f3d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1.4209199486838835"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed['units_sold'].skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "f88db18d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# units sold is highly positively skewed since skewness > 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "3b821fdc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_processed.units_sold.hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "722c336b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='units_sold', ylabel='Density'>"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAkkAAAGzCAYAAAA7YYPWAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABVW0lEQVR4nO3de1xUdf4/8NeZGwPIDDflooBgWiqICqlgZheX1K00tcx21WrrF7vftpTcTWsrs/1mF3PdyktrWrnbqrtprfvNSryRJt5BzdC8oFwEuTNcZIaZOb8/YAaQUWEcOHN5PR+PeSSHM3Pe00S8/Hze5/MRRFEUQURERERtyKQugIiIiMgZMSQRERER2cCQRERERGQDQxIRERGRDQxJRERERDYwJBERERHZwJBEREREZANDEhEREZENDElERERENjAkEREREdmgkLqAFStW4N1330VRUREGDx6MZcuWYcyYMdc8PyMjA2lpaTh58iTCw8Pxxz/+Eampqdbvr169GuvWrcOPP/4IAEhISMCbb76JESNGWM9ZuHAhXn/99TavGxISguLi4g7XbTabcenSJfj5+UEQhA4/j4iIiKQjiiJqamoQHh4OmewGY0WihDZs2CAqlUpx9erV4k8//SQ+//zzoq+vr3jx4kWb558/f1708fERn3/+efGnn34SV69eLSqVSvGLL76wnvPYY4+Jy5cvF7OyssScnBzxiSeeELVarVhQUGA957XXXhMHDx4sFhUVWR8lJSWdqj0/P18EwAcffPDBBx98uOAjPz//hr/rBVGUboPbkSNHYvjw4Vi5cqX12MCBAzF58mQsXry43fkvvvgitmzZgpycHOux1NRUHDt2DJmZmTavYTKZEBAQgA8//BCzZs0C0DSS9NVXXyE7O9vu2qurq+Hv74/8/HxoNBq7X4eIiIi6j06nQ0REBKqqqqDVaq97rmTTbQaDAUeOHMH8+fPbHE9JScG+fftsPiczMxMpKSltjt13331Ys2YNGhsboVQq2z2nvr4ejY2NCAwMbHP8zJkzCA8Ph5eXF0aOHIk333wTMTEx16xXr9dDr9dbv66pqQEAaDQahiQiIiIX05FWGckat8vKymAymRASEtLm+PV6g4qLi22ebzQaUVZWZvM58+fPR+/evTFu3DjrsZEjR2LdunX47rvvsHr1ahQXFyM5ORnl5eXXrHfx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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.kdeplot(df_processed.units_sold)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "f2e11038",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 1600x500 with 0 Axes>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='units_sold', ylabel='Density'>"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1600x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(16,5))\n",
    "plt.subplot(1,2,1)\n",
    "sns.distplot(df_processed['units_sold'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "8fbfd9a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((array([-3.77258292, -3.54529463, -3.42041149, ...,  3.42041149,\n",
       "          3.54529463,  3.77258292]),\n",
       "  array([  1,   1,   1, ..., 177, 181, 190])),\n",
       " (23.691646652457035, 34.01923076923076, 0.9359049160815701))"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAj4AAAHFCAYAAADyj/PrAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAABfkUlEQVR4nO3dd3hTZf/H8XdaaKGllNHSUpkWBJGhgAMU2SCKgKCiIILgQEBBQGTKUKYDVFScIIqCyhDFAfowZclSGbIsslo2LaO0pT2/P+4fgdICSZs0TfN5XVcue+6cnHwTeOjnuc89bJZlWYiIiIj4AD9PFyAiIiKSUxR8RERExGco+IiIiIjPUPARERERn6HgIyIiIj5DwUdERER8hoKPiIiI+AwFHxEREfEZCj4iIiLiMxR8RHzUtGnTsNls9ke+fPkoVaoUTzzxBAcOHHDpe9lsNnr16uWy6+3Zswebzcbrr79+zXMvfM49e/bY27p06UK5cuXSnVeuXDm6dOliPz548CAjRoxg06ZNrin6snqu9b0vWbIEm83GkiVLnH6PlStXMmLECE6ePOm6wkXyiHyeLkBEPGvq1KlUrlyZxMREli1bxtixY1m6dCl///03wcHBni4v2+677z5WrVpFyZIlr3re3LlzKVy4sP344MGDjBw5knLlynHzzTe7vC53fu8rV65k5MiRdOnShSJFirimYJE8QsFHxMdVrVqV2rVrA9CwYUNSU1N55ZVXmDdvHh07dsz0NWfPniUoKCgny8yy8PBwwsPDr3neLbfckgPVXJSV711Esk+3ukQknTvuuAOA//77DzC3hQoVKsTff/9Ns2bNCAkJoXHjxgAcP36cHj16cN111xEQEMD111/PkCFDSEpKyvTaH3zwATfccAOBgYFUqVKFmTNnpnv+yJEj9OjRgypVqlCoUCFKlChBo0aNWL58eabXS0tLY/To0ZQpU4YCBQpQu3Ztfvvtt3TnZHarKzOX3upasmQJt956KwBPPPGE/bbUiBEj+Pzzz7HZbKxatSrDNUaNGkX+/Pk5ePDgVd8rM5d/71cyf/586tSpQ1BQECEhITRt2jRdLSNGjODFF18EoHz58vbas3LLTCQvUvARkXR27doFkK6XJDk5mVatWtGoUSO+++47Ro4cyblz52jYsCHTp0+nb9++LFiwgMcee4wJEybQtm3bDNedP38+b7/9NqNGjeLbb7+lbNmyPProo3z77bf2c44fPw7A8OHDWbBgAVOnTuX666+nQYMGmf7injx5Mj///DOTJk3iiy++wM/PjxYtWmQaSpxRs2ZNpk6dCsDQoUNZtWoVq1at4sknn6R9+/ZERkby7rvvpnvN+fPn+eCDD3jggQeIiopy+j0z+94v9+WXX9K6dWsKFy7MV199xSeffMKJEydo0KABK1asAODJJ5/kueeeA2DOnDn22mvWrOl0TSJ5kiUiPmnq1KkWYK1evdpKSUmxTp06Zf3www9WeHi4FRISYsXFxVmWZVmdO3e2AOvTTz9N9/opU6ZYgPX111+nax8/frwFWAsXLrS3AVbBggXt17Qsyzp//rxVuXJlq0KFCles8fz581ZKSorVuHFj64EHHrC3x8TEWIAVFRVlJSYm2tsTEhKsYsWKWU2aNMnwOWNiYuxtnTt3tsqWLZvuvcqWLWt17tzZfvzHH39YgDV16tQMdQ0fPtwKCAiwDh06ZG+bNWuWBVhLly694ue5tJ5rfe+LFy+2AGvx4sWWZVlWamqqFRUVZVWrVs1KTU21X+/UqVNWiRIlrLp169rbXnvttQyfWUQM9fiI+Lg77riD/PnzExISQsuWLYmMjOSnn34iIiIi3Xnt2rVLd/y///2P4OBgHnzwwXTtF24XXX7LqXHjxumu6e/vT/v27dm1axf79++3t0+ZMoWaNWtSoEAB8uXLR/78+fntt9/Ytm1bhtrbtm1LgQIF7MchISHcf//9LFu2jNTUVOe+CCc8++yzAHz00Uf2tsmTJ1OtWjXuvvtuh67h6Pd+wfbt2zl48CCdOnXCz+/iP92FChWiXbt2rF69mrNnz2bjU4n4Bg1uFvFx06dP58YbbyRfvnxERERkOvspKCgo3YwngGPHjhEZGYnNZkvXXqJECfLly8exY8fStUdGRma47oW2Y8eOUapUKd5880369etH9+7deeWVVwgLC8Pf359hw4ZlGnyudM3k5GROnz5NaGjotb+ALIiIiKB9+/Z88MEHDBw4kC1btrB8+XI++OADh6/hyPd+qQvfZ2bnRUVFkZaWxokTJ7xm0LmIpyj4iPi4G2+80T676EouDzcAxYsXZ82aNViWle75w4cPc/78ecLCwtKdHxcXl+EaF9qKFy8OwBdffEGDBg14//3305136tSpTOu60jUDAgIoVKjQVT9TdvXu3ZvPP/+c7777jp9//pkiRYo4NRvLke/9Uhe+o9jY2AzPHTx4ED8/P4oWLerw9UR8lW51iUiWNG7cmNOnTzNv3rx07dOnT7c/f6nffvuNQ4cO2Y9TU1OZNWsW0dHRlCpVCjABKzAwMN3r/vrrrysOVp4zZw7nzp2zH586dYrvv/+eevXq4e/vn+XPBtjrSExMzPT5WrVqUbduXcaPH8+MGTPo0qWLW9c9qlSpEtdddx1ffvkllmXZ28+cOcPs2bPtM70cqV3El6nHR0Sy5PHHH+fdd9+lc+fO7Nmzh2rVqrFixQrGjBnDvffeS5MmTdKdHxYWRqNGjRg2bBjBwcG89957/PPPP+mmtLds2ZJXXnmF4cOHU79+fbZv386oUaMoX74858+fz1CDv78/TZs2pW/fvqSlpTF+/HgSEhIYOXJktj9fdHQ0BQsWZMaMGdx4440UKlSIqKiodDO2evfuTfv27bHZbPTo0SPb73k1fn5+TJgwgY4dO9KyZUueeeYZkpKSeO211zh58iTjxo2zn1utWjUA3nrrLTp37kz+/PmpVKkSISEhbq1RxBso+IhIlhQoUIDFixczZMgQXnvtNY4cOcJ1111H//79GT58eIbzW7VqxU033cTQoUPZu3cv0dHRzJgxg/bt29vPGTJkCGfPnuWTTz5hwoQJVKlShSlTpjB37txMp7P36tWLc+fO8fzzz3P48GFuuukmFixYwJ133pntzxcUFMSnn37KyJEjadasGSkpKQwfPpwRI0bYz2nTpg2BgYE0bNiQihUrZvs9r6VDhw4EBwczduxY2rdvj7+/P3fccQeLFy+mbt269vMaNGjAoEGD+Oyzz/joo49IS0tj8eLFNGjQwO01iuR2NuvSPlMREXHY999/T6tWrViwYAH33nuvp8sREQco+IiIOGnr1q38999/9O7dm+DgYDZs2JDpAHARyX00uFlExEk9evSgVatWFC1alK+++kqhR8SLqMdHREREfIZ6fERERMRnKPiIiIiIz1DwEREREZ+RK9bxGTt2LHPmzOGff/6hYMGC9tVQK1WqZD/HsixGjhzJhx9+yIkTJ7j99tt59913uemmm+znJCUl0b9/f7766isSExNp3Lgx7733nn1VWEekpaVx8OBBQkJCNGBRRETES1iWxalTp4iKikq3kW9mJ3pc8+bNralTp1qbN2+2Nm3aZN13331WmTJlrNOnT9vPGTdunBUSEmLNnj3b+vvvv6327dtbJUuWtBISEuzndO/e3bruuuusRYsWWRs2bLAaNmxo1ahRwzp//rzDtezbt88C9NBDDz300EMPL3zs27fvqr/nc+WsriNHjlCiRAmWLl3K3XffjWVZREVF0adPH1566SXA9O5EREQwfvx4nnnmGeLj4wkPD+fzzz+3rwR78OBBSpcuzY8//kjz5s0deu/4+HiKFCnCvn37MuxGLSIiIrlTQkICpUuX5uTJk4SGhl7xvFxxq+ty8fHxABQrVgyAmJgY4uLiaNasmf2cwMBA6tevz8qVK3nmmWdYv349KSkp6c6JioqiatWqrFy58orBJykpiaSkJPvxhV2gCxcurOAjIiLiZa41TCXXDW62LIu+ffty1113UbVqVQDi4uIAiIiISHduRESE/bm4uDgCAgIoWrToFc/JzNixYwkNDbU/Spcu7cqPIyIiIrlIrgs+vXr14q+//uKrr77K8NzlKc6yrGsmu2udM2jQIOLj4+2Pffv2Za1wERERyfVyVfB57rnnmD9/PosXL043EysyMhIgQ8/N4cOH7b1AkZGRJCcnc+LEiSuek5nAwED7bS3d3hIREcnbckXwsSyLXr16MWfOHP73v/9Rvnz5dM+XL1+eyMhIFi1aZG9LTk5m6dKl1K1bF4BatWqRP3/+dOfExsayefNm+zkiIiLi23LF4OaePXvy5Zdf8t133xESEmLv2QkNDaVgwYLYbDb69OnDmDFjqFixIhUrVmTMmDEEBQXRoUMH+7ndunWjX79+FC9enGLFitG/f3+qVatGkyZNPPnxREREJJfIFcHn/fffB6BBgwbp2qdOnUqXLl0AGDBgAImJifTo0cO+gOHChQsJCQmxnz9x4kTy5cvHww8/bF/AcNq0afj7++fURxEREZFcLFeu4+NJCQkJhIaGEh8fr/E+IiIiXsLR39+5YoyPiIiISE5Q8BERERGfkSvG+IiIiEjelpoKy5dDbCyULAn16oEnhuAq+IiIiIhbzZkDvXvD/v0X20qVgrfegrZtc7YW3eoSERERt5kzBx58MH3oAThwwLTPmZOz9Sj4iIiIiFukppqenszmj19o69PHnJdTFHxERETELZYvz9jTcynLgn37zHk5RcFHRERE3CI21rXnuYKCj4iI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JIcp+ORV//1n1t754guzGGG+fPDsszB0KJQo4enqREREPELBJ685fhzGjIF33jGDmAHatzcrLkdHe7Y2ERERD1PwySsSE03YGTsWTp40bQ0bwvjxcOutHi1NREQkt1Dw8XapqfD55+a21v79pq1aNRN47rlHM7VEREQuoeDjrSwLfvrJbDHx99+mrXRpszbPY4+Z1ZdFREQkHa/ZYrtcuXLYbLZ0j4EDB6Y7Z+/evdx///0EBwcTFhbG888/T/KFcS55yR9/QKNGcN99JvQUKWIWH9y+HTp3VugRERG5Aq/q8Rk1ahRPPfWU/bhQoUL2n1NTU7nvvvsIDw9nxYoVHDt2jM6dO2NZFu+8844nynW9XbvM4oNff22OAwPhuedg0CAoVsyztYmIiHgBrwo+ISEhREZGZvrcwoUL2bp1K/v27SPq/zfWfOONN+jSpQujR4+msDfvLn74sLmFNWUKnD9vxu106gSjRkHZsp6uTkRExGt4za0ugPHjx1O8eHFuvvlmRo8ene421qpVq6hatao99AA0b96cpKQk1l9YrdjbnDljAk90NEyebELPPffAxo3w2WcKPSIiIk7ymh6f3r17U7NmTYoWLcratWsZNGgQMTExfPzxxwDExcURERGR7jVFixYlICCAuLi4K143KSmJpKQk+3FCQoJ7PoAzUlLg009hxAi4UHutWmYcT6NGHi1NRETEm3m0x2fEiBEZBixf/li3bh0AL7zwAvXr16d69eo8+eSTTJkyhU8++YRjF3YXB2yZTN22LCvT9gvGjh1LaGio/VG6dGnXf1BHWRbMnQtVq0L37ib0lC8PX30Fa9cq9IiIiGSTR3t8evXqxSOPPHLVc8qVK5dp+x133AHArl27KF68OJGRkaxZsybdOSdOnCAlJSVDT9ClBg0aRN++fe3HCQkJngk/v/8OAwbAypXmOCzMrM3TvTsEBOR8PSIiInmQR4NPWFgYYWFhWXrtxo0bAShZsiQAderUYfTo0cTGxtrbFi5cSGBgILVq1bridQIDAwkMDMxSDS6xbZuZlfXdd+a4YEHo29eEIG8ekC0iIpILecUYn1WrVrF69WoaNmxIaGgof/zxBy+88AKtWrWiTJkyADRr1owqVarQqVMnXnvtNY4fP07//v156qmncueMroMHzRieTz6BtDTw84Nu3UzbJQO0RURExHW8IvgEBgYya9YsRo4cSVJSEmXLluWpp55iwIAB9nP8/f1ZsGABPXr04M4776RgwYJ06NCB119/3YOVZyIhwQxSfvNNs78WQOvWZo+tG2/0bG0iIiJ5nM2yLMvTReQmCQkJhIaGEh8f79qeouRksw7PK6/A0aOmrU4dE4Luust17yMiIuKDHP397RU9Pl4vPh5q1oR//zXHlSqZHp42bbSJqIiISA5S8MkJoaFw881w9qwZw9OtG+TTVy8iIpLT9Ns3p7z7LoSEQHCwpysRERHxWQo+OeUKe4yJiIhIzvGqvbpEREREskPBR0RERHyGgo+IiIj4DAUfERER8RkKPiIiIuIzFHxERETEZyj4iIiIiM9Q8BERERGfoeAjIiIiPkPBR0RERHyGgo+IiIj4DAUfERER8RkKPiIiIuIztDv7ZSzLAiAhIcHDlYiIiIijLvzevvB7/EoUfC5z6tQpAEqXLu3hSkRERMRZp06dIjQ09IrP26xrRSMfk5aWxsGDBwkJCcFms2V6TkJCAqVLl2bfvn0ULlw4hyvMWb70WUGfN6/zpc/rS58V9HnzOkc+r2VZnDp1iqioKPz8rjySRz0+l/Hz86NUqVIOnVu4cGGf+AsHvvVZQZ83r/Olz+tLnxX0efO6a33eq/X0XKDBzSIiIuIzFHxERETEZyj4ZEFgYCDDhw8nMDDQ06W4nS99VtDnzet86fP60mcFfd68zpWfV4ObRURExGeox0dERER8hoKPiIiI+AwFHxEREfEZCj4iIiLiMxR8XCQpKYmbb74Zm83Gpk2bPF2O27Rq1YoyZcpQoEABSpYsSadOnTh48KCny3K5PXv20K1bN8qXL0/BggWJjo5m+PDhJCcne7o0txk9ejR169YlKCiIIkWKeLocl3vvvfcoX748BQoUoFatWixfvtzTJbnNsmXLuP/++4mKisJmszFv3jxPl+Q2Y8eO5dZbbyUkJIQSJUrQpk0btm/f7umy3Ob999+nevXq9oX86tSpw08//eTpsnLE2LFjsdls9OnTJ1vXUfBxkQEDBhAVFeXpMtyuYcOGfP3112zfvp3Zs2eze/duHnzwQU+X5XL//PMPaWlpfPDBB2zZsoWJEycyZcoUBg8e7OnS3CY5OZmHHnqIZ5991tOluNysWbPo06cPQ4YMYePGjdSrV48WLVqwd+9eT5fmFmfOnKFGjRpMnjzZ06W43dKlS+nZsyerV69m0aJFnD9/nmbNmnHmzBlPl+YWpUqVYty4caxbt45169bRqFEjWrduzZYtWzxdmlv98ccffPjhh1SvXj37F7Mk23788UercuXK1pYtWyzA2rhxo6dLyjHfffedZbPZrOTkZE+X4nYTJkywypcv7+ky3G7q1KlWaGiop8twqdtuu83q3r17urbKlStbAwcO9FBFOQew5s6d6+kycszhw4ctwFq6dKmnS8kxRYsWtT7++GNPl+E2p06dsipWrGgtWrTIql+/vtW7d+9sXU89Ptl06NAhnnrqKT7//HOCgoI8XU6OOn78ODNmzKBu3brkz5/f0+W4XXx8PMWKFfN0GeKk5ORk1q9fT7NmzdK1N2vWjJUrV3qoKnGX+Ph4AJ/432pqaiozZ87kzJkz1KlTx9PluE3Pnj257777aNKkiUuup+CTDZZl0aVLF7p3707t2rU9XU6OeemllwgODqZ48eLs3buX7777ztMlud3u3bt555136N69u6dLEScdPXqU1NRUIiIi0rVHREQQFxfnoarEHSzLom/fvtx1111UrVrV0+W4zd9//02hQoUIDAyke/fuzJ07lypVqni6LLeYOXMmGzZsYOzYsS67poJPJkaMGIHNZrvqY926dbzzzjskJCQwaNAgT5ecLY5+3gtefPFFNm7cyMKFC/H39+fxxx/H8pIFwJ39rAAHDx7knnvu4aGHHuLJJ5/0UOVZk5XPm1fZbLZ0x5ZlZWgT79arVy/++usvvvrqK0+X4laVKlVi06ZNrF69mmeffZbOnTuzdetWT5flcvv27aN379588cUXFChQwGXX1ZYVmTh69ChHjx696jnlypXjkUce4fvvv0/3j2dqair+/v507NiRzz77zN2luoSjnzezv3j79++ndOnSrFy50iu6Wp39rAcPHqRhw4bcfvvtTJs2DT8/7/r/Cln5s502bRp9+vTh5MmTbq4uZyQnJxMUFMQ333zDAw88YG/v3bs3mzZtYunSpR6szv1sNhtz586lTZs2ni7FrZ577jnmzZvHsmXLKF++vKfLyVFNmjQhOjqaDz74wNOluNS8efN44IEH8Pf3t7elpqZis9nw8/MjKSkp3XOOyufKIvOKsLAwwsLCrnne22+/zauvvmo/PnjwIM2bN2fWrFncfvvt7izRpRz9vJm5kJuTkpJcWZLbOPNZDxw4QMOGDalVqxZTp071utAD2fuzzSsCAgKoVasWixYtShd8Fi1aROvWrT1YmbiCZVk899xzzJ07lyVLlvhc6AHzHXjLv8HOaNy4MX///Xe6tieeeILKlSvz0ksvZSn0gIJPtpQpUybdcaFChQCIjo6mVKlSnijJrdauXcvatWu56667KFq0KP/++y8vv/wy0dHRXtHb44yDBw/SoEEDypQpw+uvv86RI0fsz0VGRnqwMvfZu3cvx48fZ+/evaSmptrXo6pQoYL977a36tu3L506daJ27drUqVOHDz/8kL179+bZMVunT59m165d9uOYmBg2bdpEsWLFMvy75e169uzJl19+yXfffUdISIh93FZoaCgFCxb0cHWuN3jwYFq0aEHp0qU5deoUM2fOZMmSJfz888+eLs3lQkJCMozVujC+NFtjuLI1J0zSiYmJydPT2f/66y+rYcOGVrFixazAwECrXLlyVvfu3a39+/d7ujSXmzp1qgVk+sirOnfunOnnXbx4sadLc4l3333XKlu2rBUQEGDVrFkzT093Xrx4caZ/lp07d/Z0aS53pf+dTp061dOluUXXrl3tf4/Dw8Otxo0bWwsXLvR0WTnGFdPZNcZHREREfIb3DVoQERERySIFHxEREfEZCj4iIiLiMxR8RERExGco+IiIiIjPUPARERERn6HgIyIiIj5DwUfEx+zZswebzWZfmdlblCtXjkmTJrnseg0aNKBPnz4uu54n2Gw25s2bB3jvn6tITlPwEclDrrUTe5cuXTxd4jVNmzaNIkWKZGj/448/ePrpp3O+oFxgxIgR3HzzzRnaY2NjadGiRc4XJOLFtFeXSB4SGxtr/3nWrFm8/PLLbN++3d5WsGBBTpw44YnS0u2qnBXh4eEursj75dV940TcST0+InlIZGSk/REaGorNZsvQdsG///5Lw4YNCQoKokaNGqxatSrdtVauXMndd99NwYIFKV26NM8//zxnzpyxP3/ixAkef/xxihYtSlBQEC1atGDnzp325y/03Pzwww9UqVKFwMBA/vvvP5KTkxkwYADXXXcdwcHB3H777SxZsgSAJUuW8MQTTxAfH2/vpRoxYgSQ8VbXyZMnefrpp4mIiKBAgQJUrVqVH374AYBjx47x6KOPUqpUKYKCgqhWrRpfffWV09/nuHHjiIiIICQkhG7dujFw4MB0PS+Z3S5r06ZNup61L774gtq1axMSEkJkZCQdOnTg8OHD9ueXLFmCzWbjt99+o3bt2gQFBVG3bl17YJ02bRojR47kzz//tH8n06ZNA9Lf6srM1q1buffeeylUqBARERF06tSJo0eP2p//9ttvqVatGgULFqR48eI0adIk3Z+xSF6k4CPio4YMGUL//v3ZtGkTN9xwA48++ijnz58H4O+//6Z58+a0bduWv/76i1mzZrFixQp69eplf32XLl1Yt24d8+fPZ9WqVViWxb333ktKSor9nLNnzzJ27Fg+/vhjtmzZQokSJXjiiSf4/fffmTlzJn/99RcPPfQQ99xzDzt37qRu3bpMmjSJwoULExsbS2xsLP37989Qe1paGi1atGDlypV88cUXbN26lXHjxuHv7w/AuXPnqFWrFj/88AObN2/m6aefplOnTqxZs8bh7+frr79m+PDhjB49mnXr1lGyZEnee+89p7/n5ORkXnnlFf7880/mzZtHTExMprcchwwZwhtvvMG6devIly8fXbt2BaB9+/b069ePm266yf6dtG/f/prvGxsbS/369bn55ptZt24dP//8M4cOHeLhhx+2P//oo4/StWtXtm3bxpIlS2jbti3avlHyvOzvlSoiudHUqVOt0NDQDO0xMTEWYH388cf2ti1btliAtW3bNsuyLKtTp07W008/ne51y5cvt/z8/KzExERrx44dFmD9/vvv9uePHj1qFSxY0Pr666/t7w9YmzZtsp+za9cuy2azWQcOHEh37caNG1uDBg26at1ly5a1Jk6caFmWZf3yyy+Wn5+ftX37doe/j3vvvdfq16+f/fhauzzXqVPH6t69e7q222+/3apRo8ZVr9G6deur7oK+du1aC7BOnTplWdbFndR//fVX+zkLFiywACsxMdGyLMsaPnx4uve9ALDmzp1rWdbFP9eNGzdalmVZw4YNs5o1a5bu/H379lmAtX37dmv9+vUWYO3Zs+eKtYrkRerxEfFR1atXt/9csmRJAPstmPXr1zNt2jQKFSpkfzRv3py0tDRiYmLYtm0b+fLl4/bbb7dfo3jx4lSqVIlt27bZ2wICAtK9z4YNG7AsixtuuCHdtZcuXcru3bsdrn3Tpk2UKlWKG264IdPnU1NTGT16NNWrV6d48eIUKlSIhQsXsnfvXoffY9u2bdSpUydd2+XHjti4cSOtW7embNmyhISE0KBBA4AMtVztzyMr1q9fz+LFi9N9z5UrVwZg9+7d1KhRg8aNG1OtWjUeeughPvroI4+N/xLJSRrcLOKj8ufPb//ZZrMB5hbShf8+88wzPP/88xleV6ZMGXbs2JHpNS3Lsl8LzGDqS4/T0tLw9/dn/fr19ttSFxQqVMjh2gsWLHjV59944w0mTpzIpEmTqFatGsHBwfTp04fk5GSH38MRfn5+GW4NXXqr78yZMzRr1oxmzZrxxRdfEB4ezt69e2nevHmGWq7255EVaWlp3H///YwfPz7DcyVLlsTf359FixaxcuVKFi5cyDvvvMOQIUNYs2YN5cuXz/L7iuR2Cj4ikkHNmjXZsmULFSpUyPT5KlWqcP78edasWUPdunUBM6B4x44d3HjjjVe87i233EJqaiqHDx+mXr16mZ4TEBBAamrqVeurXr06+/fvZ8eOHZn2+ixfvpzWrVvz2GOPASYE7Ny586q1Xe7GG29k9erVPP744/a21atXpzsnPDw83Uy61NRUNm/eTMOGDQH4559/OHr0KOPGjaN06dIArFu3zuEaLnDkO7lczZo1mT17NuXKlSNfvsz/qbfZbNx5553ceeedvPzyy5QtW5a5c+fSt29fp2sU8Ra61SUiGbz00kusWrWKnj17smnTJnbu3Mn8+fN57rnnAKhYsSKtW7fmqaeeYsWKFfz555889thjXHfddbRu3fqK173hhhvo2LEjjz/+OHPmzCEmJoY//viD8ePH8+OPPwJm9tbp06f57bffOHr0KGfPns1wnfr163P33XfTrl07Fi1aRExMDD/99BM///wzABUqVLD3Zmzbto1nnnmGuLg4p76D3r178+mnn/Lpp5+yY8cOhg8fzpYtW9Kd06hRIxYsWMCCBQv4559/6NGjBydPnrQ/X6ZMGQICAnjnnXf4999/mT9/Pq+88opTdVz4TmJiYti0aRNHjx4lKSnpmq/p2bMnx48f59FHH2Xt2rX8+++/LFy4kK5du5KamsqaNWsYM2YM69atY+/evcyZM4cjR444FQ5FvJGCj4hkUL16dZYuXcrOnTupV68et9xyC8OGDbOPPQGYOnUqtWrVomXLltSpUwfLsvjxxx/T3bLJzNSpU3n88cfp168flSpVolWrVqxZs8beI1K3bl26d+9O+/btCQ8PZ8KECZleZ/bs2dx66608+uijVKlShQEDBth7RYYNG0bNmjVp3rw5DRo0IDIykjZt2jj1HbRv356XX36Zl156iVq1avHff//x7LPPpjuna9eudO7cmccff5z69etTvnx5e28PmB6hadOm8c0331ClShXGjRvH66+/7lQdAO3ateOee+6hYcOGhIeHOzQ1Pyoqit9//53U1FSaN29O1apV6d27N6Ghofj5+VG4cGGWLVvGvffeyw033MDQoUN54403tCCi5Hk26/Ib1CIikqkRI0Ywb948bQsh4sXU4yMiIiI+Q8FHREREfIZudYmIiIjPUI+PiIiI+AwFHxEREfEZCj4iIiLiMxR8RERExGco+IiIiIjPUPARERERn6HgIyIiIj5DwUdERER8hoKPiIiI+Iz/A3+OTFW0oJ5QAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Q-Q plot\n",
    "stats.probplot(df_processed.units_sold, plot = pylab)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "cd236f46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4622</th>\n",
       "      <td>6555</td>\n",
       "      <td>2011-02-14</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>23</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>14</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3467</th>\n",
       "      <td>4915</td>\n",
       "      <td>2011-02-07</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>18</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2312</th>\n",
       "      <td>3279</td>\n",
       "      <td>2011-01-31</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>27</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>31</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1157</th>\n",
       "      <td>1643</td>\n",
       "      <td>2011-01-24</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>24</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2011-01-17</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>19</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "4622       6555 2011-02-14      8091  216425     133.2375    133.2375   \n",
       "3467       4915 2011-02-07      8091  216425     133.9500    133.9500   \n",
       "2312       3279 2011-01-31      8091  216425     133.2375    133.2375   \n",
       "1157       1643 2011-01-24      8091  216425     134.6625    134.6625   \n",
       "2             3 2011-01-17      8091  216425     133.9500    133.9500   \n",
       "\n",
       "      is_featured_sku  is_display_sku  units_sold  month  year  day_of_week  \\\n",
       "4622                0               0          23      2  2011            0   \n",
       "3467                0               0          18      2  2011            0   \n",
       "2312                0               0          27      1  2011            0   \n",
       "1157                0               0          17      1  2011            0   \n",
       "2                   0               0          19      1  2011            0   \n",
       "\n",
       "      day_of_month  discount  \n",
       "4622            14       0.0  \n",
       "3467             7       0.0  \n",
       "2312            31       0.0  \n",
       "1157            24       0.0  \n",
       "2               17       0.0  "
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Tail of the data\n",
    "df_processed.loc[df_processed['store_id']==8091].tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "f1e824e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Logarithmic transformation of data\n",
    "df_processed['units_sold'] = np.log(df_processed['units_sold'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "54528f77",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>4622</th>\n",
       "      <td>6555</td>\n",
       "      <td>2011-02-14</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.135494</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>14</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3467</th>\n",
       "      <td>4915</td>\n",
       "      <td>2011-02-07</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.890372</td>\n",
       "      <td>2</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2312</th>\n",
       "      <td>3279</td>\n",
       "      <td>2011-01-31</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.295837</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>31</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1157</th>\n",
       "      <td>1643</td>\n",
       "      <td>2011-01-24</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.833213</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>24</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>2011-01-17</td>\n",
       "      <td>8091</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.944439</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "4622       6555 2011-02-14      8091  216425     133.2375    133.2375   \n",
       "3467       4915 2011-02-07      8091  216425     133.9500    133.9500   \n",
       "2312       3279 2011-01-31      8091  216425     133.2375    133.2375   \n",
       "1157       1643 2011-01-24      8091  216425     134.6625    134.6625   \n",
       "2             3 2011-01-17      8091  216425     133.9500    133.9500   \n",
       "\n",
       "      is_featured_sku  is_display_sku  units_sold  month  year  day_of_week  \\\n",
       "4622                0               0    3.135494      2  2011            0   \n",
       "3467                0               0    2.890372      2  2011            0   \n",
       "2312                0               0    3.295837      1  2011            0   \n",
       "1157                0               0    2.833213      1  2011            0   \n",
       "2                   0               0    2.944439      1  2011            0   \n",
       "\n",
       "      day_of_month  discount  \n",
       "4622            14       0.0  \n",
       "3467             7       0.0  \n",
       "2312            31       0.0  \n",
       "1157            24       0.0  \n",
       "2               17       0.0  "
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Tail of the data\n",
    "df_processed.loc[df_processed['store_id']==8091].tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "bfc22b33",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.39976022280938533"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed['units_sold'].skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "840af018",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 1600x500 with 0 Axes>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='units_sold', ylabel='Density'>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "((array([-3.77258292, -3.54529463, -3.42041149, ...,  3.42041149,\n",
       "          3.54529463,  3.77258292]),\n",
       "  array([0.        , 0.        , 0.        , ..., 5.17614973, 5.19849703,\n",
       "         5.24702407])),\n",
       " (0.775718123342573, 3.2518085888101997, 0.9938156688996634))"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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YdSoUSxZsiTH18ycORNPT8/Mh5+fXwkmFhERkVLPYoFlyyAwEJYssU6rHj0awsKgWzej090Sk8VisRgdoiQlJibi6elJQkICHh4eRscRESkRX+6IyPOagW1ql0ASESkK+jxjLAcHB1q2bMnWrVszz40aNYpdu3axbdu2bF+TkpJCSkpK5nFiYiJ+fn76OxQRERGIioLhw+G776zHgYHWnazbtTM2Vy4K8nlUIx9FRERERArA19eXwMDALOcaN25MRETOv+hxdHTEw8Mjy0NEREQqOIvFWjIGBlqLRzs7mDQJ9u4t1cVjQWnDGRERERGRAujQoQNHjx7Ncu7PP/+kTp06BiUSERGRMuf4cXjqKdiwwXrcqhUsWAAhIcbmKgYa+SgiIiIiUgDPPfcc27dv57XXXuP48eN8+eWXzJ8/nxEjRhgdTUREREq79HTrrtVNmliLR2dn6/G2beWyeASNfBQRERERKZBWrVqxcuVKxo8fz7Rp0/D392f27Nk8+uijRkcTERGR0uzAAQgNhd27rcedO8P8+RAQYGyuYqbyUURERESkgB544AEeeOABo2OIiIhIWZCSAjNmwH//ax356OlpHe34xBPWXa3LOZWPIiIiIiIiIiIixWHrVhg2DMLDrce9e8MHH0CNGobGKkla81FERERERERERKQoJSXBqFFwxx3W4tHHB5YtgxUrKlTxCBr5KCIiIiIiIiIiUnRWr4ann4YzZ6zHQ4ZYp1lXqWJoLKOofBQREREREREREblVly7B2LGwZIn1uG5dmDcPunUzNJbRNO1aRERERERERESksCwW65TqwEBr8WgywejREBZW4YtH0MhHERERERERERGRwomKguHD4bvvrMeBgfDJJ9CunbG5ShGNfBQRERERERERESkIi8VaMgYGWotHOzuYNAn27lXx+C8a+SgiIiIiIiIiIpJfx4/DU0/Bhg3W41atYMECCAkxNlcppZGPIiIiIiIiIiIieUlPt+5a3aSJtXh0drYeb9um4jEXGvkoIiIiIiIiIiKSmwMHIDQUdu+2HnfuDPPnQ0CAsbnKAI18FBERERERERERyU5KinUtxxYtrMWjp6d1rce1a1U85pNGPoqIiIiIiIiIiPzb1q0wbBiEh1uPe/eGDz6AGjUMjVXWaOSjiIiIiIiIiIjIDUlJMGoU3HGHtXj09oZly2DFChWP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k08+yVNPPcXGjRurnBnbarUSGxvL2rVrWbVqFW+88QazZ89m/fr1BAUFlWU5lWEY5daduk1EREREqnbqtd+J66oT134Oh4N77rmH+++/v9xxkZGR7N69u8Jznn6N5uPj4/La4XBgtVrZtGlT2RDpE/z8/Kqd3cfHp8rtL7/8Mq+++irz58+nV69eNGvWjBkzZmCz2ar9HtVxpmvs/Px8Ro4cyciRI/nkk08IDg4mOTmZUaNGlctS1d/HuXA4HFx99dW88MIL5baFhoZWeQ3evn37c35fEake9XwUkUYnODjYZfbD3NxcEhMTz/l8np6e2O32cuvd3d25/PLLefHFF9m6dStJSUl89913ZzyfxWLh4osv5umnn2bLli14enqybNkyAgICCAsL46effnLZf+3atXTr1q3Cc3Xv3p1169a5rDv9tYiIiIhUrn///uzYsYMLLrig3OLp6Un37t0pLS11eYZiVlYWu3fvrvQaDaBfv37Y7XbS09PLnffEzM2VXWeeqnfv3hw8eLDSIuiaNWsYM2YMt9xyC3369KFDhw6VPi+8Mt26dTvjNeXp19h2u53t27eXvf7tt9/IzMzk+eefZ8iQIXTt2vWcejNWp01Od+LvsF27duXaulmzZkDl1+AiUvtUfBSRRmfEiBF8/PHHrFmzhu3btzN58uRyv20+G+3ateP48eP873//IzMzk4KCAr766itef/114uPjOXDgAB999BEOh4MuXbpUea7169fz3HPPERcXR3JyMkuXLiUjI6PswvXhhx/mhRdeYMmSJezatYvHHnuM+Ph4pk+fXuH5pkyZwr59+5g5cya7du3is88+K5sRUERERETO7NFHH+WXX35h2rRpxMfHs2fPHlasWMF9990HQKdOnRgzZgx33XUXP/30E7/++iu33HILbdu2ZcyYMZWet3Pnztx8881MmjSJpUuXkpiYyMaNG3nhhRdYuXIlUPF15umGDh3KpZdeyvXXX09sbCyJiYl88803fPvttwBccMEFZb36EhISuOeee0hLSzurNpg+fTrvv/8+77//Prt372bOnDns2LHDZZ8RI0bw9ddf8/XXX/Pbb78xdepUsrOzy7ZHRkbi6enJG2+8wf79+1mxYgV//etfzyrHiTZJTEwkPj6ezMxMiouLz3jMtGnTOHr0KBMmTGDDhg3s37+fVatWcfvtt2O32894DS4itUvFRxFpdGbNmsWll17KVVddxejRo7n22mvp2LHjOZ9v8ODBTJkyhfHjxxMcHMyLL75I8+bNWbp0KSNGjKBbt268/fbbLF68mB49elR5roCAAH788UdGjx5N586d+ctf/sLLL79MTEwMAPfffz8PPvggDz74IL169eLbb79lxYoVdOrUqcLzRUZG8uWXX/Lvf/+bPn368Pbbb/Pcc8+d82cVERERaWp69+7NDz/8wJ49exgyZAj9+vXjiSeeKHsWIcAHH3xAdHQ0V111FYMGDcIwDFauXFnuUT6n++CDD5g0aRIPPvggXbp04ZprrmH9+vVEREQAFV9nVuTLL7/kwgsvZMKECXTv3p1HHnmkrHfgE088Qf/+/Rk1ahTDhg2jTZs2XHvttWfVBuPHj+fJJ5/k0UcfJTo6mgMHDnDvvfe67HP77bczefJkJk2axNChQ2nfvj3Dhw8v2x4cHMyiRYv45z//Sffu3Xn++ed56aWXzioHwPXXX8+VV17J8OHDCQ4OZvHixWc8JiwsjJ9//hm73c6oUaPo2bMn06dPJzAwEDc3tzNeg4tI7bIYemCYiIiIiIiIiJziqaeeYvny5cTHx5sdRUQaOPV8FBERERERERERkVqh4qOISA1KTk7Gz8+v0iU5OdnsiCIiIiIiIiJ1RsOuRURqUGlpKUlJSZVub9euHe7u7nUXSERERERERMREKj6KiIiIiIiIiIhIrdCwaxEREREREREREakVKj6KiIiIiIiIiIhIrVDxUURERERERERERGqFio8iIiIiIiIiIiJSK1R8FBERERERERERkVqh4qOIiIiIiIiIiIjUChUfRUREREREREREpFb8P8OCdYJ3s9XKAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 1600x500 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(16,5))\n",
    "plt.subplot(1,2,1)\n",
    "sns.distplot(df_processed['units_sold'])\n",
    "plt.subplot(1,2,2)\n",
    "stats.probplot(df_processed['units_sold'], plot = pylab)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "3088f233",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.5926190436046035"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "0.9109981340157955"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Finding the boundary values\n",
    "UL = df_processed['units_sold'].mean() + 3*df_processed['units_sold'].std()\n",
    "LL = df_processed['units_sold'].mean() - 3*df_processed['units_sold'].std()\n",
    "UL\n",
    "LL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "7a6aefa3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8580, 14)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "ce0ac8f0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "42"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed['units_sold'].loc[df_processed['units_sold']<LL].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "677cf9d9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed['units_sold'].loc[df_processed['units_sold']>UL].count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "01e52893",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Removing outliers\n",
    "condition1 = df_processed['units_sold']>UL\n",
    "condition2 = df_processed['units_sold']<LL\n",
    "df_processed = df_processed[~(condition1 & condition2)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "0f418d48",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the number of rows for testing\n",
    "test_size = int(len(df_processed)*0.2)\n",
    "end_point = len(df_processed)\n",
    "x = end_point - test_size"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "b5b2f51d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(8580, 14)"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "1716"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "8580"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "6864"
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_processed.shape\n",
    "test_size\n",
    "end_point\n",
    "x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "3d48f48f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split into train and test\n",
    "df_processed_train = df_processed.iloc[:x - 1]\n",
    "df_processed_test = df_processed.iloc[x:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "82ad4a53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(6863, 14)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(1716, 14)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Check shape of test and train\n",
    "df_processed_train.shape\n",
    "df_processed_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "24c2b50a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.276666</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.890372</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.784190</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.564949</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.110874</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "149378     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "149358     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "149420     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "149394     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "149406     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "        is_featured_sku  is_display_sku  units_sold  month  year  day_of_week  \\\n",
       "149378                0               0    4.276666      7  2013            1   \n",
       "149358                0               0    2.890372      7  2013            1   \n",
       "149420                0               0    3.784190      7  2013            1   \n",
       "149394                0               0    2.564949      7  2013            1   \n",
       "149406                0               0    4.110874      7  2013            1   \n",
       "\n",
       "        day_of_month  discount  \n",
       "149378             9       0.0  \n",
       "149358             9       0.0  \n",
       "149420             9       0.0  \n",
       "149394             9       0.0  \n",
       "149406             9       0.0  "
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>29258</th>\n",
       "      <td>41410</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.007333</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29238</th>\n",
       "      <td>41386</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.197225</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29300</th>\n",
       "      <td>41477</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.891820</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29274</th>\n",
       "      <td>41435</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.044522</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29286</th>\n",
       "      <td>41455</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.737670</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "29258      41410 2011-07-11      9112  216425     132.5250    132.5250   \n",
       "29238      41386 2011-07-11      9092  216425     134.6625    134.6625   \n",
       "29300      41477 2011-07-11      9164  216425     134.6625    134.6625   \n",
       "29274      41435 2011-07-11      9132  216425     131.1000    131.1000   \n",
       "29286      41455 2011-07-11      9147  216425     133.9500    133.9500   \n",
       "\n",
       "       is_featured_sku  is_display_sku  units_sold  month  year  day_of_week  \\\n",
       "29258                0               0    4.007333      7  2011            0   \n",
       "29238                0               0    2.197225      7  2011            0   \n",
       "29300                0               0    3.891820      7  2011            0   \n",
       "29274                0               0    3.044522      7  2011            0   \n",
       "29286                0               0    3.737670      7  2011            0   \n",
       "\n",
       "       day_of_month  discount  \n",
       "29258            11       0.0  \n",
       "29238            11       0.0  \n",
       "29300            11       0.0  \n",
       "29274            11       0.0  \n",
       "29286            11       0.0  "
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Processed data\n",
    "df_processed_train.head()\n",
    "df_processed_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "0b80f28a",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = df_processed_test.loc[:, df_processed_test.columns != 'units_sold']\n",
    "y_test = df_processed_test[['units_sold']]\n",
    "X_train = df_processed_train.loc[:, df_processed_train.columns != 'units_sold']\n",
    "y_train = df_processed_train[['units_sold']]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "d0ecc69f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>29258</th>\n",
       "      <td>41410</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29238</th>\n",
       "      <td>41386</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29300</th>\n",
       "      <td>41477</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29274</th>\n",
       "      <td>41435</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29286</th>\n",
       "      <td>41455</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "29258      41410 2011-07-11      9112  216425     132.5250    132.5250   \n",
       "29238      41386 2011-07-11      9092  216425     134.6625    134.6625   \n",
       "29300      41477 2011-07-11      9164  216425     134.6625    134.6625   \n",
       "29274      41435 2011-07-11      9132  216425     131.1000    131.1000   \n",
       "29286      41455 2011-07-11      9147  216425     133.9500    133.9500   \n",
       "\n",
       "       is_featured_sku  is_display_sku  month  year  day_of_week  \\\n",
       "29258                0               0      7  2011            0   \n",
       "29238                0               0      7  2011            0   \n",
       "29300                0               0      7  2011            0   \n",
       "29274                0               0      7  2011            0   \n",
       "29286                0               0      7  2011            0   \n",
       "\n",
       "       day_of_month  discount  \n",
       "29258            11       0.0  \n",
       "29238            11       0.0  \n",
       "29300            11       0.0  \n",
       "29274            11       0.0  \n",
       "29286            11       0.0  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>29258</th>\n",
       "      <td>4.007333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29238</th>\n",
       "      <td>2.197225</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29300</th>\n",
       "      <td>3.891820</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29274</th>\n",
       "      <td>3.044522</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29286</th>\n",
       "      <td>3.737670</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       units_sold\n",
       "29258    4.007333\n",
       "29238    2.197225\n",
       "29300    3.891820\n",
       "29274    3.044522\n",
       "29286    3.737670"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "149378     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "149358     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "149420     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "149394     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "149406     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "        is_featured_sku  is_display_sku  month  year  day_of_week  \\\n",
       "149378                0               0      7  2013            1   \n",
       "149358                0               0      7  2013            1   \n",
       "149420                0               0      7  2013            1   \n",
       "149394                0               0      7  2013            1   \n",
       "149406                0               0      7  2013            1   \n",
       "\n",
       "        day_of_month  discount  \n",
       "149378             9       0.0  \n",
       "149358             9       0.0  \n",
       "149420             9       0.0  \n",
       "149394             9       0.0  \n",
       "149406             9       0.0  "
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>4.276666</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>2.890372</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>3.784190</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>2.564949</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>4.110874</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        units_sold\n",
       "149378    4.276666\n",
       "149358    2.890372\n",
       "149420    3.784190\n",
       "149394    2.564949\n",
       "149406    4.110874"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.head()\n",
    "y_test.head()\n",
    "X_train.head()\n",
    "y_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "44762b87",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test.reset_index(drop=True, inplace=True)\n",
    "y_test.reset_index(drop=True, inplace=True)\n",
    "X_train.reset_index(drop=True, inplace=True)\n",
    "y_train.reset_index(drop=True, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "5d100330",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_sarimax = X_test\n",
    "y_test_sarimax = y_test\n",
    "X_train_sarimax = X_train\n",
    "y_train_sarimax = y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "b517472f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>41410</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>41386</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41477</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>41435</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>41455</td>\n",
       "      <td>2011-07-11</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "0      41410 2011-07-11      9112  216425     132.5250    132.5250   \n",
       "1      41386 2011-07-11      9092  216425     134.6625    134.6625   \n",
       "2      41477 2011-07-11      9164  216425     134.6625    134.6625   \n",
       "3      41435 2011-07-11      9132  216425     131.1000    131.1000   \n",
       "4      41455 2011-07-11      9147  216425     133.9500    133.9500   \n",
       "\n",
       "   is_featured_sku  is_display_sku  month  year  day_of_week  day_of_month  \\\n",
       "0                0               0      7  2011            0            11   \n",
       "1                0               0      7  2011            0            11   \n",
       "2                0               0      7  2011            0            11   \n",
       "3                0               0      7  2011            0            11   \n",
       "4                0               0      7  2011            0            11   \n",
       "\n",
       "   discount  \n",
       "0       0.0  \n",
       "1       0.0  \n",
       "2       0.0  \n",
       "3       0.0  \n",
       "4       0.0  "
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.007333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.197225</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.891820</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.044522</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.737670</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   units_sold\n",
       "0    4.007333\n",
       "1    2.197225\n",
       "2    3.891820\n",
       "3    3.044522\n",
       "4    3.737670"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "0     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "1     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "2     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "3     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "4     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "   is_featured_sku  is_display_sku  month  year  day_of_week  day_of_month  \\\n",
       "0                0               0      7  2013            1             9   \n",
       "1                0               0      7  2013            1             9   \n",
       "2                0               0      7  2013            1             9   \n",
       "3                0               0      7  2013            1             9   \n",
       "4                0               0      7  2013            1             9   \n",
       "\n",
       "   discount  \n",
       "0       0.0  \n",
       "1       0.0  \n",
       "2       0.0  \n",
       "3       0.0  \n",
       "4       0.0  "
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.276666</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.890372</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.784190</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2.564949</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4.110874</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   units_sold\n",
       "0    4.276666\n",
       "1    2.890372\n",
       "2    3.784190\n",
       "3    2.564949\n",
       "4    4.110874"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.head()\n",
    "y_test.head()\n",
    "X_train.head()\n",
    "y_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "2e1b5315",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pandas.core.frame.DataFrame"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "f569f97b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "pandas.core.frame.DataFrame"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "type(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "2b145eb0",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test.set_index('week', inplace=True)\n",
    "X_train.set_index('week', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "0837aa89",
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_random_forest(X_train, y_train):\n",
    "    # Creating a Random Forest regressor\n",
    "    #rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "\n",
    "    # Training the model\n",
    "    #rf_regressor.fit(X_train, y_train)\n",
    "\n",
    "    # Making predictions on the testing set\n",
    "    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)\n",
    "    rf_regressor = RFE(estimator = rf_regressor, n_features_to_select=7)\n",
    "    fit = rf_regressor.fit(X_train, y_train)\n",
    "    y_pred = fit.predict(X_test)\n",
    "    selected_features = X_train.columns[rf_regressor.support_]\n",
    "    print(\"Selected Features:\",selected_features)\n",
    "    \n",
    "    return y_pred, fit"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "65373fd6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Selected Features: Index(['record_ID', 'store_id', 'total_price', 'base_price', 'month',\n",
      "       'day_of_month', 'discount'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "y_pred, fit = train_random_forest(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "b40b8f74",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([3.58011911, 2.28392991, 3.16194372, ..., 3.00695344, 3.59257953,\n",
       "       3.50965033])"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "b8b1e11b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 87.73 %.\n"
     ]
    }
   ],
   "source": [
    "#Evaluate accuracy using MAPE\n",
    "y_true = np.array(y_test['units_sold'])\n",
    "sumvalue=np.sum(y_true)\n",
    "mape=np.sum(np.abs((y_true - y_pred)))/sumvalue*100\n",
    "accuracy=100-mape\n",
    "print('Accuracy:', round(accuracy,2),'%.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "7d2ab269",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE: 0.5333787307439695\n",
      "MSE: 0.2844928704100479\n"
     ]
    }
   ],
   "source": [
    "# Find RMSE\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "rmse = np.sqrt(mse)\n",
    "print(\"RMSE:\",rmse)\n",
    "print(\"MSE:\",mse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "4c701a54",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_predictions(y_test, y_pred):\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    plt.scatter(y_test, y_pred, color='blue')\n",
    "    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')\n",
    "    plt.xlabel('Actual units_sold')\n",
    "    plt.ylabel('Predicted units_sold')\n",
    "    plt.title('Actual vs. Predicted units_sold')\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "a9922dde",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test1 = y_test.values.flatten()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "30d065b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([4.00733319, 2.19722458, 3.8918203 , ..., 3.21887582, 3.76120012,\n",
       "       3.04452244])"
      ]
     },
     "execution_count": 64,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_test1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "4006ae74",
   "metadata": {},
   "outputs": [],
   "source": [
    "comp = pd.DataFrame(data=[y_test1,y_pred]).T\n",
    "comp.columns=['y_test','y_pred']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "fe856a80",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>y_test</th>\n",
       "      <th>y_pred</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.007333</td>\n",
       "      <td>3.580119</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.197225</td>\n",
       "      <td>2.283930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.891820</td>\n",
       "      <td>3.161944</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.044522</td>\n",
       "      <td>2.808911</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.737670</td>\n",
       "      <td>3.577226</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1711</th>\n",
       "      <td>3.496508</td>\n",
       "      <td>3.084232</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1712</th>\n",
       "      <td>3.367296</td>\n",
       "      <td>2.882606</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1713</th>\n",
       "      <td>3.218876</td>\n",
       "      <td>3.006953</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1714</th>\n",
       "      <td>3.761200</td>\n",
       "      <td>3.592580</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1715</th>\n",
       "      <td>3.044522</td>\n",
       "      <td>3.509650</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1716 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        y_test    y_pred\n",
       "0     4.007333  3.580119\n",
       "1     2.197225  2.283930\n",
       "2     3.891820  3.161944\n",
       "3     3.044522  2.808911\n",
       "4     3.737670  3.577226\n",
       "...        ...       ...\n",
       "1711  3.496508  3.084232\n",
       "1712  3.367296  2.882606\n",
       "1713  3.218876  3.006953\n",
       "1714  3.761200  3.592580\n",
       "1715  3.044522  3.509650\n",
       "\n",
       "[1716 rows x 2 columns]"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "comp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "29089638",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA0EAAAIhCAYAAACIfrE3AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8pXeV/AAAACXBIWXMAAA9hAAAPYQGoP6dpAAC5kUlEQVR4nOzdeXgT1foH8G8aKFuhQClrKyDKooKCCoigILjgcosF2URBVPSyLxYVUUBRFIGCICouuGBBtFWuet3Qoii4/RSRiywia9lByl5oen5/HKdNs8ycSSaZJP1+nmeettOTmZPJJJl3zjnvcQghBIiIiIiIiMqIOLsrQEREREREFE4MgoiIiIiIqExhEERERERERGUKgyAiIiIiIipTGAQREREREVGZwiCIiIiIiIjKFAZBRERERERUpjAIIiIiIiKiMoVBEBERERERlSkMgogopj333HNwOBy46KKLAt7G7t27MXnyZKxZs8a6iuno3LkzOnfuHJZ9mTVo0CA4HI7ipUKFCmjWrBkmTZqE06dPh3z/27Ztg8PhwOuvv168bvLkyXA4HKa3lZWVhdmzZ1tXOTeNGjXCoEGDQrJtIw6HA5MnTy7+e/369Zg8eTK2bdtmS33MGDRoEBo1aqRU1vN5EhGZwSCIiGLaa6+9BgD43//+hx9++CGgbezevRtTpkwJWxAU6SpVqoTVq1dj9erV+OCDD9CuXTs8/vjjGDhwoC31ueeee7B69WrTjwtlEGSn1atX45577in+e/369ZgyZUpUBEFEROHCIIiIYtbPP/+M3377DTfddBMA4NVXX7W5RrEhLi4O7du3R/v27dG9e3e8+eab6NSpE5YuXYq8vDy/jzt16lRI6pOSkoL27duHZNvRqH379khJSbG7GkREEY1BEBHFLC3oefrpp9GhQwcsWbIEJ0+e9CqXl5eHIUOGIDU1FfHx8ahfvz569eqFffv2YcWKFbj88ssBAHfddVdxNzCtG46/rmu+uvVMmTIF7dq1Q82aNVGtWjW0adMGr776KoQQpp9bjx490LBhQxQVFXn9r127dmjTpk3x3++++y7atWuHxMREVK5cGeeeey4GDx5sep96tCBk+/btAGR3sJtvvhk5OTlo3bo1KlasiClTpgAA9u7di/vuuw8pKSmIj49H48aNMWXKFBQWFpba5u7du9G7d29UrVoViYmJ6NOnD/bu3eu1b3/d4bKysnDFFVcgISEBCQkJuOSSS4rPic6dO+Pjjz/G9u3bS3Xv05w5cwZTp05F8+bNUaFCBSQnJ+Ouu+7CgQMHSu3j7NmzGD9+POrWrYvKlSujY8eO+PHHH5WO2YoVK+BwOLBixYpS6311+Rs0aBASEhLw559/4sYbb0RCQgJSU1Mxbtw4FBQUlHq8+/n5+uuv47bbbgMAdOnSpfh5atv+9ddfcfPNN6N27dqoUKEC6tevj5tuugm7du1Seg4A8Ndff6Fv376oX78+KlSogDp16qBr166lWk6Lioowffr04uNZu3Zt3HnnnUr7OXr0KO69914kJSUhISEBN9xwAzZt2qRcPyIiX8rZXQEiolA4deoUFi9ejMsvvxwXXXQRBg8ejHvuuQfvvvtuqW5beXl5uPzyy3H27FlMmDABrVq1wqFDh/DZZ5/h77//Rps2bbBw4ULcddddmDhxYnGrUiB32rdt24b77rsP55xzDgDg+++/x4gRI5CXl4fHHnvM1LYGDx6MtLQ0fPXVV+jWrVvx+g0bNuDHH3/Ec889B0B2jerTpw/69OmDyZMno2LFiti+fTu++uor0/XX8+effwIAkpOTi9f98ssv+OOPPzBx4kQ0btwYVapUwd69e9G2bVvExcXhscceQ5MmTbB69WpMnToV27Ztw8KFCwHI169bt27YvXs3pk2bhqZNm+Ljjz9Gnz59lOrz2GOP4YknnkB6ejrGjRuHxMRErFu3rjhImz9/PoYMGYItW7bg/fffL/XYoqIipKWlYeXKlRg/fjw6dOiA7du3Y9KkSejcuTN+/vlnVKpUCQBw77334s0338QDDzyAa6+9FuvWrUN6ejqOHTsW9DH1dPbsWfzrX//C3XffjXHjxuGbb77BE088gcTERL/nz0033YSnnnoKEyZMwPPPP18cHDdp0gQnTpzAtddei8aNG+P5559HnTp1sHfvXuTm5pqq/4033giXy4Xp06fjnHPOwcGDB7Fq1SocOXKkuMy///1vLFiwAMOHD8fNN9+Mbdu24dFHH8WKFSvwyy+/oFatWj63LYRAjx49sGrVKjz22GO4/PLL8d1336F79+7qB46IyBdBRBSD3nzzTQFAvPjii0IIIY4dOyYSEhJEp06dSpUbPHiwKF++vFi/fr3fbf30008CgFi4cKHX/66++mpx9dVXe60fOHCgaNiwod9tulwucfbsWfH444+LpKQkUVRUZLhNd2fPnhV16tQR/fv3L7V+/PjxIj4+Xhw8eFAIIcSMGTMEAHHkyBHd7akaOHCgqFKlijh79qw4e/asOHDggJgzZ45wOBzi8ssvLy7XsGFD4XQ6xcaNG0s9/r777hMJCQli+/btpdZr9fzf//4nhBDihRdeEADEsmXLSpW79957vV6LSZMmCfevs7/++ks4nU5x++236z6Xm266yedrtHjxYgFAZGdnl1qvnQfz588XQgjxxx9/CABizJgxpcq9/fbbAoAYOHCg7v5zc3MFAJGbm1tq/datW72e48CBAwUAsXTp0lJlb7zxRtGsWbNS6wCISZMmFf/97rvv+tzPzz//LACIDz74QLeeeg4ePCgAiNmzZ/stox2noUOHllr/ww8/CABiwoQJxes83zeffPKJACDmzJlT6rFPPvmk1/MkIjKD3eGIKCa9+uqrqFSpEvr27QsASEhIwG233YaVK1di8+bNxeU++eQTdOnSBS1atAh5nbRWm8TERDidTpQvXx6PPfYYDh06hP3795vaVrly5TBgwADk5OQgPz8fAOByufDWW28hLS0NSUlJAFDcla93796GY3ZUnThxAuXLl0f58uWRnJyM0aNHo3v37l4tKq1atULTpk1Lrfvoo4/QpUsX1K9fH4WFhcWLdmf/66+/BgDk5uaiatWq+Ne//lXq8f379zes3xdffAGXy4Vhw4YF9Pw++ugjVK9eHbfcckupOl5yySWoW7ducfe13NxcAMDtt99e6vG9e/dGuXLWd7RwOBy45ZZbSq1r1apVceuWWeeddx5q1KiBBx98EC+++CLWr19vehs1a9ZEkyZN8Oyzz2LWrFn49ddfvbpoasfJM1te27Zt0aJFC3z55Zd+t+/vGKucB0REehgEEVHM+fPPP/HNN9/gpptughACR44cwZEjR9CrVy8AJRnjAODAgQNhGUT+448/4rrrrgMAvPzyy/juu+/w008/4ZFHHgEQWNKAwYMH4/Tp01iyZAkA4LPPPsOePXtw1113FZe56qqr8MEHH6CwsBB33nknUlJScNFFF2Hx4sUBP5dKlSrhp59+wk8//YS1a9fiyJEj+Pjjj9GgQYNS5erVq+f12H379uHDDz8sDqK05cILLwQAHDx4EABw6NAh1KlTx+vxdevWNayfNm4n0Nd13759OHLkCOLj473quXfv3lJ19FWncuXKFQehVqpcuTIqVqxYal2FChUCTk2emJiIr7/+GpdccgkmTJiACy+8EPXr18ekSZNw9uxZpW04HA58+eWXuP766zF9+nS0adMGycnJGDlyZHGXOu04+Tof6tevX/x/Xw4dOuTzeKqcB0REejgmiIhizmuvvQYhBN577z289957Xv9/4403MHXqVDidTiQnJ5saBO6pYsWKxS0x7rQLZc2SJUtQvnx5fPTRR6UuZD/44IOA933BBRegbdu2WLhwIe677z4sXLgQ9evXLw62NGlpaUhLS0NBQQG+//57TJs2Df3790ejRo1wxRVXmN5vXFwcLrvsMsNyvpIV1KpVC61atcKTTz7p8zH169cHACQlJflMMOArMYInbVzSrl27kJqaaljeVx2TkpLw6aef+vx/1apVi+uo1ck9ACwsLNS9sNdo54FnYgPPcyeUWrZsiSVLlkAIgbVr1+L111/H448/jkqVKuGhhx5S2kbDhg2LE05s2rQJS5cuxeTJk3HmzBm8+OKLxcdpz549XoHp7t27/Y4HAuQx1o6neyCkch4QEelhSxARxRSXy4U33ngDTZo0QW5urtcybtw47NmzB5988gkAoHv37sjNzcXGjRv9brNChQoAfLfWNGrUCJs2bSp1IXvo0CGsWrWqVDmHw4Fy5crB6XQWrzt16hTeeuutoJ7vXXfdhR9++AHffvstPvzwQwwcOLDUPjyfx9VXX41nnnkGgMwMFm4333wz1q1bhyZNmuCyyy7zWrQgqEuXLjh27Bj+85//lHp8VlaW4T6uu+46OJ1OvPDCC7rlKlSo4PM1vfnmm3Ho0CG4XC6fdWzWrBkAFGcFfPvtt0s9funSpV6Z7nzRsgeuXbu21HrP5xwsvfNX43A4cPHFFyMzMxPVq1fHL7/8EtC+mjZtiokTJ6Jly5bF27jmmmsAAIsWLSpV9qeffsIff/yBrl27+t1ely5dAHgfY5XzgIhID1uCiCimfPLJJ9i9ezeeeeYZn6mrL7roIsybNw+vvvoqbr75Zjz++OP45JNPcNVVV2HChAlo2bIljhw5gk8//RRjx45F8+bN0aRJE1SqVAlvv/02WrRogYSEBNSvXx/169fHHXfcgZdeegkDBgzAvffei0OHDmH69OmoVq1aqf3edNNNmDVrFvr3748hQ4bg0KFDmDFjRvEFaqD69euHsWPHol+/figoKPAad/HYY49h165d6Nq1K1JSUnDkyBHMmTMH5cuXx9VXX11crly5crj66qt1x2dY4fHHH8cXX3yBDh06YOTIkWjWrBlOnz6Nbdu24b///S9efPFFpKSk4M4770RmZibuvPNOPPnkkzj//PPx3//+F5999pnhPho1aoQJEybgiSeewKlTp9CvXz8kJiZi/fr1OHjwYHGq7pYtWyInJwcvvPACLr300uIWrr59++Ltt9/GjTfeiFGjRqFt27YoX748du3ahdzcXKSlpeHWW29FixYtMGDAAMyePRvly5dHt27dsG7dOsyYMcPr9felbt266NatG6ZNm4YaNWqgYcOG+PLLL5GTkxP0cXZ30UUXAQAWLFiAqlWromLFimjcuDFWr16N+fPno0ePHjj33HMhhEBOTg6OHDmCa6+9Vmnba9euxfDhw3Hbbbfh/PPPR3x8PL766iusXbu2uCWpWbNmGDJkCObOnYu4uDh07969ODtcamoqxowZ43f71113Ha666iqMHz8eJ06cwGWXXYbvvvsu6JsHRETMDkdEMaVHjx4iPj5e7N+/32+Zvn37inLlyom9e/cKIYTYuXOnGDx4sKhbt64oX768qF+/vujdu7fYt29f8WMWL14smjdvLsqXL++VleqNN94QLVq0EBUrVhQXXHCBeOedd3xmh3vttddEs2bNRIUKFcS5554rpk2bJl599VUBQGzdurW4nEp2OHf9+/cXAMSVV17p9b+PPvpIdO/eXTRo0EDEx8eL2rVrixtvvFGsXLmyVDkASvvUssMZadiwobjpppt8/u/AgQNi5MiRonHjxqJ8+fKiZs2a4tJLLxWPPPKIOH78eHG5Xbt2iZ49e4qEhARRtWpV0bNnT7Fq1SrD7HCaN998U1x++eWiYsWKIiEhQbRu3brU4w4fPix69eolqlevLhwOR6ltnD17VsyYMUNcfPHFxY9v3ry5uO+++8TmzZuLyxUUFIhx48aJ2rVri4oVK4r27duL1atXi4YNGxpmhxNCiD179ohevXqJmjVrisTERDFgwIDirG2e2eF8HXdfz93z/BRCiNmzZ4vGjRsLp9NZvO0NGzaIfv36iSZNmohKlSqJxMRE0bZtW/H6668b1luzb98+MWjQING8eXNRpUoVkZCQIFq1aiUyMzNFYWFhcTmXyyWeeeYZ0bRpU1G+fHlRq1YtMWDAALFz585S2/P1vjly5IgYPHiwqF69uqhcubK49tprxYYNG5gdjoiC4hAigFn6iIiIiIiIohTHBBERERERUZnCMUFERETkpaioyGvOH0+hmA+JiCgc2BJEREREXgYPHuw1T5LnQkQUrTgmiIiIiLxs27bNcM4ilfmiiIgiEYMgIiIiIiIqU9gdjoiIiIiIypSoHtFYVFSE3bt3o2rVqnA4HHZXh4iIiIiIbCKEwLFjx1C/fn3Exem39UR1ELR7926kpqbaXQ0iIiIiIooQO3fuREpKim6ZqA6CqlatCkA+0WrVqtlcGyIiIiIissvRo0eRmppaHCPoieogSOsCV61aNQZBRERERESkNEyGiRGIiIiIiKhMYRBERERERERlCoMgIiIiIiIqUxgEERERERFRmcIgiIiIiIiIyhQGQUREREREVKYwCCIiIiIiojKFQRAREREREZUpDIKIiIiIiKhMYRBERERERERlCoMgIiIiIiIqUxgEERERERFRmcIgiIiIiIiIyhQGQUREREREVKYwCCIiIiIiojKFQRAREREREZUpDIKIiIiIiCgwO3faXYOAMAgiIiIiIiJztmwBbr4ZaNUKOHDA7tqYxiCIiIiIiIjUnToFtG8PfPwxcOIE8M03dtfINAZBRERERESkrlIlICMDuPZaYO1aoGdPu2tkGoMgIiIiIiLyb9Mm4IYbgC+/LFk3bhzw2WdA8+b21SsI5eyuABERERERRaATJ4CpU4GZM4GzZ4F9+4BffgEcDsDptLt2QWFLEBERERERlRACWLpUtvI8/bQMgLp3l+scDrtrZwm2BBERERERkfTHH8Dw4cBXX8m/GzcGZs8GbrklZgIggEEQERERERFp1q2TAVDFisBDDwHjx8tECDGGQRARERERUVklBLBtm2zxAYBevYBJk4CBA0vWxSCOCSIiIiIiKovWrgWuvhpo2xb4+2+5zuEAJk+O6QAIYBBERERERFS2HDkCjBoFtGkDrFwps8D9+KPdtQorBkFERERERGVBURHw+utAs2bAc88BLpfs/rZhA3D99XbXLqw4JoiIiIiIKNadOQNccw3w3Xfy72bNgLlzgWuvtbdeNmFLEBERERFRrIuPl4FPlSrA9OlyPFAZDYAABkFERERERLGnqAh45RXgzz9L1j3zDLBxI5CRIYOiMoxBEBERERFRLPnpJ6B9e+Dee4ExY0rW16oFNGhgX70iCIMgIiIiIqJYcPCgDHzatZOBULVqQNeuslWISmFiBCIiIiKiaOZyAQsWAI88UjLfzx13yLE/devaW7cIxSCIiIiIiCiavfwyMHSo/P3ii4F584COHe2tU4RjdzgiIiIiomgjRMnvgwYBl10mg5+ff2YApMDWIGjy5MlwOByllrpssiMiIiIi8q2wEJgzB+jcWf4OABUrAj/+CAwbBpRjRy8Vth+lCy+8EMuXLy/+2+l02lgbIiIiIqII9c03MtBZt07+vWQJMGCA/N3hsK9eUcj2IKhcuXLKrT8FBQUoKCgo/vvo0aOhqhYRERERUWTYvVvO7ZOVJf+uWROYNg3o18/eekUx28cEbd68GfXr10fjxo3Rt29f/PXXX37LTps2DYmJicVLampqGGtKRERERBRGLhcwYwbQrJkMgBwO4P77gU2bgCFDAPagCphDCPdRVeH1ySef4OTJk2jatCn27duHqVOnYsOGDfjf//6HpKQkr/K+WoJSU1ORn5+PatWqhbPqREREREShJQRw1VXAt9/KuX+efx649FK7axWxjh49isTERKXYwNYgyNOJEyfQpEkTjB8/HmPHjjUsb+aJEhERERFFvJ07gcREOdEpAKxdKzO+DRoExNneiSuimYkNIupIVqlSBS1btsTmzZvtrgoRERERUfgUFMhxPs2bA1OmlKxv1QoYPJgBkMUi6mgWFBTgjz/+QL169eyuChERERFReHz6KdCyJTBhAnDyJPDrr3I8EIWMrUHQAw88gK+//hpbt27FDz/8gF69euHo0aMYOHCgndUiIiIiIgq9bduAW28FuncHNm8G6tYF3noL+PJLJj0IMVtTZO/atQv9+vXDwYMHkZycjPbt2+P7779Hw4YN7awWEREREVFoLVsG9O0LnD4tA55Ro4BJk0rGAlFI2RoELVmyxM7dExERERHZo21boHx5oH17YN484MIL7a5RmRJRY4KIiIiIiGLSli3A9Oklf9erJ7O+ffUVAyAbMAgiIiIiIgqVkyeBxx6Tgc6DDwKff17yv6ZN5QSoFHa2docjIiIiIopJQgAffACMGQNs3y7XXXstwLHvEYFBEBERERGRlTZuBEaOLGn1OeccIDNTZoJjy09EYBBERERERGSVoiLg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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plot_predictions(y_test1, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "f3cd4c2b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x283f7e950>]"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot (y_test1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "29b4c6d4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x283fd9d10>]"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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+48//hj169dH+/bt8cQTT4RYmqKlHOy2g2gpBwA4fPgwZs+ejfvuuy/st0jVhTjr4nuXY8eOoaysDA0bNgz5vmHDhsjNzXVJKudQFAVjx45F37590aFDh6rvhwwZgltvvRXNmjXDnj178Mwzz2DAgAFYvXo1EhISkJubi2rVqqFOnToh6XmpnHr27IkPP/wQbdu2xeHDh/HCCy/gsssuw+bNm6ueQase7Nu3DwCiogyC+eqrr5CXl1d1+jjgj3qgRtS7z83NRWpqalj6qampniybwsJCPPXUU7jzzjtDDoMbPnw4WrRogbS0NGzatAnjx4/H+vXrq5YLo6EcRLSDaCiHSj744AMkJSXh5ptvDvk+knXBlwpKJcEzSKBiIFd/Fw2MGTMGGzZswJIlS0K+v+2226r+36FDB3Tr1g3NmjXD7NmzwyplMF4qpyFDhlT9v2PHjujduzdatWqFDz74oMoBzko98FIZBDN16lQMGTIEGRkZVd/5oR7oIeLda13vxbIpKSnB7bffjvLycrz99tshv91///1V/+/QoQPatGmDbt26Yc2aNbj00ksBeL8cRLUDr5dDJe+99x6GDx+OxMTEkO8jWRd8ucRTv359xMbGhmlzR44cCZtReZ2HHnoI33zzDRYsWIDGjRsbXpueno5mzZph586dAIC0tDQUFxfj5MmTIdd5uZxq1qyJjh07YufOnVW7eYzqQTSVwb59+zB//nz88Y9/NLzOD/VA1LtPS0vD4cOHw9I/evSop8qmpKQEv//977Fnzx7MmzcvxHqixaWXXor4+PiQOhIN5RCMlXYQLeWwePFibN++3bSvAJytC75UUKpVq4auXbtWmaQqmTdvHi677DKXpBKLoigYM2YMZs2ahZ9//hktWrQwvef48ePIzs5Geno6AKBr166Ij48PKaecnBxs2rTJs+VUVFSErVu3Ij09vcpMGfx8xcXFWLhwYdXzRVMZTJs2Dampqbj22msNr/NDPRD17nv37o38/HysWLGi6ppff/0V+fn5nimbSuVk586dmD9/PurVq2d6z+bNm1FSUlJVR6KhHNRYaQfRUg5Tp05F165d0blzZ9NrHa0LXC61UcSMGTOU+Ph4ZerUqcqWLVuURx99VKlZs6ayd+9et0UTwoMPPqikpKQoWVlZSk5OTtXf2bNnFUVRlFOnTimPP/648ssvvyh79uxRFixYoPTu3Vtp1KiRUlBQUJXOqFGjlMaNGyvz589X1qxZowwYMEDp3LmzUlpa6tajcfH4448rWVlZyu7du5Xly5cr1113nZKUlFT1nl966SUlJSVFmTVrlrJx40bljjvuUNLT06OqDBRFUcrKypSmTZsqTz75ZMj30VwPTp06paxdu1ZZu3atAkB59dVXlbVr11btThH17q+55hqlU6dOyrJly5Rly5YpHTt2VK677rqIP68eRuVQUlKiXH/99Urjxo2VdevWhfQVRUVFiqIoym+//aY899xzysqVK5U9e/Yos2fPVtq1a6d06dIlaspBZDuQuRzM2oSiKEp+fr5So0YNZeLEiWH3R7ou+FZBURRFeeutt5RmzZop1apVUy699NKQLbheB4Dm37Rp0xRFUZSzZ88qgwYNUho0aKDEx8crTZs2Ve6++25l//79IemcO3dOGTNmjFK3bl2levXqynXXXRd2jczcdtttSnp6uhIfH69kZGQoN998s7J58+aq38vLy5Vnn31WSUtLUxISEpR+/fopGzduDEnD62WgKIry448/KgCU7du3h3wfzfVgwYIFmm3g7rvvVhRF3Ls/fvy4Mnz4cCUpKUlJSkpShg8frpw8eTJCT2mOUTns2bNHt69YsGCBoiiKsn//fqVfv35K3bp1lWrVqimtWrVSHn74YeX48eMh+Xi5HES2A5nLwaxNKIqiTJ48WalevbqSl5cXdn+k60JAURSFz+ZCEARBEAThLL70QSEIgiAIQm5IQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjpIQSEIgiAIQjr+P1DZa2yXgwhjAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot (y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "bee387ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculation of safety stock factor\n",
    "def calculate_safety_factor(desired_service_level, standard_deviation):\n",
    "    # Calculation Z-score corresponding to the desired service level\n",
    "    z_score = stats.norm.ppf(desired_service_level)\n",
    "    \n",
    "    #Calculate safety factor\n",
    "    safety_factor = z_score * standard_deviation\n",
    "    \n",
    "    return safety_factor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "ad0d8a97",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get user input for sku_id\n",
    "#store_id = int(input(\"Enter store_id: \"))\n",
    "store_id=8091"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "a991d51b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get desired service level\n",
    "#desired_service_level = float(input(\"Enter desired service level (ex: 0.95 for 95%): \"))\n",
    "desired_service_level = 0.9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "68be59c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculation of standard_deviation\n",
    "filtered_df = df_processed[(df['store_id'] == store_id) & (df_processed['sku_id'] == sku_id)]\n",
    "standard_deviation = filtered_df['units_sold'].std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "1c80396e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculation of re-order point\n",
    "def calculate_reorder_point (demand_forecast, lead_time, safety_factor):\n",
    "    average_demand = np.mean(demand_forecast)\n",
    "    demand_std = np.std(demand_forecast)\n",
    "    safety_stock = safety_factor * demand_std\n",
    "    safety_stock = safety_stock.round()\n",
    "    reorder_point = average_demand * lead_time + safety_stock\n",
    "    reorder_point = reorder_point.round()\n",
    "    return reorder_point, safety_stock"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "c41210f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get user input for sku_id\n",
    "#lead_time = int(input(\"Enter lead time in weeks: \"))\n",
    "lead_time = 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "b8991707",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1716, 12)"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "(1716, 1)"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape\n",
    "y_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "a5230d33",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>week</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2011-07-11</th>\n",
       "      <td>41410</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2011-07-11</th>\n",
       "      <td>41386</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2011-07-11</th>\n",
       "      <td>41477</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2011-07-11</th>\n",
       "      <td>41435</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2011-07-11</th>\n",
       "      <td>41455</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "            record_ID  store_id  sku_id  total_price  base_price  \\\n",
       "week                                                               \n",
       "2011-07-11      41410      9112  216425     132.5250    132.5250   \n",
       "2011-07-11      41386      9092  216425     134.6625    134.6625   \n",
       "2011-07-11      41477      9164  216425     134.6625    134.6625   \n",
       "2011-07-11      41435      9132  216425     131.1000    131.1000   \n",
       "2011-07-11      41455      9147  216425     133.9500    133.9500   \n",
       "\n",
       "            is_featured_sku  is_display_sku  month  year  day_of_week  \\\n",
       "week                                                                    \n",
       "2011-07-11                0               0      7  2011            0   \n",
       "2011-07-11                0               0      7  2011            0   \n",
       "2011-07-11                0               0      7  2011            0   \n",
       "2011-07-11                0               0      7  2011            0   \n",
       "2011-07-11                0               0      7  2011            0   \n",
       "\n",
       "            day_of_month  discount  \n",
       "week                                \n",
       "2011-07-11            11       0.0  \n",
       "2011-07-11            11       0.0  \n",
       "2011-07-11            11       0.0  \n",
       "2011-07-11            11       0.0  \n",
       "2011-07-11            11       0.0  "
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>4.007333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2.197225</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.891820</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3.044522</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>3.737670</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   units_sold\n",
       "0    4.007333\n",
       "1    2.197225\n",
       "2    3.891820\n",
       "3    3.044522\n",
       "4    3.737670"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.head()\n",
    "y_test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "3293bcca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>41410</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.007333</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>41386</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.197225</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41477</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.891820</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>41435</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>131.1000</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.044522</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>41455</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.737670</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   record_ID  store_id  sku_id  total_price  base_price  is_featured_sku  \\\n",
       "0      41410      9112  216425     132.5250    132.5250                0   \n",
       "1      41386      9092  216425     134.6625    134.6625                0   \n",
       "2      41477      9164  216425     134.6625    134.6625                0   \n",
       "3      41435      9132  216425     131.1000    131.1000                0   \n",
       "4      41455      9147  216425     133.9500    133.9500                0   \n",
       "\n",
       "   is_display_sku  month  year  day_of_week  day_of_month  discount  \\\n",
       "0               0      7  2011            0            11       0.0   \n",
       "1               0      7  2011            0            11       0.0   \n",
       "2               0      7  2011            0            11       0.0   \n",
       "3               0      7  2011            0            11       0.0   \n",
       "4               0      7  2011            0            11       0.0   \n",
       "\n",
       "   units_sold  \n",
       "0    4.007333  \n",
       "1    2.197225  \n",
       "2    3.891820  \n",
       "3    3.044522  \n",
       "4    3.737670  "
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.reset_index(drop=True, inplace=True)\n",
    "test_df = pd.concat([X_test, y_test], axis=1)\n",
    "test_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "5a594a60",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1711</th>\n",
       "      <td>1210</td>\n",
       "      <td>9713</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.496508</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1712</th>\n",
       "      <td>1066</td>\n",
       "      <td>9578</td>\n",
       "      <td>216425</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.367296</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1713</th>\n",
       "      <td>1044</td>\n",
       "      <td>9532</td>\n",
       "      <td>216425</td>\n",
       "      <td>128.9625</td>\n",
       "      <td>128.9625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.218876</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1714</th>\n",
       "      <td>1107</td>\n",
       "      <td>9611</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.761200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1715</th>\n",
       "      <td>1086</td>\n",
       "      <td>9672</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3.044522</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      record_ID  store_id  sku_id  total_price  base_price  is_featured_sku  \\\n",
       "1711       1210      9713  216425     133.2375    133.2375                0   \n",
       "1712       1066      9578  216425     132.5250    132.5250                0   \n",
       "1713       1044      9532  216425     128.9625    128.9625                0   \n",
       "1714       1107      9611  216425     133.9500    133.9500                0   \n",
       "1715       1086      9672  216425     133.9500    133.9500                0   \n",
       "\n",
       "      is_display_sku  month  year  day_of_week  day_of_month  discount  \\\n",
       "1711               0      1  2011            0            17       0.0   \n",
       "1712               0      1  2011            0            17       0.0   \n",
       "1713               0      1  2011            0            17       0.0   \n",
       "1714               0      1  2011            0            17       0.0   \n",
       "1715               0      1  2011            0            17       0.0   \n",
       "\n",
       "      units_sold  \n",
       "1711    3.496508  \n",
       "1712    3.367296  \n",
       "1713    3.218876  \n",
       "1714    3.761200  \n",
       "1715    3.044522  "
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#test_df['units_sold'] = np.exp(df['units_sold'])\n",
    "test_df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "6d652cc7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1711</th>\n",
       "      <td>1210</td>\n",
       "      <td>9713</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>33.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1712</th>\n",
       "      <td>1066</td>\n",
       "      <td>9578</td>\n",
       "      <td>216425</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>132.5250</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>29.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1713</th>\n",
       "      <td>1044</td>\n",
       "      <td>9532</td>\n",
       "      <td>216425</td>\n",
       "      <td>128.9625</td>\n",
       "      <td>128.9625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>25.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1714</th>\n",
       "      <td>1107</td>\n",
       "      <td>9611</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>43.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1715</th>\n",
       "      <td>1086</td>\n",
       "      <td>9672</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>133.9500</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>17</td>\n",
       "      <td>0.0</td>\n",
       "      <td>21.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      record_ID  store_id  sku_id  total_price  base_price  is_featured_sku  \\\n",
       "1711       1210      9713  216425     133.2375    133.2375                0   \n",
       "1712       1066      9578  216425     132.5250    132.5250                0   \n",
       "1713       1044      9532  216425     128.9625    128.9625                0   \n",
       "1714       1107      9611  216425     133.9500    133.9500                0   \n",
       "1715       1086      9672  216425     133.9500    133.9500                0   \n",
       "\n",
       "      is_display_sku  month  year  day_of_week  day_of_month  discount  \\\n",
       "1711               0      1  2011            0            17       0.0   \n",
       "1712               0      1  2011            0            17       0.0   \n",
       "1713               0      1  2011            0            17       0.0   \n",
       "1714               0      1  2011            0            17       0.0   \n",
       "1715               0      1  2011            0            17       0.0   \n",
       "\n",
       "      units_sold  \n",
       "1711        33.0  \n",
       "1712        29.0  \n",
       "1713        25.0  \n",
       "1714        43.0  \n",
       "1715        21.0  "
      ]
     },
     "execution_count": 80,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_df['units_sold'] = np.exp(test_df['units_sold'])\n",
    "test_df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "bff61471",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "      <th>units_sold</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41477</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>134.6625</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>11</td>\n",
       "      <td>0.00</td>\n",
       "      <td>49.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1157</th>\n",
       "      <td>14455</td>\n",
       "      <td>9845</td>\n",
       "      <td>216425</td>\n",
       "      <td>117.5625</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>2011</td>\n",
       "      <td>0</td>\n",
       "      <td>14</td>\n",
       "      <td>14.25</td>\n",
       "      <td>118.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      record_ID  store_id  sku_id  total_price  base_price  is_featured_sku  \\\n",
       "2         41477      9164  216425     134.6625    134.6625                0   \n",
       "1157      14455      9845  216425     117.5625    131.8125                0   \n",
       "\n",
       "      is_display_sku  month  year  day_of_week  day_of_month  discount  \\\n",
       "2                  0      7  2011            0            11      0.00   \n",
       "1157               0      3  2011            0            14     14.25   \n",
       "\n",
       "      units_sold  \n",
       "2           49.0  \n",
       "1157       118.0  "
      ]
     },
     "execution_count": 81,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "demand_forecast = test_df[(df['store_id'] == store_id) & (df_processed['sku_id'] == sku_id)]\n",
    "demand_forecast.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "a2ea2858",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2        49.0\n",
       "1157    118.0\n",
       "Name: units_sold, dtype: float64"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "demand_forecast = demand_forecast['units_sold']\n",
    "demand_forecast.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "5923d6eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "demand_forecast = demand_forecast.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "77361d63",
   "metadata": {},
   "outputs": [],
   "source": [
    "safety_factor = calculate_safety_factor(desired_service_level, standard_deviation)\n",
    "reorder_point = calculate_reorder_point (demand_forecast, lead_time, safety_factor)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "52f18553",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(195.0, 28.0)"
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "reorder_point"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "ae4af37c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>record_ID</th>\n",
       "      <th>week</th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>211535</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>141.7875</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.276666</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>211511</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>129.6750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.890372</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>211602</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>141.0750</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.784190</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>211560</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>131.8125</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.564949</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>211580</td>\n",
       "      <td>2013-07-09</td>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>133.2375</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.110874</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        record_ID       week  store_id  sku_id  total_price  base_price  \\\n",
       "149378     211535 2013-07-09      9112  216425     141.7875    141.7875   \n",
       "149358     211511 2013-07-09      9092  216425     129.6750    129.6750   \n",
       "149420     211602 2013-07-09      9164  216425     141.0750    141.0750   \n",
       "149394     211560 2013-07-09      9132  216425     131.8125    131.8125   \n",
       "149406     211580 2013-07-09      9147  216425     133.2375    133.2375   \n",
       "\n",
       "        is_featured_sku  is_display_sku  units_sold  month  year  day_of_week  \\\n",
       "149378                0               0    4.276666      7  2013            1   \n",
       "149358                0               0    2.890372      7  2013            1   \n",
       "149420                0               0    3.784190      7  2013            1   \n",
       "149394                0               0    2.564949      7  2013            1   \n",
       "149406                0               0    4.110874      7  2013            1   \n",
       "\n",
       "        day_of_month  discount  \n",
       "149378             9       0.0  \n",
       "149358             9       0.0  \n",
       "149420             9       0.0  \n",
       "149394             9       0.0  \n",
       "149406             9       0.0  "
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Starting RNN (Recurrent Neural Network)\n",
    "df_nrr = df_processed\n",
    "df_nrr.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "9faeb680",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop unnecessary columns\n",
    "df_nrr = df_nrr.drop(columns=['record_ID', 'week'])  # Drop unnecessary columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "48a9174e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Normalize numerical features\n",
    "scaler = StandardScaler()\n",
    "df_nrr[['total_price', 'base_price']] = scaler.fit_transform(df_nrr[['total_price', 'base_price']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "191528b5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>store_id</th>\n",
       "      <th>sku_id</th>\n",
       "      <th>total_price</th>\n",
       "      <th>base_price</th>\n",
       "      <th>is_featured_sku</th>\n",
       "      <th>is_display_sku</th>\n",
       "      <th>units_sold</th>\n",
       "      <th>month</th>\n",
       "      <th>year</th>\n",
       "      <th>day_of_week</th>\n",
       "      <th>day_of_month</th>\n",
       "      <th>discount</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>149378</th>\n",
       "      <td>9112</td>\n",
       "      <td>216425</td>\n",
       "      <td>1.945837</td>\n",
       "      <td>1.802947</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.276666</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149358</th>\n",
       "      <td>9092</td>\n",
       "      <td>216425</td>\n",
       "      <td>0.528680</td>\n",
       "      <td>0.185577</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.890372</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149420</th>\n",
       "      <td>9164</td>\n",
       "      <td>216425</td>\n",
       "      <td>1.862475</td>\n",
       "      <td>1.707807</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3.784190</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149394</th>\n",
       "      <td>9132</td>\n",
       "      <td>216425</td>\n",
       "      <td>0.778766</td>\n",
       "      <td>0.470995</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.564949</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>149406</th>\n",
       "      <td>9147</td>\n",
       "      <td>216425</td>\n",
       "      <td>0.945490</td>\n",
       "      <td>0.661274</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4.110874</td>\n",
       "      <td>7</td>\n",
       "      <td>2013</td>\n",
       "      <td>1</td>\n",
       "      <td>9</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        store_id  sku_id  total_price  base_price  is_featured_sku  \\\n",
       "149378      9112  216425     1.945837    1.802947                0   \n",
       "149358      9092  216425     0.528680    0.185577                0   \n",
       "149420      9164  216425     1.862475    1.707807                0   \n",
       "149394      9132  216425     0.778766    0.470995                0   \n",
       "149406      9147  216425     0.945490    0.661274                0   \n",
       "\n",
       "        is_display_sku  units_sold  month  year  day_of_week  day_of_month  \\\n",
       "149378               0    4.276666      7  2013            1             9   \n",
       "149358               0    2.890372      7  2013            1             9   \n",
       "149420               0    3.784190      7  2013            1             9   \n",
       "149394               0    2.564949      7  2013            1             9   \n",
       "149406               0    4.110874      7  2013            1             9   \n",
       "\n",
       "        discount  \n",
       "149378       0.0  \n",
       "149358       0.0  \n",
       "149420       0.0  \n",
       "149394       0.0  \n",
       "149406       0.0  "
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_nrr.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "1b9e8a42",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into features (X) and target (y)\n",
    "X_nrr = df_nrr.drop(columns=['units_sold'])\n",
    "y_nrr = df_nrr['units_sold']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "a616bf64",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into training and testing sets\n",
    "X_nrr_train, X_nrr_test, y_nrr_train, y_nrr_test = train_test_split(X_nrr, y_nrr, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "c6f5bd1e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reshape input data for LSTM\n",
    "X_nrr_train = np.array(X_nrr_train).reshape(X_nrr_train.shape[0], X_nrr_train.shape[1], 1)\n",
    "X_nrr_test = np.array(X_nrr_test).reshape(X_nrr_test.shape[0], X_nrr_test.shape[1], 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "0abd8902",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "149378    4.276666\n",
       "149358    2.890372\n",
       "149420    3.784190\n",
       "149394    2.564949\n",
       "149406    4.110874\n",
       "Name: units_sold, dtype: float64"
      ]
     },
     "execution_count": 93,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_nrr.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "36a2b45c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#X_nrr.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "fc56c6f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the RNN model\n",
    "model = Sequential()\n",
    "model.add(LSTM(units=50, return_sequences=True, input_shape=(X_nrr_train.shape[1], 1)))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(LSTM(units=50, return_sequences=True))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(LSTM(units=50))\n",
    "model.add(Dropout(0.2))\n",
    "model.add(Dense(units=1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "1158495a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:At this time, the v2.11+ optimizer `tf.keras.optimizers.Adam` runs slowly on M1/M2 Macs, please use the legacy Keras optimizer instead, located at `tf.keras.optimizers.legacy.Adam`.\n"
     ]
    }
   ],
   "source": [
    "# Compile the model\n",
    "optimizer = Adam(learning_rate=0.001)\n",
    "model.compile(optimizer=optimizer, loss='mean_squared_error')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f371c7e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "215/215 [==============================] - 6s 13ms/step - loss: 1.0414 - val_loss: 0.5498\n",
      "Epoch 2/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.6522 - val_loss: 0.5360\n",
      "Epoch 3/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.6449 - val_loss: 0.5180\n",
      "Epoch 4/100\n",
      "215/215 [==============================] - 3s 13ms/step - loss: 0.6415 - val_loss: 0.5190\n",
      "Epoch 5/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.6281 - val_loss: 0.5120\n",
      "Epoch 6/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.6275 - val_loss: 0.5038\n",
      "Epoch 7/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.6172 - val_loss: 0.5048\n",
      "Epoch 8/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.6094 - val_loss: 0.4980\n",
      "Epoch 9/100\n",
      "215/215 [==============================] - 2s 10ms/step - loss: 0.6098 - val_loss: 0.5129\n",
      "Epoch 10/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5992 - val_loss: 0.4912\n",
      "Epoch 11/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5872 - val_loss: 0.4696\n",
      "Epoch 12/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.5755 - val_loss: 0.4677\n",
      "Epoch 13/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5719 - val_loss: 0.4657\n",
      "Epoch 14/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5701 - val_loss: 0.4633\n",
      "Epoch 15/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5572 - val_loss: 0.4586\n",
      "Epoch 16/100\n",
      "215/215 [==============================] - 2s 11ms/step - loss: 0.5560 - val_loss: 0.4679\n",
      "Epoch 17/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5458 - val_loss: 0.4848\n",
      "Epoch 18/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.5446 - val_loss: 0.4548\n",
      "Epoch 19/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5383 - val_loss: 0.4541\n",
      "Epoch 20/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.5364 - val_loss: 0.4550\n",
      "Epoch 21/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5389 - val_loss: 0.4571\n",
      "Epoch 22/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.5308 - val_loss: 0.4508\n",
      "Epoch 23/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5336 - val_loss: 0.4513\n",
      "Epoch 24/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5248 - val_loss: 0.4500\n",
      "Epoch 25/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5303 - val_loss: 0.4456\n",
      "Epoch 26/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5211 - val_loss: 0.4625\n",
      "Epoch 27/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.5219 - val_loss: 0.4472\n",
      "Epoch 28/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5264 - val_loss: 0.4402\n",
      "Epoch 29/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.5240 - val_loss: 0.4484\n",
      "Epoch 30/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5157 - val_loss: 0.4495\n",
      "Epoch 31/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5219 - val_loss: 0.4528\n",
      "Epoch 32/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5173 - val_loss: 0.4685\n",
      "Epoch 33/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.5141 - val_loss: 0.4378\n",
      "Epoch 34/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5146 - val_loss: 0.4405\n",
      "Epoch 35/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.5068 - val_loss: 0.4385\n",
      "Epoch 36/100\n",
      "215/215 [==============================] - 4s 17ms/step - loss: 0.5079 - val_loss: 0.4368\n",
      "Epoch 37/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.5027 - val_loss: 0.4378\n",
      "Epoch 38/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.5009 - val_loss: 0.4404\n",
      "Epoch 39/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.5022 - val_loss: 0.4389\n",
      "Epoch 40/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.4944 - val_loss: 0.4372\n",
      "Epoch 41/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4935 - val_loss: 0.4305\n",
      "Epoch 42/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4960 - val_loss: 0.4290\n",
      "Epoch 43/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.4940 - val_loss: 0.4420\n",
      "Epoch 44/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4900 - val_loss: 0.4380\n",
      "Epoch 45/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4921 - val_loss: 0.4460\n",
      "Epoch 46/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.4835 - val_loss: 0.4326\n",
      "Epoch 47/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4841 - val_loss: 0.4532\n",
      "Epoch 48/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.4793 - val_loss: 0.4389\n",
      "Epoch 49/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4812 - val_loss: 0.4246\n",
      "Epoch 50/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4781 - val_loss: 0.4370\n",
      "Epoch 51/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4753 - val_loss: 0.4209\n",
      "Epoch 52/100\n",
      "215/215 [==============================] - 2s 9ms/step - loss: 0.4639 - val_loss: 0.4257\n",
      "Epoch 53/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4660 - val_loss: 0.4172\n",
      "Epoch 54/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4662 - val_loss: 0.4145\n",
      "Epoch 55/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4619 - val_loss: 0.4165\n",
      "Epoch 56/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.4713 - val_loss: 0.4242\n",
      "Epoch 57/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4702 - val_loss: 0.4201\n",
      "Epoch 58/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4578 - val_loss: 0.4258\n",
      "Epoch 59/100\n",
      "215/215 [==============================] - 2s 7ms/step - loss: 0.4613 - val_loss: 0.4161\n",
      "Epoch 60/100\n",
      "215/215 [==============================] - 2s 8ms/step - loss: 0.4619 - val_loss: 0.4199\n",
      "Epoch 61/100\n",
      " 73/215 [=========>....................] - ETA: 1s - loss: 0.4624"
     ]
    }
   ],
   "source": [
    "# Train the model\n",
    "model.fit(X_nrr_train, y_nrr_train, epochs=100, batch_size=32, validation_data=(X_nrr_test, y_nrr_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab45ac25",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate the model\n",
    "y_nrr_pred = model.predict(X_nrr_test).flatten()\n",
    "rmse_nrr = np.sqrt(mean_squared_error(y_nrr_test, y_nrr_pred))\n",
    "mape_nrr = np.mean(np.abs((y_nrr_test - y_nrr_pred) / y_nrr_test)) * 100\n",
    "loss_nrr = model.evaluate(X_nrr_test, y_nrr_test)\n",
    "\n",
    "print(\"Test Loss:\", loss_nrr)\n",
    "print(\"Root Mean Squared Error (RMSE):\", rmse_nrr)\n",
    "print(\"Mean Absolute Percentage Error (MAPE):\", mape_nrr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d218bf8",
   "metadata": {},
   "outputs": [],
   "source": [
    "plot_predictions(y_nrr_test, y_nrr_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a8a73593",
   "metadata": {},
   "outputs": [],
   "source": [
    "rmse, rmse_nrr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53059f25",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find RMSE\n",
    "mse_nrr = mean_squared_error(y_nrr_test, y_nrr_pred)\n",
    "rmse_nrr = np.sqrt(mse_nrr)\n",
    "print(\"RMSE:\",rmse_nrr)\n",
    "print(\"MSE:\",mse_nrr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b4ec09d",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Evaluate accuracy using MAPE\n",
    "y_nrr_true = np.array(y_nrr_test)\n",
    "sumvalue=np.sum(y_nrr_true)\n",
    "mape_nrr=np.sum(np.abs((y_nrr_true - y_nrr_pred)))/sumvalue*100\n",
    "accuracy_nrr=100-mape_nrr\n",
    "print('Accuracy:', round(accuracy_nrr,2),'%.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2ef4873c",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_nrr_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b82ed06",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_nrr_true"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "111b6ae7",
   "metadata": {},
   "outputs": [],
   "source": [
    "comp_nrr = pd.DataFrame(data=[y_nrr_true,y_nrr_pred]).T\n",
    "comp_nrr.columns=['y_nrr_test','y_nrr_pred']\n",
    "comp_nrr"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2e4fd8b0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# SARIMAX"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8495f46b",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_processed.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d6229e9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_processed.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94a9fe3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9366df02",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_sarimax.shape\n",
    "y_train_sarimax.shape\n",
    "X_test_sarimax.shape\n",
    "y_test_sarimax.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb277826",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_sarimax.reset_index(drop=True, inplace=True)\n",
    "y_test_sarimax.reset_index(drop=True, inplace=True)\n",
    "X_train_sarimax.reset_index(drop=True, inplace=True)\n",
    "y_train_sarimax.reset_index(drop=True, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92993752",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_sarimax.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "87d52639",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_sarimax = X_train_sarimax.loc[y_train_sarimax.index]  # Align indices with y_train\n",
    "y_train_sarimax = y_train_sarimax.loc[X_train_sarimax.index]  # Align indices with X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0339849e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define and fit the SARIMAX model\n",
    "model_sarimax = SARIMAX(endog=y_train_sarimax, exog=X_train_sarimax, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))\n",
    "results_sarimax = model_sarimax.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "93790b35",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions\n",
    "predictions = results_sarimax.predict(start=len(y_train_sarimax), end=len(y_train_sarimax)+len(y_test_sarimax)-1, exog=X_test_sarimax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0e1b7ed3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate RMSE\n",
    "rmse_sarimax = np.sqrt(mean_squared_error(y_test_sarimax, predictions))\n",
    "print(\"Root Mean Squared Error (RMSE):\", rmse_sarimax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb920d13",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "558f9b8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Evaluate accuracy using MAPE\n",
    "y_true_sarimax = np.array(y_test_sarimax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e271e9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "sumvalue=np.sum(y_true_sarimax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9e2135e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88815e6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "predictions = predictions.values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9548bf95",
   "metadata": {},
   "outputs": [],
   "source": [
    "mape_sarimax=np.sum(np.abs((y_true_sarimax - predictions)))/sumvalue*100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff174834",
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy_sarimax=100-mape_nrr\n",
    "print('Accuracy:', round(accuracy_sarimax,2),'%.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "235d9735",
   "metadata": {},
   "outputs": [],
   "source": [
    "rmse, rmse_nrr, rmse_sarimax"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d038cfc1",
   "metadata": {},
   "outputs": [],
   "source": [
    "np.exp(rmse), np.exp(rmse_nrr), np.exp(rmse_sarimax)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6faff90",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Holt Winters\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
