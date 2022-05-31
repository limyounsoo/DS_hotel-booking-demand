import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import warnings
import plotly.express as px
warnings.filterwarnings(action='ignore')
#for checking if a date is a holiday
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

## Data exploration
# Read dataset
data = pd.read_csv('hotel_bookings.csv')

# Show the first 5 rows of Data
print(f'First 5 rows of data\n{data.head()}', end='\n\n')

# Numbers of row and column
print(f'Data shape\n{data.shape}')

# Data information
print(f'Data information{data.info()}', end='\n\n')

# Plot the heatmap to see correlation with columns
fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(data.corr(), annot=True, ax=ax)
plt.show()
print(data.corr()['adr'].sort_values(ascending=False), end='\n\n')

# Showing target column('adr')
sns.displot(data, x="adr")
plt.xlim(0, 500)
plt.show()

# Showing adr in each hotel type
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.distplot(data[data['hotel']=="Resort Hotel"].adr,ax=ax[0])
ax[0].set_title('adr in Resort Hotel')
sns.distplot(data[data['hotel']=="City Hotel"].adr,ax=ax[1])
ax[1].set_title('adr in City Hotel')
plt.show()

# Showing adr in canceled or not
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.distplot(data[data['is_canceled']==0].adr,ax=ax[0])
ax[0].set_title('adr in not canceled')
sns.distplot(data[data['is_canceled']==1].adr,ax=ax[1])
ax[1].set_title('adr in canceled')
plt.show()

## Data preprocessing
# Cleaning data
# Copy the dataset to save original dataset
df = data.copy()

# Find the missing value, show the total null values for each column and sort it in descending order
print(f'Missing value in dataset\n{df.isnull().sum().sort_values(ascending=False)[:10]}', end='\n\n')

# If no id of agent or company is null, just replace it with 0
df[['agent','company']] = df[['agent','company']].fillna(0.0)

# For the missing values in the country column, replace it with mode (value that appears most often)
df['country'].fillna(data.country.mode().to_string(), inplace=True)

# for missing children value, replace it with rounded mean value
df['children'].fillna(round(data.children.mean()), inplace=True)

# Drop Rows where there is no adult, baby and child(zero humans in room)
df = df.drop(df[(df.adults+df.babies+df.children)==0].index)

# Convert datatype of these columns from float to integer
df[['children', 'company', 'agent']] = df[['children', 'company', 'agent']].astype('int64')

# Find outliers in 'adr'
df_sorted_by_value = df['adr'].sort_values(ascending=False)
print(df_sorted_by_value, end='\n\n')

# Drop too much or too small values in 'adr'
df = df.drop(index = df[df['adr'] > 5000].index)
df = df.drop(index = df[df['adr'] <= 0].index)

# Find outliers in 'babies'
df_sorted_by_value = df['babies'].sort_values(ascending=False)
print(df_sorted_by_value, end='\n\n')

# Drop too much babies in 'babies'
df = df.drop(index = df[df['babies'] > 3].index)

# Find outliers in 'children'
df_sorted_by_value = df['children'].sort_values(ascending=False)
print(df_sorted_by_value, end='\n\n')

# Drop too much children in 'children'
df = df.drop(index = df[df['children'] > 3].index)

# Repalce SC & Undefined to one category(no meal package)
df.loc[(df.meal == 'SC')| (df.meal == 'Undefined'), 'meal'] = 'SC_Undefined'

# Feature engineering for better prediction
# Change month(str) to int
df['month_number'] = df.arrival_date_month.apply(lambda x: datetime.datetime.strptime(x,'%B').month )
# Edit dataframe for 'to_datetime' function
temp = df[['arrival_date_year','month_number','arrival_date_day_of_month']]
temp.rename({'arrival_date_year' : 'year' ,'month_number':'month' , 'arrival_date_day_of_month' : 'day'  } ,axis = 1 , inplace = True)
# Create arrival_date
df['arrival_date'] = pd.to_datetime(temp)

# Create adr_bins column for adr classification
'''
    0 : 0 to 100
    1 : 100 to 200
    2 : 200 to 300
    3 : 300 to 400
    4 : 400+
'''
df = df.assign(adr_bins = pd.cut(df["adr"], [0, 100, 200, 300 ,400, 1000], 
                        labels=[0, 1, 2, 3, 4]))

# A binary column, represents if the transaction was booked in a resort or city hotel.
df['is_resort'] =  df.hotel.map(lambda x : 1 if x == 'Resort Hotel' else 0  )
# For retrieving columns of number of nights and number of guests in total.
df['total_nights'] = df.stays_in_weekend_nights + df.stays_in_week_nights
df['total_guests'] = df.adults + df.children + df.babies
# A binary column, represents if the arrival day is a holiday, and if the vacation is during the weekend.
cal = calendar()
holidays = cal.holidays(start = min(df.arrival_date) ,end =  max(df.arrival_date))
df['is_holiday'] = df['arrival_date'].isin(holidays)
df['is_weekend'] = df.stays_in_weekend_nights > 0 

print(f'After feature engineering : {df.info()}', end='\n\n')

# Categorical data encoding
def transform(dataframe): 
    le = LabelEncoder()
    categorical_features = list(dataframe.columns[dataframe.dtypes == object])
    dataframe[categorical_features]=dataframe[categorical_features].apply(lambda x: le.fit_transform(x))
    return dataframe

df = transform(df)

# Modeling - regression
# Make new dataframe with adding new columns using in regression
df_reg = df[['is_resort', 'month_number', 'lead_time', 'arrival_date_week_number', 'distribution_channel', 'customer_type',
                    'market_segment', 'is_holiday' , 'previous_cancellations','previous_bookings_not_canceled'
                  ,'adults', 'children', 'babies', 'meal' ,'is_repeated_guest' , 'required_car_parking_spaces'
                , 'reserved_room_type' ,'total_nights','is_weekend', 'adr']]
print(f'Data frame for regression \n{df_reg.describe()}', end='\n\n')
print(df['adr'].info(), end='\n\n')
X_reg = df_reg.drop('adr', axis=1)
y_reg = df_reg['adr']
scaler = ['StandardScaler()', 'MinMaxScaler()', 'RobustScaler()']
regModel = ['LinearRegression()', 'DecisionTreeRegressor(max_depth=17)', 'RandomForestRegressor()']

# Train model and find best combination of regressor and scaler
def best_comb_reg(scaler, model):
    best_score = 0
    
    for element in scaler:
        scaler = eval(element)
        scaled = scaler.fit_transform(X_reg)
        x_train, x_test, y_train, y_test = train_test_split(scaled, y_reg, test_size = 0.2, random_state=42) # Using random_state to fixing random rate
        for element2 in model:
            regressor = eval(element2)
            regressor = regressor.fit(x_train,y_train)
            test_score = regressor.score(x_test, y_test)
            print(f'Using {regressor} in {scaler} score : {test_score}')

            if test_score > best_score:
                best_score = test_score
                best_scaler = element
                best_model = regressor
                x_bTrain = x_train
                x_bTest = x_test
                y_Btrain = y_train
                y_Btest = y_test
        print('')
    return best_score, best_scaler, best_model, x_bTrain, x_bTest, y_Btrain, y_Btest
                
best_score_reg, best_scaler_reg, best_model_reg, best_x_train_reg, best_x_test_reg, best_y_train_reg, best_y_test_reg = best_comb_reg(scaler, regModel)
print('')
print(f'Best scaler, model, score in regression : {best_scaler_reg, best_score_reg, best_model_reg}', end='\n\n')

# Modeling - classification
# Make new dataframe with adding new columns using in classification
df_clf = df[['is_resort', 'month_number', 'lead_time', 'arrival_date_week_number', 'distribution_channel', 'customer_type',
                    'market_segment', 'is_holiday' , 'previous_cancellations','previous_bookings_not_canceled'
                  ,'adults', 'children', 'babies', 'meal' ,'is_repeated_guest' , 'required_car_parking_spaces'
                , 'reserved_room_type' ,'total_nights','is_weekend', 'adr_bins']]
print(df_clf.describe(include='all'), end='\n\n')
print(df['adr_bins'].value_counts(), end='\n\n')
X_clf = df_clf.drop('adr_bins', axis=1)
y_clf = df_clf['adr_bins']
clfModel = ['DecisionTreeClassifier()', 'KNeighborsClassifier()', 'RandomForestClassifier()', 'XGBClassifier(learning_rate = 0.1, max_depth = 5, n_estimators = 500)']

# Train model and find best combination of classifier and scaler
def best_comb_clf(scaler, model):
    best_acc = 0
    
    for element in scaler:
        scaler = eval(element)
        scaled = scaler.fit_transform(X_clf)
        x_train, x_test, y_train, y_test = train_test_split(scaled, y_clf, test_size = 0.2, random_state=42) #Using random_state to fixing random rate
        for element2 in model:
            classifier = eval(element2)
            classifier = classifier.fit(x_train,y_train)
            
            y_pred = classifier.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            print(f'Using {classifier} in {scaler} score : {acc}')

            if acc > best_acc:
                best_acc = acc
                best_scaler = element
                best_model = classifier
                x_bTest = x_test
                y_bTest = y_test
                y_bPred = y_pred
        print('')
    return best_acc, best_scaler, best_model, x_bTest, y_bTest, y_bPred

best_score_clf, best_scaler_clf, best_model_clf, best_x_test_clf, best_y_test_clf, best_y_pred_clf = best_comb_clf(scaler, clfModel)
print('')
print(f'Best scaler, model, score in classificiation : {best_scaler_clf, best_score_clf, best_model_clf}', end='\n\n')

# Evaluation - regression
# K-fold validation for best model in regression
kfold = KFold(n_splits=5, shuffle=True, random_state=7)
result = cross_val_score(best_model_reg, X_reg, y_reg, cv = kfold)
print(f"K-fold validation with {best_model_reg} : {result}")
print(f"Average in K-fold validation: {result.mean()}", end='\n\n')

# Evaluation - classification
# K-fold validation for best model in classification
kfold = KFold(n_splits=5, shuffle=True, random_state=7)
result = cross_val_score(best_model_clf, X_clf, y_clf, cv = kfold)
print(f"K-fold validation with {best_model_clf} : {result}")
print(f"Average in K-fold validation: {result.mean()}", end='\n\n')

# Confusion matrix and classification report with best model in classification
conf = confusion_matrix(best_y_test_clf, best_y_pred_clf)
clf_report = classification_report(best_y_test_clf, best_y_pred_clf)

print(f"Confusion Matrix : \n{conf}")
'''
    Precision : The ratio of the number of samples (TP) actually belonging to the positive class among the samples (TP+FP) predicted to be positive class.
    The higher the better. It is easy to think of it as the ratio of the 'correct prediction value' based on the 'prediction value'.

    Recall : The ratio of the number of samples (TP) in the actual positive class (TP+FN) predicted to be positive class.
    As expected, the higher the better. It is easy to think of it as the ratio of the 'correct prediction value' based on the 'actual value'.

    Accuracy : The percentage of the total number of samples predicted correctly. The higher the better.

    F1 score : The weighted harmonic mean of precision and recall is called the F-score.

    Support : Actual number of samples for each label.

    Macro avg : Averaging the means(simple average), The same weights are given for each class. The imbalance in the number of samples is not considered.

    Weighted abg : Calculating a weighted average of the number of samples belonging to each class. Consider an imbalance in the number of samples.
'''
print(f"Classification Report : \n{clf_report}")