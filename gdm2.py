import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import accuracy_score

'''This model uses data with capped outliers'''

#importing the csv file
gdm_full = pd.read_csv("C:\\Users\\USER\\PycharmProjects\\pythonProject\\Datasets\\Gestational Diabetic Dat Set.csv")

'''gdm_train = pd.read_csv("C:\\Users\\USER\\PycharmProjects\\pythonProject\\Datasets\\GDM_train.csv")
gdm_test = pd.read_csv("C:\\Users\\USER\\PycharmProjects\\pythonProject\\Datasets\\GDM_test.csv")'''
#print(gdm_train.head())

#data preprocesing on train set
#print(gdm_train.isna().sum())
print(gdm_full.columns)

#plotting the distribution of the columns with missing values
'''fig, ax = plt.subplots(2,2)

ax[0,0].hist(gdm_full['BMI'])
ax[0,1].hist(gdm_full['HDL'])
ax[1,0].hist(gdm_full['Sys BP'])
ax[1,1].hist(gdm_full['OGTT'])

ax[0,0].set_title('BMI')
ax[0,1].set_title('HDL')
ax[1,0].set_title('Sys BP')
ax[1,1].set_title("OGTT")

# Set the main title for the whole figure
fig.suptitle('Distribution of BMI, HDL, Sys BP, OGTT', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()'''


'''print(gdm_full.isna().sum())
print("the shape of the gdm full is:",gdm_full.shape)'''

#handling missing values on train set
#BMI: impute mean
mean_BMI = gdm_full["BMI"].mean()
gdm_full["BMI"].fillna(mean_BMI, inplace= True)

#HDL: impute median (left skewed)
med_HDL = gdm_full["HDL"].median()
gdm_full['HDL'].fillna(med_HDL, inplace=True)

#Sys BP: impute mean
mean_sysBP = gdm_full["Sys BP"].mean()
gdm_full["Sys BP"].fillna(mean_sysBP, inplace =True)

#OGTT: impute median(right skewed)
med_OGTT = gdm_full["OGTT"].median()
gdm_full["OGTT"].fillna(med_OGTT, inplace= True)

#confirming no missing values
print(gdm_full.isna().sum())

#cheking data types
#print(gdm_full.info()) # all data is numeric

#correlation analysis
#print(gdm_full.columns)
columns_to_corr = [
    'Age',
    'No of Pregnancy',
    'Gestation in previous Pregnancy',
    'BMI',
    'HDL',
    'unexplained prenetal loss',
    'Large Child or Birth Default',
    'PCOS',
    'Sys BP',
    'Dia BP',
    'OGTT',
    'Hemoglobin',
    'Sedentary Lifestyle',
    'Prediabetes'
]

# Calculate Spearman correlation of each column with 'Class Label(GDM /Non GDM)'
'''correlations = {col: gdm_full[col].corr(gdm_full['Class Label(GDM /Non GDM)'], method='spearman') for col in columns_to_corr}

# Create a DataFrame for plotting
corr_df = pd.DataFrame(list(correlations.items()), columns=['Column', 'Spearman Correlation'])

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(corr_df['Column'], corr_df['Spearman Correlation'], color='#90EE90')
plt.xlabel('Spearman Correlation')
plt.title('Spearman Correlation of Columns with Class Label(GDM /Non GDM)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print(corr_df)'''

# checking outliers
'''cols_1 = gdm_train[['Age', 'No of Pregnancy',
       'Gestation in previous Pregnancy', 'BMI', 'HDL']]
cols_2 = gdm_train[['Family History',
       'unexplained prenetal loss', 'Large Child or Birth Default', 'PCOS']]
cols_3 = gdm_train[['Sys BP', 'Dia BP', 'OGTT', 'Hemoglobin', 'Sedentary Lifestyle',
       'Prediabetes']]
cols_1.plot(kind = "box" , subplots = True, layout =(1,5), figsize=(5,10), sharey = False, sharex= False)
plt.suptitle("Boxplots of Age, No of Pregnancy, Gestation in Previous Pregnancy, BMI, and HDL")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
cols_2.plot(kind = "box" , subplots = True, layout =(1,5), figsize=(5,10), sharey = False, sharex= False)
plt.suptitle('Boxplots of Family History,unexplained prenetal loss, Large Child or Birth Default and PCOS')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
cols_3.plot(kind = "box" , subplots = True, layout =(1,6), figsize=(5,10), sharey = False, sharex= False)
plt.suptitle('Boxplots for Sys BP, Dia BP, OGTT, Hemoglobin, Sedentary Lifestyle and Prediabetes')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()'''

columns_with_outliers = ['BMI', 'HDL', 'Sys BP', 'Dia BP', 'OGTT', 'Hemoglobin']

#counting outliers in a column
'''def count_outliers(column):
    Q1 = gdm_full[column].quantile(0.25)
    Q3 = gdm_full[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return gdm_full[(gdm_full[column] < lower_bound) | (gdm_full[column] > upper_bound)].shape[0]

# Calculate outliers count for each column
outliers_count = {col: count_outliers(col) for col in columns_with_outliers}

print(outliers_count)'''
print(gdm_full.shape)

#dealing with outliers by capping outliers
def cap_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    return df

gdm_full_capped = cap_outliers(gdm_full, columns_with_outliers)

# Verify the outliers are capped
'''print(gdm_full_capped[columns_with_outliers].describe())
print(gdm_full_capped.head())'''


# counting individuals with above normal ranges
# BMI
'''print(gdm_full[gdm_full["BMI"]< 18.5 ]) # 35 people with less than usual BMI
print(gdm_full[gdm_full["BMI"] > 24.9]) # 1751 people with above the normal range of BMI
'''
#HDL
'''print(gdm_full[gdm_full["HDL"]< 50 ]) # 1307 people with low/ poor HDL'''

#SYS BP
'''print(gdm_full[gdm_full["Sys BP"] < 120]) # 342 people with normal Sys BP
print(gdm_full[(gdm_full["Sys BP"]> 119) & ( gdm_full['Sys BP'] <= 129)]) # 480 people with elevated Sys BP
print(gdm_full[(gdm_full["Sys BP"]> 129) & ( gdm_full['Sys BP'] <= 139)]) # 273 people with hypertension Stage 1
print(gdm_full[gdm_full["Sys BP"]> 139]) # 725 people with elevated Sys BP'''


#Dia BP
'''print(gdm_full[gdm_full["Dia BP"] < 80]) # 1303 people with normal Dia BP
print("===================")
print(gdm_full[(gdm_full["Dia BP"]> 79) & ( gdm_full['Dia BP'] < 90)]) # 1522 people with elevated Dia BP excluding those with systolic pressure between 130 and 139
print("===================")
print(gdm_full[(gdm_full["Dia BP"]> 79) & ( gdm_full['Dia BP'] < 90) & (gdm_full["Sys BP"] > 129 )& (gdm_full["Sys BP"] <140)]) # 226 people with Hypertension stage 1 if systolic pressure is 130-139 mmHg
print(gdm_full[gdm_full["Dia BP"] > 90]) # 635 people with Hyperstension stage 2 (Dia BP)'''


#OGTT using FBG
'''print(gdm_full[gdm_full["OGTT"] < 100]) # 82 people with normal using fasting blood glucose
print("===================")
print(gdm_full[(gdm_full["OGTT"]>=100) & (gdm_full['OGTT'] <= 125)]) # 264 people with prediabetes using fasting blood glucose
print("===================")
print(gdm_full[gdm_full["OGTT"] > 125]) # 2666  people with Diabetes  using fasting blood glucose
'''
#OGTT ( 2hour post glucose drink)
'''print(gdm_full[gdm_full["OGTT"] < 140]) # 668 people with normal using fasting blood glucose
print("===================")
print(gdm_full[(gdm_full["OGTT"]>=140) & ( gdm_full['OGTT'] <= 199)]) # 1611 people with prediabetes using fasting blood glucose
print("===================")
print(gdm_full[gdm_full["OGTT"] > 199]) # 733  people with Diabetes  using fasting blood glucose
'''

#splitting the data into feature and target variables
X = gdm_full_capped.drop(['Class Label(GDM /Non GDM)', 'Case Number'], axis=1)
y = gdm_full_capped['Class Label(GDM /Non GDM)']


#spliting the data into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1999)

# standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled features back to a DataFrame
X_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
print(X_scaled_df)

# Stochastic Gradient boosting
sgbt_classifier = GradientBoostingRegressor(max_depth=1, subsample=0.8, max_features= 0.2, n_estimators=300, random_state= 1999)
sgbt_classifier.fit(X_train_scaled,y_train)
#predict
y_pred = sgbt_classifier.predict(X_test_scaled)

# evaluating  the model
sgbt_mse = MSE(y_test,y_pred)
print(f'SGBT Mean Squared Error: {sgbt_mse}')
#RMSE test
sgbt_rmse = sgbt_mse ** (1/2)
print('SGBT RMSE: {:.2f}'.format(sgbt_rmse))
# calculating the accuracy score
'''sgbt_accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy: {sgbt_accuracy}')'''

#print(gdm_full.isna().sum())
