# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import timeit

# Importing the dataset
dataset = pd.read_csv('train.csv')
#dataset = dataset[0:5000]
X = dataset.iloc[:, 1:127].values
data = dataset.iloc[:, 1:].values
dataframe = pd.DataFrame(data, columns = dataset.columns[1:])
y = dataset.iloc[:, -1].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, (11,14,16,28,33,34,35,36,37,46,51,60,68)])
X[:, (11,14,16,28,33,34,35,36,37,46,51,60,68)] = imputer.transform(X[:, (11,14,16,28,33,34,35,36,37,46,51,60,68)])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
produxtInfo2 = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 1] = produxtInfo2
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
    
X_column_label = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'B1', 'B2', 'C1',
       'C2', 'C3', 'C4', 'D1', 'D2', 'D3', 'D4', 'E1']
for i in range(len(dataframe.columns)-1):
    if(i == 1):
        continue
    else:
        X_column_label.append(dataframe.columns[i])
X_dataframe = pd.DataFrame(X, columns = X_column_label)

data = X_dataframe


X = data.iloc[:, :-1].values
#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)
for i in range(len(y)):
    y[i] = y[i]-1


########################################################################################
########################################################################################
########################################################################################
######################################################################################
######################################################################################
######################################################################################
###################################################################################

#######################################################################
#Building optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones(shape = (len(X), 1)).astype(int), values = X, axis = 1)

b = []
for i in range(len(X[1])):
    b.append(i)
    
display_p = []
display_length = []
display_index = []
display_number = []
display_r2 = []
display_adj_r2 = []

accuracy_percent = []
training_time = []
testing_time = []

########################################## p = 0.05 #########################################################
List_of_column_no_removed = []
maxx = 1.0
while( maxx > 0.05):
#ij = 0
#while(ij < 0.01):
    X_opt = X[: , b]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    summary = regressor_OLS.summary()
    summary_csv = summary.as_csv()
    
    summary_rsq_text = summary_csv[106:116]
    summary_rsq_value = summary_csv[129:134]

    summary_adj_rsq_text = summary_csv[175:190] 
    summary_adj_rsq_value = summary_csv[198:203]
    
    summary_csv =summary_csv[690: -224]
    
    print()
    print(summary_rsq_text + ' :' + " " + summary_rsq_value)
    print(summary_adj_rsq_text + ' :' + " " + summary_adj_rsq_value)
    
    text_file = open("Output.csv", "w")
    text_file.write(summary_csv)
    text_file.close()
    df = dataset = pd.read_csv('Output.csv')
    p = df.iloc[:, 4].values
    index = df.iloc[:, 0].values
    
    dframe = pd.DataFrame(index = index)
    dframe['p'] = p
    
    dframe['p'] = p
    maxx = dframe['p'].max()
    
    imax = 0
    for i in range(len(df)):
        if(dframe.iloc[i]['p'] == maxx):
            imax = i
    
    print('p: ', maxx, 'length: ', len(b)-1, '\nRemovable index: ', imax)
    
   
    if(maxx > 0.05):
        print('Removing Number: ', b[imax])
        print('#################################################\n')
              
        
        display_p.append(maxx)
        display_length.append(len(b)-1)
        display_index.append(imax)
        display_r2.append(summary_rsq_value)
        display_adj_r2.append(summary_adj_rsq_value)
        List_of_column_no_removed.append(b[imax])
        b.remove(b[imax])
        
              
        d = pd.DataFrame()
        d['length'] = display_length
        d['index'] = display_index
        d['number'] = List_of_column_no_removed
        d['p value'] = display_p
        d[summary_adj_rsq_text] = display_adj_r2
        d[summary_rsq_text] = display_r2
#        d["training time"] = training_time
#        d["testing time"] = testing_time 
#        d["accuracy"] = accuracy_percent
        
        save = d.to_csv(sep=',')
        text_file = open("train_file.csv", "w")
        text_file.write(save)
        text_file.close()
        
        df = pd.read_csv('train_file.csv')

        
        s=[]
        number = []
        b = []
        for i in range(len(X[1])):
            number.append(i)
            b.append(i)
           
        num = list(df['number'])
        for i in range(len(df)):
        #    if (m == df[summary_adj_rsq_text][i]):
        #        s.append(i)
            if(num[i] in number):
                b.remove(num[i])
      
#########################################################################################################################        
#########################################################################################################################
#########################################################################################################################
##########################################      0.01      ###############################################################        
#########################################################################################################################
#########################################################################################################################        
#########################################################################################################################
#########################################################################################################################  
                
########################################## p = 0.01 #########################################################

while( maxx > 0.01):
#ij = 0
#while(ij < 0.01):
    X_opt = X[: , b]
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    summary = regressor_OLS.summary()
    summary_csv = summary.as_csv()
    
    summary_rsq_text = summary_csv[106:116]
    summary_rsq_value = summary_csv[129:134]

    summary_adj_rsq_text = summary_csv[175:190] 
    summary_adj_rsq_value = summary_csv[198:203]
    
    summary_csv =summary_csv[690: -224]
    
    print()
    print(summary_rsq_text + ' :' + " " + summary_rsq_value)
    print(summary_adj_rsq_text + ' :' + " " + summary_adj_rsq_value)
    
    text_file = open("Output.csv", "w")
    text_file.write(summary_csv)
    text_file.close()
    df = dataset = pd.read_csv('Output.csv')
    p = df.iloc[:, 4].values
    index = df.iloc[:, 0].values
    
    dframe = pd.DataFrame(index = index)
    dframe['p'] = p
    
    dframe['p'] = p
    maxx = dframe['p'].max()
    
    imax = 0
    for i in range(len(df)):
        if(dframe.iloc[i]['p'] == maxx):
            imax = i
    
    print('p: ', maxx, 'length: ', len(b)-1, '\nRemovable index: ', imax)
    
   
    if(maxx > 0.01):
        print('Removing Number: ', b[imax])
        print('#################################################\n')
              
        
        display_p.append(maxx)
        display_length.append(len(b)-1)
        display_index.append(imax)
        display_r2.append(summary_rsq_value)
        display_adj_r2.append(summary_adj_rsq_value)
        List_of_column_no_removed.append(b[imax])
        b.remove(b[imax])
        
              
        d = pd.DataFrame()
        d['length'] = display_length
        d['index'] = display_index
        d['number'] = List_of_column_no_removed
        d['p value'] = display_p
        d[summary_adj_rsq_text] = display_adj_r2
        d[summary_rsq_text] = display_r2
#        d["training time"] = training_time
#        d["testing time"] = testing_time 
#        d["accuracy"] = accuracy_percent
        
        save = d.to_csv(sep=',')
        text_file = open("train_file.csv", "w")
        text_file.write(save)
        text_file.close()
        
        df = pd.read_csv('train_file.csv')

        
        s=[]
        number = []
        b = []
        for i in range(len(X[1])):
            number.append(i)
            b.append(i)
           
        num = list(df['number'])
        for i in range(len(df)):
        #    if (m == df[summary_adj_rsq_text][i]):
        #        s.append(i)
            if(num[i] in number):
                b.remove(num[i])
                      
                
#########################################################################################################################
#########################################################################################################################                                
#########################################################################################################################
#########################################################################################################################                  
#########################################################################################################################
                
#Building optimal model using Backward Elimination
X = X[:, b]
X = X[:, 1:]

#########################################################################################################################                                
#########################################################################################################################
#########################################################################################################################                  
#########################################################################################################################

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#####################################################################################
# After selection of important features
#######################################################################################

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier( n_estimators = 10, criterion = 'entropy', random_state = 0, n_jobs = -1)


temp = []
for i in range(10):
    start = timeit.default_timer()
    
    classifier.fit(X_train, y_train)
    
    stop = timeit.default_timer()
    After_time_001 = stop - start
    temp.append(After_time_001)
    
After_time_001 = sum(temp)/10

temp = []    
# Predicting the Test set results
for i in range(10):
    start = timeit.default_timer()
    
    y_pred = classifier.predict(X_test)
    
    stop = timeit.default_timer()
    After_time_y_pred_001 = stop - start
    temp.append(After_time_y_pred_001)
After_time_y_pred_001 = sum(temp)/10


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy=0
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if(i==j):
            accuracy += cm[i][j]
accuracy = (accuracy/len(X_test)) * 100
After_accuracy_001 = accuracy


#Applying kfold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv=10, n_jobs = -1)

#accuracy_by_10fold_cv = (sum(accuracies) / 10)
accuracy_by_10fold_cv = accuracies.mean() * 100
std = accuracies.std() * 100
accuracy = accuracy_by_10fold_cv
After_accuracy_001 = accuracy

print('\n\t'+'Train\t'  + str(After_time_001) + '\t' + str(After_accuracy_001) + ' Accuracy')
print('\n\t'+'Test\t'  + str(After_time_y_pred_001) + '\t' + str(After_accuracy_001) + ' Accuracy')
print('\n\t'+ 'b = ', len(b))
training_time.append(After_time_001)
testing_time.append(After_time_y_pred_001)
accuracy_percent.append(After_accuracy_001)

d = pd.DataFrame()

d["training time"] = training_time
d["testing time"] = testing_time 
d["accuracy"] = accuracy_percent

save = d.to_csv(sep=',')
text_file = open("train_file.csv", "w")
text_file.write(save)
text_file.close()

###################################################################################################################

