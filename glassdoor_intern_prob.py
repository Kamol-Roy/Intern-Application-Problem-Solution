import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


filename=r"ozan_p_pApply_intern_challenge_03_20_min.csv"


df=pd.read_csv(filename,parse_dates=['search_date_pacific'])

#Adding a user category column based on mgoc_id

unique_mgoc_ids=df['mgoc_id'].unique()
unique_mgoc_ids.sort()
mapping={v:k for k,v in enumerate(unique_mgoc_ids)}
df['user_cat']=df.apply(lambda x: mapping[x.mgoc_id], axis=1 )


#visualize the data 

df_scatter=df[list(df.columns)[:8]][:1000]
       
fig=plt.figure(figsize=(12,12))
axes = plt.subplot2grid((3,1), (0,0), rowspan=3, colspan=1)
pd.plotting.scatter_matrix(df_scatter, alpha=0.8,ax=axes)
plt.tight_layout()
#fig.savefig(r"scatter_mat.png",dpi=500)
plt.show()

del df_scatter





def model_run(feature_list,scaling=False):
    # Train test split by search date pacific
    
    train_split=df[(df['search_date_pacific'] > datetime(2018, 1, 20)) & (df['search_date_pacific'] < datetime(2018, 1, 27))]
    
    test_split=df[df['search_date_pacific']==datetime(2018, 1, 27)]
    
                  
    #As it is taking too much time to train the model I am randomly sampling from the train and test data
    rm_train_split=train_split.sample(10000)
    rm_test_split=test_split.sample(1000)

    
    #Creating Features and Labels
    
    '''
    feature_list=list(df.columns)[:7]
    
    X_train= np.array(train_split[feature_list])
    X_validation= np.array(test_split[feature_list])
    
    Y_train= np.array(train_split['apply'])
    Y_validation = np.array(test_split['apply'])
    '''

    X_train= np.array(rm_train_split[feature_list])
    X_validation= np.array(rm_test_split[feature_list])
    
    Y_train= np.array(rm_train_split['apply'])
    Y_validation = np.array(rm_test_split['apply'])
    
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    if scaling==True:
        print("Applying Standard Scaler")
        X_train=scaler.transform(X_train)
        X_validation=scaler.transform(X_validation)
        

        
    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'
    
    
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    
    
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
     
     
    # Compare Algorithms
    fig = plt.figure(figsize=(10,7))
    plt.title('10 fold Cross Validation Results')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.ylim(.5,1)
    plt.show()
    
    return X_train,Y_train,X_validation,Y_validation



    
#Run the models without considering user Category    
    
feature_list=list(df.columns)[:7]
X_train,Y_train,X_validation,Y_validation=model_run(feature_list,scaling=True)



# Make predictions on validation dataset with your best model
best_model = GaussianNB()
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_validation)

print("Accuracy on Training Sample", best_model.score(X_train, Y_train))

print("Accuracy on Test Sample", accuracy_score(Y_validation, predictions))

print("Confusion Matrix :\n",confusion_matrix(Y_validation, predictions))
print("Classification Report: \n", classification_report(Y_validation, predictions))
 


#Run the models considering user Category    

feature_list=list(df.columns)[:7]+['user_cat'] 
X_train,Y_train,X_validation,Y_validation=model_run(feature_list)
         
         
# Make predictions on validation dataset with your best model
best_model = KNeighborsClassifier(n_neighbors=1)
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_validation)


print("Accuracy on Training Sample", best_model.score(X_train, Y_train))
print("Accuracy on Test Sample", accuracy_score(Y_validation, predictions))
print("Confusion Matrix :\n",confusion_matrix(Y_validation, predictions))
print("Classification Report: \n", classification_report(Y_validation, predictions))
       













