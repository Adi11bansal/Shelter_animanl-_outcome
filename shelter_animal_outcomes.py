import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import re
import numpy as np
import pickle
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm
from sklearn import svm, grid_search
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
sns.set(style='ticks')


data = pd.read_csv("train.csv")
data = data[data["OutcomeType"] != "NaN"]
data['AnimalType'] =data['AnimalType'].fillna("Dog")
data['Breed'] =data['Breed'].fillna("Mix")
data['SexuponOutcome'] =data['SexuponOutcome'].fillna("Neutered Male")

del data["AnimalID"]
del data["OutcomeSubtype"]

data['Name'] =data['Name'].fillna(0)
data.loc[data['Name'] != 0, "Name"] = 1
data['Hour'] = data.DateTime.map( lambda x: pd.to_datetime(x).hour )
data['Day_of_Week'] = data.DateTime.map( lambda x: pd.to_datetime(x).dayofweek )
sns.countplot(data=data, x='Hour',hue="OutcomeType", palette = "Set3")
plt.show()



plt.figure(figsize=(20,10))
sns.countplot(data=data, x='Day_of_Week',hue="OutcomeType", palette = "Set3")
plt.show()

data.loc[data["Day_of_Week"]== 5, "Day_of_Week"] = 10
data.loc[data["Day_of_Week"]== 6, "Day_of_Week"] = 10
data.loc[data["Day_of_Week"]!= 10, "Day_of_Week"] = 0
data.loc[data["Day_of_Week"]== 10, "Day_of_Week"] = 1

def day(x):
    if x == 16 or x == 19:
        x = "16and19"
    elif x == 17 or x == 18:
        x = "17and18"
    elif x == 7 or x == 8:
        x = "7and8"
    elif x == 12 or x == 13:
        x = "12and13"
    elif x == 14 or x == 15:
        x = "14and15"
    elif x == 9:
        x = "9"
    elif x == 0:
        x = "0"
    else:
        x = "other_time"
    return x
       
data['Hour'] = data.Hour.apply(day)

def age_as_float(x):
    x = str(x)
    x_list = x.split(" ")
    if len(x_list)==2:
        if x_list[1] =='year': return 1.0
        elif x_list[1] =='years': return float(x_list[0])
        elif x_list[1] =='month': return float(x_list[0])/12
        elif x_list[1] =='months': return float(x_list[0])/12
        elif x_list[1] =='week': return float(x_list[0])
        elif x_list[1] =='weeks': return float(x_list[0])/54
        elif x_list[1] =='days': return float(x_list[0])/365
        else: return 0
    else:return 0


data['AgeuponOutcome'] = data.AgeuponOutcome.apply(age_as_float)
data.loc[data['AgeuponOutcome']== 0, "AgeuponOutcome"] = data['AgeuponOutcome'].median()


plt.figure(figsize=(15,8))
sns.pointplot(x="AnimalType", y="AgeuponOutcome", hue="OutcomeType", data=data)
plt.ylabel('Year\nAgeuponOutcome')
plt.show()



plt.figure(figsize=(15,8))
sns.pointplot(x="Day_of_Week", y="AnimalType", hue="OutcomeType", data=data)
plt.show()




def create_dummies(var):
    var_unique = var.unique()
    var_unique.sort()
    dummy = pd.DataFrame()
    for val in var_unique:
        d = var == val
        dummy[var.name + "_" + str(val)] = d.astype(int)
    return(dummy)

sex_dummies = create_dummies(data["SexuponOutcome"])
data = pd.concat([data, sex_dummies], axis=1)



hour = create_dummies(data["Hour"])
data = pd.concat([data, hour], axis=1)


items_counts = data['Breed'].value_counts()
to_my_breads = data['Breed'].value_counts() > 150

my_breeds = ["Domestic Shorthair", "Chihuahua Shorthair", "Labrador Retriever", 
             "Domestic Medium Hair", "German Shepherd", "Domestic Longhair", "Siamese", "Australian Cattle Dog", 
             "Dachshund", "Miniature Poodle", "Border Collie","Australian Shepherd", 
             "Pit Bull", "Boxer"]
def breeds(x):
    x = str(x)
    breed = "other"
    for b in my_breeds:
        if re.search(b, x):
            breed = b
    return breed
data['My_Breeds'] = data.Breed.apply(breeds)








sns.countplot(data=data, x='OutcomeType',hue="SexuponOutcome", palette = "Set3")
plt.show()

sns.countplot(data=data, x='OutcomeType',hue="OutcomeType", palette = "Set3")
plt.show()

sns.countplot(data=data, x='OutcomeType',hue="AnimalType", palette = "Set3")
plt.show()

plt.figure(figsize=(15,8))
sns.pointplot(x="Day_of_Week", y="SexuponOutcome", hue="OutcomeType", data=data)
plt.show()





g = sns.PairGrid(data,
                 x_vars=["AnimalType", "OutcomeType","SexuponOutcome"],
                 y_vars=["Day_of_Week"],
                 aspect=.75, size=7 )
g.map(sns.pointplot, palette="pastel");
plt.show()


plt.figure(figsize=(15,8))
sns.pointplot(x="Day_of_Week", y="My_Breeds", hue="OutcomeType", data=data)
plt.show()


plt.figure(figsize=(15,8))
sns.pointplot(x="AgeuponOutcome", y="My_Breeds", hue="OutcomeType", data=data)
plt.xlabel('Year')
plt.show()




g = sns.FacetGrid(data, col="My_Breeds", col_wrap=5, size=1.5)
g = g.map(plt.plot, "AnimalType", "OutcomeType", marker=".")
plt.show()

plt.figure(figsize=(15,8))
sns.countplot(data=data, x='OutcomeType',hue="My_Breeds", palette = "Set3")
plt.show()


data.loc[data["AnimalType"]== "Dog", "AnimalType"] = 1
data.loc[data["AnimalType"]!= 1, "AnimalType"] = 0

breed_dummies = create_dummies(data["My_Breeds"])
data = pd.concat([data, breed_dummies], axis=1)
del data["My_Breeds"]
del data["Breed"]
del data["Hour"]
del data["SexuponOutcome"]


##data.convert_objects(convert_numeric=True)
##def handle_non_numerical_data(data):
##    columns=data.columns.values
##
##    for column in columns:
##        text_digit_vals={}
##        def convert_to_int(val):
##            return text_digit_vals[val]
##
##        if data[column].dtype != np.int64 and data[column].dtype !=np.float64:
##            column_contents=data[column].values.tolist()
##            unique_elements=set(column_contents)
##            x=0
##            for unique in unique_elements:
##                if unique not in text_digit_vals:
##                     text_digit_vals[unique]=x
##                     x+=1
##               
##
##            data[column]=list(map(convert_to_int,data[column]))
##    return data
##data=handle_non_numerical_data(data)
features = data.columns.tolist() 
features.remove("DateTime")
features.remove("OutcomeType")
features.remove("Color")
data.dropna(inplace=True)
shuffled_rows = np.random.permutation(data.index)
highest_train_row = int(data.shape[0] * .70)
train = data.loc[shuffled_rows[:highest_train_row], :]
my_test = data.loc[shuffled_rows[highest_train_row:], :]


alg = linear_model.LogisticRegression(random_state=1)

alg.fit(train[features], train["OutcomeType"])
probs = alg.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)


alg9 = GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3)
alg9.fit(train[features], train["OutcomeType"])
probs = alg9.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10]}
svr = svm.SVC(probability=True)
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(train[features], train["OutcomeType"])
probs = clf.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

alg10 = GaussianNB()
alg10.fit(train[features], train["OutcomeType"])
probs = alg10.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

alg2 = RandomForestClassifier()
alg2.fit(train[features], train["OutcomeType"])
probs = alg2.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

alg7 = CalibratedClassifierCV()
alg7.fit(train[features], train["OutcomeType"])
probs = alg7.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

alg6 = DecisionTreeClassifier()
alg6.fit(train[features], train["OutcomeType"])
probs = alg6.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

alg4 =AdaBoostClassifier()
alg4.fit(train[features], train["OutcomeType"])
probs = alg4.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

alg8 = svm.SVC(probability=True)
alg8.fit(train[features], train["OutcomeType"])
probs = alg8.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

eclf = VotingClassifier(estimators=[('lr', alg), ('GBC', alg9),('calC', alg7)], voting='soft')
eclf.fit(train[features], train["OutcomeType"])
probs = eclf.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

eclf0 = VotingClassifier(estimators=[('lr', alg), ('svm', alg8)], voting='soft')
eclf0.fit(train[features], train["OutcomeType"])
probs = eclf0.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

eclf1 = VotingClassifier(estimators=[('lr', alg), ('calC', alg7)], voting='soft')
eclf1.fit(train[features], train["OutcomeType"])
probs = eclf1.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

eclf2 = VotingClassifier(estimators=[('lr', alg), ('svm', alg8), ("GBC", alg9)], voting='soft')
eclf2.fit(train[features], train["OutcomeType"])
probs = eclf2.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

eclf3 = VotingClassifier(estimators=[('lr', alg), ("GBC", alg9)], voting='soft')
eclf3.fit(train[features], train["OutcomeType"])
probs = eclf3.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

eclf4 = VotingClassifier(estimators=[('svm', alg8), ("GBC", alg9)], voting='soft')
eclf4.fit(train[features], train["OutcomeType"])
probs = eclf4.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

eclf5 = VotingClassifier(estimators=[('lr', alg), ('svm_gs', clf), ("GBC", alg9)], voting='soft')
eclf5.fit(train[features], train["OutcomeType"])
probs = eclf5.predict_proba(my_test[features])
score = metrics.log_loss(my_test["OutcomeType"], probs)
print(score)

test = pd.read_csv("test.csv")
test['AnimalType'] =test['AnimalType'].fillna("Dog")
test['Breed'] =test['Breed'].fillna("Mix")
test['SexuponOutcome'] =test['SexuponOutcome'].fillna("Neutered Male")
test['Name'] =test['Name'].fillna(0)
test.loc[test['Name'] != 0, "Name"] = 1
test.loc[test["AnimalType"]== "Dog", "AnimalType"] = 1
test.loc[test["AnimalType"]!= 1, "AnimalType"] = 0
test['Hour'] = test.DateTime.map( lambda x: pd.to_datetime(x).hour )
test['Day_of_Week'] = test.DateTime.map( lambda x: pd.to_datetime(x).dayofweek )
test.loc[test["Day_of_Week"]== 5, "Day_of_Week"] = 10
test.loc[test["Day_of_Week"]== 6, "Day_of_Week"] = 10
test.loc[test["Day_of_Week"]!= 10, "Day_of_Week"] = 0
test.loc[test["Day_of_Week"]== 10, "Day_of_Week"] = 1
test['Hour'] = test.Hour.apply(day)
test['AgeuponOutcome'] = test.AgeuponOutcome.apply(age_as_float)
test.loc[test['AgeuponOutcome']== 0, "AgeuponOutcome"] = test['AgeuponOutcome'].median()
sex_dummies = create_dummies(test["SexuponOutcome"])
test = pd.concat([test, sex_dummies], axis=1)
del test["SexuponOutcome"]
hour = create_dummies(test["Hour"])
test = pd.concat([test, hour], axis=1)
del test["Hour"]
test['My_Breeds'] = test.Breed.apply(breeds)
breed_dummies = create_dummies(test["My_Breeds"])
test = pd.concat([test, breed_dummies], axis=1)
del test["My_Breeds"]
del test["Breed"]

result = pd.DataFrame(alg7.predict_proba(test[features]), index=test.index, columns=alg7.classes_)
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission.head()
mid = sample_submission['ID']
result.insert(0, 'ID', mid)
result.dropna(inplace=True)
pickle_out=open('results.pickle','wb')
pickle.dump(result,pickle_out)
pickle_out.close()
pickle_in=open('results.pickle','rb')
result=pickle.load(pickle_in)
result.to_csv("etcareva11.csv", index=False)
