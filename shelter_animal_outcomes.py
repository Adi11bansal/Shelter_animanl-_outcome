import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
import re
import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn import svm, grid_search
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier


data = pd.read_csv("train.csv")
data = data[data["OutcomeType"] != "NaN"]
data['AnimalType'] =data['AnimalType'].fillna("Dog")
data['Breed'] =data['Breed'].fillna("Mix")
data['SexuponOutcome'] =data['SexuponOutcome'].fillna("Neutered Male")



del data["AnimalID"]
del data["OutcomeSubtype"]


data['Name'] =data['Name'].fillna(0)
data.loc[data['Name'] != 0, "Name"] = 1
data.loc[data["AnimalType"]== "Dog", "AnimalType"] = 1
data.loc[data["AnimalType"]!= 1, "AnimalType"] = 0
data['Hour'] = data.DateTime.map( lambda x: pd.to_datetime(x).hour )


data['Day_of_Week'] = data.DateTime.map( lambda x: pd.to_datetime(x).dayofweek )
##sns.countplot(data=data, x='Day_of_Week',hue="OutcomeType", palette = "Set3")


columns = ('Adoption','Died','Euthanasia','Return_to_owner','Transfer')

rows = ['%d Day_of_Week' % x for x in(1,2,3,4,5,6,7)]

values =np.arange(0,2500,500)
value_increment =1000

colors =plt.cm.BuPu(np.linspace(0,0.5,len(rows)))
n_rows =len(data)


index =np.arange(len(columns)) + 0.3
bar_width =0.4

y_offset =np.zeros(len(columns))


cell_text =[]
try:
    for row in range(n_rows):
        plt.bar(index,data[row],bar_width,bottom=y_offset,color =colors[row])
        y_offset =y_offset_data[row]
        cell_text.append(['%1.1f' % (x / 1000.0) for x in y_offset])



        colors =colors[::-1]

        cell_text.reverse()


        the_table = plt.table(cellText=cell_text,
                              rowLabels=rows,
                              rowColours=colors,
                              colLabels=columns,
                              loc='bottom')
        plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.ylabel("Loss in ${0}'s".format(value_increment))
        plt.yticks(values * value_increment, ['%d' % val for val in values])
        plt.xticks([])
        plt.title('Loss by Disaster')
        plt.show()

except Exception as e:
    pass




data.loc[data["Day_of_Week"]== 5, "Day_of_Week"] = 10
data.loc[data["Day_of_Week"]== 6, "Day_of_Week"] = 10
data.loc[data["Day_of_Week"]!= 10, "Day_of_Week"] = 0
data.loc[data["Day_of_Week"]== 10, "Day_of_Week"] = 1


sns.countplot(data=data, x='Hour',hue="OutcomeType", palette = "Set3")
plt.xticks(rotation=45)

plt.show()



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


def create_dummies(var):
    var_unique = var.unique()
    var_unique.sort()
    dummy = pd.DataFrame()
    for val in var_unique:
    #for val in var_unique[:-1]:
        # which columns are equal to our unique value
        d = var == val
        # make a new column with a dummy variable
        dummy[var.name + "_" + str(val)] = d.astype(int)
    return(dummy)

sex_dummies = create_dummies(data["SexuponOutcome"])
data = pd.concat([data, sex_dummies], axis=1)


del data["SexuponOutcome"]


hour = create_dummies(data["Hour"])
data = pd.concat([data, hour], axis=1)
del data["Hour"]

items_counts = data['Breed'].value_counts()
to_my_breads = data['Breed'].value_counts() > 150

my_breeds = ["Domestic Shorthair", "Chihuahua Shorthair", "Labrador Retriever", 
             "Domestic Medium Hair", "German Shepherd", "Domestic Longhair", "Siamese", "Australian Cattle Dog", 
             "Dachshund", "Miniature Poodle", "Border Collie","Australian Shepherd", "Rat Terrier", "Catahoula", 
              "Husky", "Rottweiler", "Bulldog", "Pit Bull", "Boxer"]
def breeds(x):
    x = str(x)
    breed = "other"
    for b in my_breeds:
        if re.search(b, x):
            breed = b
    return breed
data['My_Breeds'] = data.Breed.apply(breeds) 
sns.countplot(data=data, x='My_Breeds',hue="OutcomeType", palette = "Set3")
plt.xticks(rotation=45)
plt.show()



breed_dummies = create_dummies(data["My_Breeds"])
data = pd.concat([data, breed_dummies], axis=1)
del data["My_Breeds"]
del data["Breed"]


'''
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
##data=handle_non_numerical_data(data)'''
features = data.columns.tolist() 
features.remove("DateTime")
features.remove("OutcomeType")
features.remove("Color")

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



####parameters = {'kernel':('linear', 'rbf'), 'C':[0.001, 0.01, 0.1, 1, 10]}
####svr = svm.SVC(probability=True)
####clf = grid_search.GridSearchCV(svr, parameters)
####clf.fit(train[features], train["OutcomeType"])
####probs = clf.predict_proba(my_test[features])
####score = metrics.log_loss(my_test["OutcomeType"], probs)
####print(score)
####
####eclf5 = VotingClassifier(estimators=[('lr', alg), ('svm_gs', clf), ("GBC", alg9)], voting='soft')
####eclf5.fit(train[features], train["OutcomeType"])
####probs = eclf5.predict_proba(my_test[features])
####score = metrics.log_loss(my_test["OutcomeType"], probs)
####print(score)





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

result = pd.DataFrame(alg.predict_proba(test[features]), index=test.index, columns=alg.classes_)
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission.head()
mid = sample_submission['ID']
result.insert(0, 'ID', mid)
result.to_csv("etcareva13.csv", index=False)

