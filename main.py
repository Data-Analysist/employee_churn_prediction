import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy.stats import f_oneway
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import warnings

df = pd.read_csv('HR.csv')
df.head()
df.info()
# Work_accident vs. left in percentages
outcome_Work_accident = pd.crosstab(index = df['left'],
                                    columns = df['Work_accident'],
                                    normalize = 'index') # percentages based on index

outcome_Work_accident.index= ['Did not leave', 'Left']
print(outcome_Work_accident)
outcome_Work_accident.plot(kind = 'bar', stacked = True)

plt.title('Work_accident vs. left')
plt.xlabel('Outcome')
plt.ylabel('Employees')
plt.xticks(rotation = 0, horizontalalignment = 'center')

plt.show()

# promotion_last_5years vs. left in percentages
outcome_promotion_last_5years = pd.crosstab(index = df['left'],
                           columns = df['promotion_last_5years'], normalize = 'index')

outcome_promotion_last_5years.index= ['Did not leave', 'Left']
print(outcome_promotion_last_5years)
outcome_promotion_last_5years.plot(kind = 'bar', stacked = True)

plt.title('promotion_last_5years vs. left')
plt.xlabel('Outcome')
plt.ylabel('Employees')
plt.xticks(rotation = 0, horizontalalignment = 'center')

plt.show()

# department vs. left in percentages
outcome_department = pd.crosstab(index = df['left'],
                                 columns = df['department'],
                                 normalize = 'index')

outcome_department.index= ['Did not leave', 'Left']
print(outcome_department)
outcome_department.plot.barh(stacked = True)

plt.title('department vs. left')
plt.xlabel('Employees')
plt.ylabel('Outcome')
plt.xticks(rotation = 0, horizontalalignment = 'center')
plt.show()

# number of employees by department
df['department'].value_counts()
plt.figure(figsize = (10,5))

chart = sns.countplot(data = df, x = 'department')

chart.set_xticks(list(range(1, len(df['department'].value_counts())+1)))
chart.set_xticklabels(
    chart.get_xticklabels(),
    rotation = 45,
    horizontalalignment = 'right',
    fontweight = 'light',
    fontsize = 'large'
)

plt.show()

# salary counts
df['salary'].value_counts()
sns.countplot(x = 'salary',  data = df)
plt.show()

# salary vs. department in percentages
salary_dept = pd.crosstab(index = df['department'],
                          columns = df['salary'],
                          normalize='index')

print(salary_dept)

salary_dept.plot.barh(stacked = True)

plt.title('salary vs. department')
plt.xlabel('Salary')
plt.ylabel('Department')
plt.xticks(rotation = 0, horizontalalignment = 'center')

plt.show()

# salary vs. left in percentages
outcome_salary = pd.crosstab(index = df['left'],
                             columns = df['salary'],
                             normalize = 'index')

outcome_salary.index= ['Did not leave', 'Left']
print(outcome_salary)
outcome_salary.plot(kind = 'bar', stacked = True)

plt.title('salary vs. left')
plt.xlabel('Outcome')
plt.ylabel('Employees')
plt.xticks(rotation = 0, horizontalalignment = 'center')

plt.show()

# time_spend_company vs. left in percentages
outcome_time_spend_company = pd.crosstab(index = df['left'],
                                         columns = df['time_spend_company'],
                                         normalize = 'index')

outcome_time_spend_company.index = ['Did not leave', 'Left']
print(outcome_time_spend_company)
outcome_time_spend_company.plot.barh(stacked = True)

plt.title('time_spend_company vs. left')
plt.xlabel('Time spent, in years')
plt.ylabel('Outcome')

plt.show()

# mean number_project vs. left
proj_left = df.groupby('left').number_project.mean()
print(proj_left)
proj_left.plot(kind = 'bar', stacked = True)

plt.title('number_project vs. left')
plt.xlabel('Left')
plt.ylabel('number_project')
plt.xticks(rotation = 0, horizontalalignment = 'center')

plt.show()

# number_project vs. left in percentages
outcome_number_project = pd.crosstab(index = df['left'],
                                     columns = df['number_project'],
                                     normalize='index')

outcome_number_project.index = ['Did not leave', 'Left']
print(outcome_number_project)

outcome_number_project.plot.barh(stacked = True)

plt.title('number_project vs. left')
plt.xlabel('Number of projects')
plt.ylabel('Outcome')

plt.show()


#Numerical features
df[['satisfaction_level', 'last_evaluation', 'average_montly_hours']].describe()
f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': (.15, .85)})

sns.boxplot(x = df['satisfaction_level'], ax = ax_box)
sns.histplot(x = df['satisfaction_level'], ax = ax_hist, bins = 10, kde = True)

ax_box.set(xlabel = '')
ax_hist.set(xlabel = 'satisfaction_level distribution')
ax_hist.set(ylabel = 'frequency')

plt.show()

# trying log transformation
f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': (.15, .85)})

sns.boxplot(x = df['satisfaction_level'], ax = ax_box)
sns.histplot(x = df['satisfaction_level'], ax = ax_hist, bins = 10, kde = True).set_yscale('log')

ax_box.set(xlabel = '')
ax_hist.set(xlabel = 'satisfaction_level distribution (log)')
ax_hist.set(ylabel = 'frequency')

plt.show()

# trying Yeo-Johnson transformation

power = PowerTransformer(method = 'yeo-johnson', standardize = True)
sat_trans = power.fit_transform(df[['satisfaction_level']])
sat_trans = pd.DataFrame(sat_trans)
sat_trans.hist(bins = 20)

plt.show()

# satisfaction level vs. left
sat_left = df.groupby('left').satisfaction_level.mean()
print(sat_left)
sat_left.plot(kind = 'bar', stacked = True)

plt.title('satisfaction_level vs. left')
plt.xlabel('Left')
plt.ylabel('satisfaction_level')
plt.xticks(rotation = 0, horizontalalignment = 'center')

plt.show()

# satisfaction level by department
sat_dept = df.groupby('department').satisfaction_level.mean().sort_values()
print(sat_dept)
sat_dept.plot.barh(stacked = True)
plt.title('satisfaction_level by department')
plt.xlabel('satisfaction_level')
plt.ylabel('department')
plt.xticks(rotation = 0, horizontalalignment = 'center')
plt.xlim(0.55, 0.65)
plt.show()

f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': (.15, .85)})

sns.boxplot(x = df['last_evaluation'], ax = ax_box)
sns.histplot(x = df['last_evaluation'], ax = ax_hist, bins = 15, kde = True)

ax_box.set(xlabel = '')
ax_hist.set(xlabel = 'last_evaluation distribution')
ax_hist.set(ylabel = 'frequency')

plt.show()

# trying log transformation
f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': (.15, .85)})

sns.boxplot(x = df['last_evaluation'], ax = ax_box)
sns.histplot(x = df['last_evaluation'], ax = ax_hist, bins = 15, kde = True).set_yscale('log')

ax_box.set(xlabel = '')
ax_hist.set(xlabel = 'last_evaluation distribution (log)')
ax_hist.set(ylabel = 'frequency')

plt.show()

# trying Yeo-Johnson transformation
power = PowerTransformer(method = 'yeo-johnson', standardize = True)

eval_trans = power.fit_transform(df[['last_evaluation']])
eval_trans = pd.DataFrame(eval_trans)
eval_trans.hist(bins = 20)

plt.show()

# last_evaluation vs. left
eval_left = df.groupby('left').last_evaluation.mean()
print(eval_left)
eval_left.plot(kind = 'bar', stacked = True)

plt.title('last_evaluation  vs. left')
plt.xlabel('left')
plt.ylabel('last_evaluation')
plt.xticks(rotation = 0, horizontalalignment = 'center')
plt.ylim(0.7, 0.74)

plt.show()

#Average monthly hours
f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': (.15, .85)})

sns.boxplot(x = df['average_montly_hours'], ax = ax_box)
sns.histplot(x = df['average_montly_hours'], ax = ax_hist, bins = 15, kde = True)

ax_box.set(xlabel = '')
ax_hist.set(xlabel = 'average_montly_hours distribution')
ax_hist.set(ylabel = 'frequency')

plt.show()
# trying log transformation
f, (ax_box, ax_hist) = plt.subplots(2, sharex = True, gridspec_kw = {'height_ratios': (.15, .85)})

sns.boxplot(x = df['average_montly_hours'], ax = ax_box)
sns.histplot(x = df['average_montly_hours'], ax = ax_hist, bins = 15, kde = True).set_yscale('log')

ax_box.set(xlabel = '')
ax_hist.set(xlabel = 'average_montly_hours distribution (log)')
ax_hist.set(ylabel = 'frequency')

plt.show()

# trying Yeo-Johnson transformation
power = PowerTransformer(method = 'yeo-johnson', standardize = True)

hours_trans = power.fit_transform(df[['average_montly_hours']])
hours_trans = pd.DataFrame(hours_trans)
hours_trans.hist(bins = 20)

plt.show()

# trying Yeo-Johnson transformation
power = PowerTransformer(method = 'yeo-johnson', standardize = True)

hours_trans = power.fit_transform(df[['average_montly_hours']])
hours_trans = pd.DataFrame(hours_trans)
hours_trans.hist(bins = 20)

plt.show()


# average_montly_hours vs. left
hours_left = df.groupby('left').average_montly_hours.mean()
print(hours_left)
hours_left.plot(kind = 'bar', stacked = True)

plt.title('average_montly_hours vs. left')
plt.xlabel('left')
plt.ylabel('average_montly_hours')
plt.xticks(rotation = 0, horizontalalignment = 'center')
plt.ylim(150, 220)

plt.show()

# satisfaction_level outliers
q1 = df.satisfaction_level.quantile(0.25)
q3 = df.satisfaction_level.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
print(lower_bound, upper_bound)

outliers_sat = df[(df.satisfaction_level < lower_bound) | (df.satisfaction_level > upper_bound)]
outliers_sat.head()

# last_evaluation outliers
q1 = df.last_evaluation.quantile(0.25)
q3 = df.last_evaluation.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
print(lower_bound, upper_bound)

eval = df[(df.last_evaluation < lower_bound) | (df.last_evaluation > upper_bound)]
eval.head()

# average_montly_hours outliers
q1 = df.average_montly_hours.quantile(0.25)
q3 = df.average_montly_hours.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)
print(lower_bound, upper_bound)

hours = df[(df.average_montly_hours < lower_bound) | (df.average_montly_hours > upper_bound)]
hours.head()

# looking at the means
sal_hours = df.groupby('salary').average_montly_hours.mean().sort_values()
print(sal_hours)
sal_hours.plot.barh(stacked = True)

plt.title('average_montly_hours by salary')
plt.xlabel('average_montly_hours')
plt.ylabel('salary')
plt.xticks(rotation = 0, horizontalalignment = 'center')
plt.xlim(190, 208)

plt.show()

# splitting data into three samples
low = df[df['salary'] == 'low']
low = low[['average_montly_hours']]

medium = df[df['salary'] == 'medium']
medium = medium[['average_montly_hours']]

high = df[df['salary'] == 'high']
high = high[['average_montly_hours']]
# size of each sample
print(len(low), len(medium), len(high))

f_oneway(low, medium, high)


#Data transformation

# log transformation of numerical variables
df['sat_level_log'] = np.log(df['satisfaction_level'])
df['last_eval_log'] = np.log(df['last_evaluation'])
df['av_hours_log'] = np.log(df['average_montly_hours'])

# changing column order
columnsTitles = ['satisfaction_level', 'sat_level_log', 'last_evaluation', 'last_eval_log', 'number_project', 'average_montly_hours', 'av_hours_log', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'department', 'salary', 'left']
df = df.reindex(columns = columnsTitles)

# transforming 'salary' catogories into 'int'
labelencoder = LabelEncoder()
df['salary'] = labelencoder.fit_transform(df['salary'])

# transforming 'deparment' catogories into 'int'
labelencoder = LabelEncoder()
df['department'] = labelencoder.fit_transform(df['department'])


#correlation

df_c = df[
    ['satisfaction_level',
    'sat_level_log',
    'last_evaluation',
    'last_eval_log',
    'number_project',
    'average_montly_hours',
    'av_hours_log',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years',
    'salary'
    ]
  ]

plt.figure(figsize = (12,10))

cor = df_c.corr(method = 'kendall')

ax = sns.heatmap(cor,
                 annot = True,
                 vmin = -1,
                 vmax = 1,
                 center = 0,
                 cmap = plt.cm.Reds) # cmap = sns.diverging_palette(20, 220, n = 200)

ax.set_xticklabels(ax.get_xticklabels(),
                   rotation = 45,
                   horizontalalignment = 'right')

plt.show()

# largest and smallest correlation
c = df_c.corr(method = 'kendall').abs()
s = c.unstack()
so = s.sort_values(kind = 'quicksort')

#Multicollineary
warnings.filterwarnings('ignore')

# Getting variables for which to compute VIF and adding intercept term
X = df[
    ['satisfaction_level',
    'last_eval_log',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years'
    ]
  ]

X['Intercept'] = 1


# Compute and view VIF
vif = pd.DataFrame()
vif['variables'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif)


#Modelling
#Linear Discriminant Analysis
df_model = df[
    ['satisfaction_level',
    'last_eval_log',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years'
    ]
  ]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_model, df['left'], test_size = 0.30, random_state = 42)
# feature scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# LDA model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

lda.predict(X_test)
accuracy_score(y_test, lda.predict(X_test))

#Logistic regression

df_model_2 = df[
    ['satisfaction_level',
    'last_eval_log',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years'
    ]
  ]

X_train, X_test, y_train, y_test = train_test_split(df_model_2, df['left'], test_size = 0.30, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lr = LogisticRegression(random_state = 42)
lr.fit(X_train, y_train)
print(confusion_matrix(y_test, lr.predict(X_test)))
print(accuracy_score(y_test, lr.predict(X_test)))

#Random Forest Classifier

df_model_3 = df[
    ['satisfaction_level',
    'last_eval_log',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years',
    'department',
    'salary'
    ]
  ]

X_train, X_test, y_train, y_test = train_test_split(df_model_3, df['left'], test_size = 0.30, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = RandomForestClassifier(criterion = 'gini',
                                    n_estimators = 100,
                                    max_depth = 9,
                                    random_state = 42,
                                    n_jobs = -1)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

feat_labels = ['satisfaction_level',
    'last_eval_log',
    'number_project',
    'average_montly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years',
    'department',
    'salary'
    ]
for feature in zip(feat_labels, classifier.feature_importances_):
    print(feature)

from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(classifier, threshold = 0.10)
sfm.fit(X_train, y_train)
for feature_list_index in sfm.get_support(indices = True):
    print(feat_labels[feature_list_index])
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators = 100, random_state = 42, n_jobs = -1)

# new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)
y_important_pred = clf_important.predict(X_important_test)
print(accuracy_score(y_test, y_important_pred))