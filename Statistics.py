#!/usr/bin/env python
# coding: utf-8

# <center style = "color:#e85046"><h1><u> WORLD LIFE EXPECTANCY AND IT'S DEPENDENT FACTORS</u></h1></center>

# <img src="world.gif" width="750" align="center">

# # Understanding the Research Context

# The Global Health Observatory (GHO) data repository under World Health Organization (WHO) keeps track of the health status as well as many other related factors for all countries.
# 
# The data-set related to life expectancy, health factors for 193 countries has been collected from the same WHO data repository website. Among all categories of health-related factors only those critical factors were chosen which are more representative. 
# 
# It has been observed that in the past 15 years , there has been a huge development in health sector resulting in improvement of human mortality rates especially in the developing nations in comparison to the past 30 years. Therefore, in this project we have considered data from year 2000-2015 for 193 countries for further analysis. The result indicated that most of the missing data was for population, Hepatitis B and GDP. The missing data were from less known countries like Vanuatu, Tonga, Togo, Cabo Verde etc.

# # Experimental Design

# Below are the steps that will be conducted in this analysis in order to fulfill the project goal satisfactorily:
# 
# 1. Load data and preview preliminary characteristics of the dataset
# 
# 2. Data cleaning (check for and deal with outliers, anomalies and missing data)
# 
# 3. Perform univariate and bivariate analysis
# 
# 4. Specify the null and alternate hypotheses
# 
# 5. Perform hypothesis testing
# 
# 6. Discuss the hypothesis test results
# 
# 7. Provide project summary and conclusion

# # General Information of the Data
# <b>Country</b> : Country<br>
# <b>Year</b> : Year<br>
# <b>Status</b> : Developed or Developing status of the country<br>
# <b>Life expectancy</b> : Number of years a person is expected to live.<br>
# <b>Adult Mortality</b> : Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)<br>
# <b>infant deaths</b> : Number of Infant (very young child or baby) Deaths per 1000 population<br>
# <b>Alcohol</b> : Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)<br>
# <b>Percentage</b> : Expenditure on health as a percentage of Gross Domestic Product per capita(%)<br>
# <b>Hepatitis B</b> : Hepatitis B (HepB) immunization coverage among 1-year-olds (%)<br>
# <b>Measles</b> : Measles - number of reported cases<br>
# <b>BMI</b> : Average Body Mass Index of entire population<br>
# <b>Under-five deaths</b> : Number of under-five deaths per 1000 population<br>
# <b>Polio</b> : Polio (Pol3) immunization coverage among 1-year-olds (%)<br>
# <b>Total expenditure</b> : General government expenditure on health as a percentage of total government expenditure (%)<br>
# <b>Diphtheria</b> : Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)<br>
# <b>HIV/AIDS</b> : Deaths per 1 000 live births HIV/AIDS (0-4 years)<br>
# <b>GDP</b> : Gross Domestic Product per capita (in USD)<br>
# <b>Population </b>: Population of the country<br>
# <b>Thinness 1-19 years</b> : Prevalence of thinness among children and adolescents for Age 10 to 19 (% )<br>
# <b>Income composition of resources</b> : Human Development Index in terms of income composition of resources (index ranging <b>from 0 to 1)<br>
# Thinness 5-9 years</b> : Prevalence of thinness among children for Age 5 to 9(%)<br>
# <b>Schooling</b> : Number of years of Schooling(years).
# 

# # Reading the Data

# In[110]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import plotly.express as px
import plotly.io as pio
import scikit_posthocs


# In[111]:


# Loading data from the source (csv file)

df = pd.read_csv('Life Expectancy Data.csv')
df


# # Checking the Data

# In[112]:


# Checking the top 5 records of the dataset

df = pd.read_csv('Life Expectancy Data.csv')
df.head()


# In[113]:


# Checking the bottom 5 records of the dataset

df = pd.read_csv('Life Expectancy Data.csv')
df.tail()


# In[114]:


# Previewing a random sample of the dataset

df.sample(n=5)


# In[115]:


# Checking the no. of rows and columns
df.shape


# In[116]:


df.info()


# We can see that Year column in of dtype int64, let's convert it into object

# In[117]:


df['Year'] = df['Year'].apply(str)
df.rename(columns = {' thinness  1-19 years': 'thinness 10-19 years'}, inplace=True)


# In[118]:


df.info()


# # Missing Values

# In[119]:


df.isnull().sum()


# In[120]:


df.dropna(axis=0, how='any', subset =['Population', 'GDP'], inplace=True)


# In[121]:


df.isnull().sum()


# # Types of Data
# 
# ### Qualitative

# In[122]:


cat_cols = [df.columns[i] 
            for i in range(0, df.shape[1]-1)  
            if df.iloc[:,i].dtype=='O']
cat_cols


# ### Quantitative

# In[123]:


num_cols = [c for c in df.columns if c not in cat_cols]
num_cols


# # Outliers

# In[124]:


fig, axes = plt.subplots(len(num_cols),1,figsize=(8,50))
for i,c in enumerate(df[num_cols]):
    df[[c]].boxplot(ax=axes[i], vert=False)


# ### Conclusion:

# 1.Outliers are absent in BMI
# 
# 
# 2.Less outliers in Lower Fence - LIFE EXPECTANCY, INCOME COMPOSITION OF RESOURCES, SCHOOLING.
# 
# 
# 3.More outliers in Lower Fence - HEPATITIS B, POLIO, DIPTHERIA.
# 
# 
# 4.Less outliers in Upper Fence - ALCOHOL, TOTAL EXPENDITURE.
# 
# 
# 5.More outliers in Upper Fence - ADULT MORTALITY, INFANT DEATHS, PERCENTAGE EXPENDITURE, MEASLES, UNDER-FIVE DEATHS, HIV/AIDS, GDP, POPULATION, THINNESS 1-19 YEARS, THINNESS 5-9 YEARS.
# 
# 
# Hence Outliers can have a disproportionate effect on statistical results, such as the mean, which can result in  misleading interpretations.
# 

# # Data Cleaning

# In[125]:


print('Max of infant deaths: ',df['infant deaths'].max())
print('Max of Measles',df['Measles '].max())
print('Max of BMI',df[' BMI '].max())
print('Max of under-five deaths',df['under-five deaths '].max())


# >infant_deaths (per 1000 of population) has max value 1800. Mathematically unlikely they are having more infant deaths than actual infants. Likely a data collection error. We will drop outliers over 1000.
# 
# >measles (number of reported cases per 1000) has possible outliers at 212k. Same explanation and course.
# 
# >bmi has a distribution that does not make sense. A bmi over 40 is morbidly obese and somehow some countries recorded a mean of almost 90. That means there is a country (or multiple) with a such a large population of people that are shorter than 5'0" and weigh over 400 lbs. This variable may be dropped entirely.
# 
# >under_five_deaths (per 1000 of population) has outliers of 2500. Same explantion and course of action.

# In[126]:


df.drop([' BMI ', 'infant deaths', 'Measles ', 'under-five deaths '], axis = 1, inplace=True)
num_cols = [c for c in df.columns if c not in cat_cols]


# # Distribution

# In[127]:


fig, axes = plt.subplots(8, 2, figsize=(18,60))
axes = [ax for axes_rows in axes for ax in axes_rows]

for i, c in enumerate(num_cols):
    plot = sns.distplot(df[c], ax=axes[i])
    plot.axvline(x=df[c].mean(), linewidth=3, color='g', label="mean", alpha=0.5)
    plot.axvline(x=df[c].median(), linewidth=3, color='y', label="median", alpha=0.5)
    
    details = str(c)+'\nSD = '+str( round(df[c].std(), 2) ) + '\n' + 'Var = '+str( round(df[c].var(), 2) ) + '\n' + 'Range = ' + str(round(df[c].max()-df[c].min(), 2))
    plot.set_xlabel(details)
    axes[i].legend(loc="best", fontsize= 12)


# In[128]:


for i, c in enumerate(num_cols):
    print(i+1,') ',c)
    print('\tKurtosis: ', df[c].kurt(), ', Skewness: ', df[c].skew())
    print()


# <center><h3> INSIGHTS </h3></center>
# 
# >Life expectancy is negatively skewed and platykurtic with few outlier’s present. Life expentency lies between 40-90 and with maximum density around 70 years.
# 
# >Infant deaths is positively skewed and highly leptokurtic. Since, the variance and standard deviation is high there are more outliers present.
# 
# >Hepatitis B is highly negatively skewed and leptokurtic. There are larger countries where immunisation is low.
# 
# POSITIVE SKEWED      | NEGATIVE SKEWED |  LEPTOKURTIC | PLATYKURTIC
# :---------:|:-----------:|:-----------:|:-----------:
# Adult Mortality | Life expectancy  | Adult Mortality | Life expectancy
#  Alcohol| Hepatitis B | percentage expenditure | Alcohol
#  percentage| Polio | Hepatitis B | Total expenditure
#  expenditure | Diphtheria | Polio | 
#  Total expenditure| Income composition of resources | Diphtheria | 
#  HIV/AIDS| Schooling | HIV/AIDS | 
# GDP |  | GDP | 
#  Population| | Population | 
#  thinness 10-19 years| | thinness 10-19 years | 
#  thinness 5-9 years| | thinness 5-9 years | 
#    | | Income composition of resources | 
#   | | Schooling | 

# # Bivariate Analysis

# In this analysis, we will check various relationships between different measures and dimensions

# ### Correlation

# In[129]:


plt.figure(figsize=(15, 10))
_ = sns.heatmap(df[num_cols].corr(), annot=True)


# #### Conclusion
#  
#  
# >Positively Correlated:
# 1.  Hepatitis B vaccine rate is relatively positively correlated with Polio and Diphtheria vaccine rates
# 2.	Thinness 10-19 years with respect to Thinness 5-9 years is positively correlated.
# 3.	GDP with respect to percentage expenditure is also very much positively correlated.
# 4.  Life Expectancy is positively correlated with Income Composition of Resources and Schooling.
# 
# >Negatively Correlated:
# 1.	HIV with respect to Life expectancy is negatively correlated.
# 2.	Thinness 10-19 years, thinness 5-9 years are in negative correlation with respect to Life expectancy.
# 3.  Life Expectancy negatively correlated with Adult Mortality.
# 
# >No-Correlation:
# 1.	Life expectancy (target variable) is extremely lowly correlated to population (nearly no correlation at all).
# 2.	Again, HIV with respect to Total expense has a negligible correlation.
# 3.  Hepatitis-B has nearly no correlation with precentage expenditure.
# 
# 

# #### Trend Analysis

# In[130]:


fig, axes = plt.subplots(3, 2, figsize=(20,15))
axes = [ax for axes_rows in axes for ax in axes_rows]

for i, c in enumerate(['Adult Mortality', 'Hepatitis B', 'Polio', 'Income composition of resources','Diphtheria ']):
    ageDataSN = df.groupby("Year", as_index = False)[c].mean()
    sns.lineplot(data = ageDataSN, x='Year', y=c, ax=axes[i]).set_title(c)


# #### Conclusion:
# 
# 
# 1. Adult Mortality - There was a certain rise and fall in Adult Mortality from the year 2000 to 2015 but it was lowest in the year 2013.
# 
# 
# 2. Hepatitis B Immunization - From 2000 to 2003 their was a fall and a sudden peak from 2003 to 2008 and from 2008 there is a rise and fall till 2015.
# 
# 
# 3. Polio -  From 2000 to 2008 there is a rise and fall from 2008 and a slight fall till 2015.
# 
# 
# 4. Income composition of resources - There was a certain  rise in Income composition of resources from the year 2000 to 2015.
# 
# 
# 5. Diptheria - From 2000 to 2008 there is a rise and from 2008 there was a rise and fall till 2015.

# ## Mean Gross domestic product (GDP) of each country

# In[131]:


ageDataSN = df.groupby("Country", as_index = False)["GDP"].mean().sort_values(by="GDP",ascending = False)
plt.figure(figsize=(10,50))
sns.barplot(x = "GDP", 
            y = "Country", 
            data = ageDataSN)
plt.xlabel('GDP')
plt.ylabel('Country')
plt.title('GDP of countries')
plt.show()


# ## Conclusion:
# 
# After observing the above graphs we can come to a conclution that , the GDP is highest in Switzerland and the country having the lowest GDP is Burundi.

# ## Each Countries mean total expenditure towards the health sector 

# In[132]:


ageDataSN = df.groupby("Country", as_index = False)["Total expenditure"].mean().sort_values(by="Total expenditure",ascending = False)
plt.figure(figsize=(10,40))
sns.barplot(x = "Total expenditure", 
            y = "Country", 
            data = ageDataSN)
plt.xlabel('Total expenditure')
plt.ylabel('Country')
plt.title('Total expenditure of countries towards health sector')
plt.show()


# ### Conclusion
# 
# After observing the above graphs we can come to a conclution that ,country TUVALU has the most expenditure towards health sector and the country having the least expenditure towards health sector is Timor-Leste.

# ## Income composition of resources of each country

# In[133]:


ageDataSN = df.groupby("Country", as_index = False)["Income composition of resources"].mean().sort_values(by="Income composition of resources",ascending = False)
plt.figure(figsize=(10,50))
sns.barplot(x = "Income composition of resources", 
            y = "Country", 
            data = ageDataSN)
plt.xlabel('Income composition of resources')
plt.ylabel('Country')
plt.title('Income composition of resources of each countries')
plt.show()


# ### Conclusion
# After observing the above graphs we can come to a conclution that ,the country having the highest Income composition of resources is Norway and the country having the lowest Income composition of resources is Tuvalu.

# ## Tests

# ### Test-1
# The term “life expectancy” refers to the number of years a person can expect to live.
# 
# Research:
# 
# Worldwide, the average life expectancy at birth was 71 years (70 years for males and 72 years for females) 
# over the period 2010–2015 according to United Nations World Population Prospects 2015 Revision.
# 
# According to WHO Life expectancy increased by 5 years between 2000 and 2015.
# 
# 
# <b>Let's claim that mean Life expectancy is less than 71</b>
# 
# H0: mean Life expectancy >= 71<br>
# H1: mean Life expectancy < 71

# In[134]:


#Left tailed test
st.shapiro(df['Life expectancy '].dropna())


# From the above result, we can see that the <b>p-value is less than 0.05 (alpha)</b>, 
# thus we can say that the data is not normally distributed.

# H0: median Life expectancy >= 71<br>
# H1: median Life expectancy < 71

# In[135]:



M_1 = 71 #hypothesized median
Sample_data1 = np.array(df['Life expectancy '].dropna())

# calculate the difference between diameter and M_0
diff1 = Sample_data1 - M_1

# perform wilcoxon signed rank test
# pass the differnces to the parameter, 'x'
# pass the one-tailed condition to the parameter, 'alternative'
test_stat1, p_value1 = st.wilcoxon(x = diff1, alternative = 'less')

# print the test statistic and corresponding p-value
print('Test statistic:', test_stat1)
print('p-value:', p_value1)


# <b>p-value < alpha, reject the null hypothesis (H0)</b>
# 
# Inference:
# Thus, we conclude that median Life expectancy is less than 71 years

# ### Test-2
# <b>Test the claim that life expectancy in developed countries is same as that of in the developing countries</b>

# The null and alternative hypothesis<br>
# H0: Life expectancy in developed countries = Life expectancy in developing countries<br>
# H1: Life expectancy in developed countries != Life expectancy in developing countries<br>
# 2-tailed  test

# In[136]:



Life_expectancy_Developed = df[ df['Status'] == 'Developed' ]['Life expectancy '].dropna()
Life_expectancy_Developing = df[ df['Status'] == 'Developing' ]['Life expectancy '].dropna()

print(st.shapiro(Life_expectancy_Developed))
print(st.shapiro(Life_expectancy_Developing))


# <b>p< alpha(0.05)</b> hence it does not follow normal distribution.

# In[137]:


st.mannwhitneyu(Life_expectancy_Developed,Life_expectancy_Developing,  alternative = 'two-sided')


# <b>p< alpha(0.05)</b>, reject the null hypothesis
# 
# Inference:
# Life_expectancy in developed countries is different as compared to developing countries

# ### Test-3
# <b>Test the claim that life expectancy in the top 5 developed countries is same (as per the latest results/Year)</b>

# In[138]:


#Finding top 5 countries based on Life expectancy
df1 = df[['Country', 'Status', 'Life expectancy ','Year']].dropna()
df2 = df1[df1['Status']=='Developed']
df3 = df2[df2['Year']=='2015'].sort_values(by='Life expectancy ', ascending=False).head(n=5)
df3


# The null and alternative hypothesis<br>
# H0: Life expectancy of top 5 developed countries is same<br>
# H1: Life expectancy of top 5 developed countries is different

# In[139]:


Slovenia = df[ df['Country'] == 'Slovenia' ]['Life expectancy '].dropna()
Denmark = df[ df['Country'] == 'Denmark' ]['Life expectancy '].dropna()
Cyprus = df[ df['Country'] == 'Cyprus' ]['Life expectancy '].dropna()
Japan = df[ df['Country'] == 'Japan' ]['Life expectancy '].dropna()
Switzerland = df[ df['Country'] == 'Switzerland' ]['Life expectancy '].dropna()

for i in [Slovenia, Denmark, Cyprus, Japan, Switzerland]:
    print(st.shapiro(i))
    
st.levene(Slovenia.values, Denmark.values, Cyprus.values, Japan.values, Switzerland.values)


# <b>p< alpha(0.05)</b> hence it does not follow normal distribution.

# In[140]:


# perform kruskal-wallis H test
test_stat1, p_val1 = st.kruskal(Slovenia, Denmark, Cyprus, Japan, Switzerland)

# print the test statistic and corresponding p-value
print('Test statistic:', test_stat1)
print('p-value:', p_val1)


# <b>p_value < alpha</b>, reject the null hypothesis H0. So, Life expectancy of top 5 developed countries is different

# ### Post-hoc test

# In[141]:


data = df[(df['Country']=='Slovenia')| 
          (df['Country']=='Denmark')|
          (df['Country']=='Cyprus')|
          (df['Country']=='Japan') |
          (df['Country']=='Switzerland')]
df2 = data[['Country','Life expectancy ']]


# In[142]:


scikit_posthocs.posthoc_conover(a = df2, val_col = 'Life expectancy ', group_col = 'Country')


# The Pvalue for countries (Cyprus,Japan) , (Cyprus , Switzerland),(Denmark,Japan),(Denmark,Switzerland),(Japan,Slovenia) and (Slovenia,Switzerland) is less than alpha and thus there is difference in Life expectancies between these countries

# ### Test-4
# <b>Test the claim that population depends upon the status and Year</b>

# In[143]:


df3 = df[['Status', 'Year', 'Population']].dropna()
df3


# The null and alternative hypothesis<br>
# <br>
# Status<br>
# H0: Population depends upon the Status<br>
# H1: Population does not depend upon the Status<br>
# <br>
# Year<br>
# H0: Population depends upon the Year<br>
# H1: Population does not depend upon the Year

# In[144]:


m3 = ols('Population~Status+Year', data = df3).fit()
anova_table3=anova_lm(m3)
anova_table3


# Inference: Population depends upon the Year, but not on the status

# ### Test-5
# <b>Test the claim that total expenditure on health is more in developed countries as compared to developing countries</b>

# The null and alternative hypothesis<br>
# 
# H0: Total_expenditure in developed countries <= Total_expenditure in developing countries<br>
# H1: Total_expenditure in developed countries > Total_expenditure in developing countries<br>
# Right tailed test

# In[145]:


Total_expenditure_Developed = df[ df['Status'] == 'Developed' ]['Total expenditure'].dropna()
Total_expenditure_Developing = df[ df['Status'] == 'Developing' ]['Total expenditure'].dropna()

print(st.shapiro(Total_expenditure_Developed.dropna()))
print(st.shapiro(Total_expenditure_Developing.dropna()))


# <b>p< alpha(0.05)</b> hence it does not follow normal distribution.

# In[146]:


st.mannwhitneyu(Total_expenditure_Developed, Total_expenditure_Developing, alternative = 'greater')


# <b>p_value < alpha</b>, reject the null hypothesis
# 
# Inference: Total_expenditure on health in developed countries is more as compared to developing countries

# ### Test-6
# <b>Test that claim that proportion of Hepitatis-B immunization is same for developed and developing countires</b>
# 
# Hepatitis B : Hepatitis B (HepB) immunization coverage among 1-year-olds (%)

# H0: 𝑃1−𝑃2=0 ie. Proportion for Hep B immunization is same for Developed and Developing Countries<br>
# H1: 𝑃1−𝑃2≠0 ie. Proportion for Hep B immunization is different for Developed and Developing Countries

# In[147]:


#Here ⍺ = 0.1, for a two-tailed test calculate the critical z-value.
developed_total_hep  = df[df['Status']=='Developed']['Hepatitis B'].dropna().sum()
n1 =(len(df[df['Status']=='Developed']['Hepatitis B'].dropna()))*100
p1 = developed_total_hep/n1

developing_total_hep  = df[df['Status']=='Developing']['Hepatitis B'].dropna().sum()
n2 = (len(df[df['Status']=='Developing']['Hepatitis B'].dropna()))*100
p2 = developing_total_hep/n2

hypo_po = (n1*p1 + n2*p2)/(n1 + n2)
proportion_statistic = np.sqrt(hypo_po*(1-hypo_po)*(1/n1 + 1/n2))
z_critical = np.abs(round(st.norm.isf(q = 0.05/2), 2))
print('Z-statistic:', proportion_statistic)
print('Critical value:', z_critical)


# Since <b>z-statistic is less than critical value</b> we fail to reject the null hypothesis and hence, Proportion for Hep B immunization is same for Developed and Developing Countries

# ### Test-7
# 
# We have divided Alcohol rate into 3 groups. Below 6, between 6 and 12 and less than 12. <b>We are testing the claim that Alcohol depends on country status.</b>

# In[148]:


print('Developed')
print('Below 6: ',df[(df['Status']=='Developed') & (df['Alcohol']<6)].shape[0])
print('Between 6 and 12: ',df[(df['Status']=='Developed') & ((df['Alcohol']>6) & (df['Alcohol']<12))].shape[0])
print('Above 12: ',df[(df['Status']=='Developed') & (df['Alcohol']>12)].shape[0])
print('\nDeveloping')
print('Below 6: ',df[(df['Status']=='Developing') & (df['Alcohol']<6)].shape[0])
print('Between 6 and 12: ',df[(df['Status']=='Developing') & ((df['Alcohol']>6) & (df['Alcohol']<12))].shape[0])
print('Above 12: ',df[(df['Status']=='Developing') & (df['Alcohol']>12)].shape[0])


# | Country Status  | Alcohol below 6 | Alcohol b/w 6 and 12 | Alcohol above 12 |
# | ------- | ---- | --------- | -------- |
# | Developed    | 14 | 295 | 80 |
# | Developing  | 1330 | 366 | 29 |

# H0: Relationship exists between country status and alcohol level<br>
# H1: Relationship does not exists between country status and alcohol level

# In[149]:


quality_array = np.array([[14, 295, 80],[1330, 366, 29]])
chi_sq_Stat, p_value, deg_freedom, exp_freq = st.chi2_contingency(quality_array)
chi_sq_Stat, p_value, deg_freedom, arr = st.chi2_contingency(quality_array)
print('Chi-square statistic %3.5f P value %1.6f Degrees of freedom %d' %(chi_sq_Stat, p_value,deg_freedom))


# <b>P-value is less than alpha(0.05)</b> so we reject null hypothesis. Hence, Relationship does not exists between country status and alcohol level.

# ### Central Limit Theorem Simulation

# In[150]:


series1 = df['Alcohol'].dropna()
print(st.shapiro(series1))

def sample_mean_calculater(population,sample_size):
    sample_means =[]
    for i in range(100):
        sample = np.random.choice(population,size = sample_size,replace= True)
        mean = sample.mean()
        sample_means.append(mean)
    
    return sample_means

sns.distplot(series1)
list1 = [5,10,15,25,30,50,80,100,150,200,350, 500]
fig, axes = plt.subplots(6, 2, figsize=(15,25))
axes = [ax for axes_rows in axes for ax in axes_rows]
    
for i, l in enumerate(list1):
    figure = sns.distplot(sample_mean_calculater(series1,l), ax=axes[i])
    figure.set_xlabel('sample size: N=%i' %l)
    figure.set_ylabel('Distribution of sample means')


# # Summary
# 1. The half of the population has life expectancy less than 71 years.
# 2. Life expectancy for developed countries and developing countries is different.
# 3. Top 5 developed countries life expectancy is different.
# 4. Population depends on the year but not on the status of the country. 
# 5. Total expenditure in developed countries is more as compared to developing countries.
# 6. The proportion of Hepatitis-B immunization in developed countries and developing countries is same.
# 7. The alcohol level consumption rate is not dependent on country status.

# In[ ]:




