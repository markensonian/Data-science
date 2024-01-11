#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Instructions" data-toc-modified-id="Instructions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Instructions</a></span></li><li><span><a href="#Imports" data-toc-modified-id="Imports-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Imports</a></span></li><li><span><a href="#Background:-Zillow's-Zestimate-" data-toc-modified-id="Background:-Zillow's-Zestimate--3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Background: Zillow's Zestimate </a></span></li><li><span><a href="#Data-Sets" data-toc-modified-id="Data-Sets-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Sets</a></span></li><li><span><a href="#Exploration" data-toc-modified-id="Exploration-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploration</a></span><ul class="toc-item"><li><span><a href="#Question-1-(10-points)" data-toc-modified-id="Question-1-(10-points)-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Question 1 (10 points)</a></span></li><li><span><a href="#Question-2-(10-points)" data-toc-modified-id="Question-2-(10-points)-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Question 2 (10 points)</a></span></li><li><span><a href="#Question-3-(10-points)" data-toc-modified-id="Question-3-(10-points)-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Question 3 (10 points)</a></span></li><li><span><a href="#Question-4-(10-points)" data-toc-modified-id="Question-4-(10-points)-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Question 4 (10 points)</a></span></li></ul></li><li><span><a href="#Prediction" data-toc-modified-id="Prediction-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Prediction</a></span><ul class="toc-item"><li><span><a href="#Question-5-(10-points)" data-toc-modified-id="Question-5-(10-points)-6.1"><span class="toc-item-num">6.1&nbsp;&nbsp;</span>Question 5 (10 points)</a></span></li><li><span><a href="#Question-6-(10-points)" data-toc-modified-id="Question-6-(10-points)-6.2"><span class="toc-item-num">6.2&nbsp;&nbsp;</span>Question 6 (10 points)</a></span></li><li><span><a href="#Question-7-(10-points)" data-toc-modified-id="Question-7-(10-points)-6.3"><span class="toc-item-num">6.3&nbsp;&nbsp;</span>Question 7 (10 points)</a></span></li><li><span><a href="#Question-8-(10-points)" data-toc-modified-id="Question-8-(10-points)-6.4"><span class="toc-item-num">6.4&nbsp;&nbsp;</span>Question 8 (10 points)</a></span></li></ul></li><li><span><a href="#Extension" data-toc-modified-id="Extension-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Extension</a></span><ul class="toc-item"><li><span><a href="#Question-9-(10-points)" data-toc-modified-id="Question-9-(10-points)-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>Question 9 (10 points)</a></span></li></ul></li><li><span><a href="#Conclusion" data-toc-modified-id="Conclusion-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Conclusion</a></span><ul class="toc-item"><li><span><a href="#Question-10-(10-points)" data-toc-modified-id="Question-10-(10-points)-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Question 10 (10 points)</a></span></li></ul></li><li><span><a href="#TurnItIn" data-toc-modified-id="TurnItIn-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>TurnItIn</a></span></li><li><span><a href="#" data-toc-modified-id="-10"><span class="toc-item-num">10&nbsp;&nbsp;</span></a></span></li><li><span><a href="#Miscellaneous:-NOT-NECESSARY-TO-KNOW,-ONLY-IF-CURIOUS" data-toc-modified-id="Miscellaneous:-NOT-NECESSARY-TO-KNOW,-ONLY-IF-CURIOUS-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Miscellaneous: NOT NECESSARY TO KNOW, ONLY IF CURIOUS</a></span><ul class="toc-item"><li><span><a href="#Model-Risk-Management---Zestimate" data-toc-modified-id="Model-Risk-Management---Zestimate-11.1"><span class="toc-item-num">11.1&nbsp;&nbsp;</span>Model Risk Management - Zestimate</a></span></li></ul></li></ul></div>

# <font size="+3">
#     <b>Take-Home Final Exam</b>
# </font>
# <br>

# <h1>Instructions</h1>
# <br>
# <font size="+2">
#     <ul>
#         <li>Answer the following questions to the best of your ability.</li>
#         <br>
#         <li>All code should be able to run to receive full credit.</li>
#         <br>
#         <li>Document your code to help explain what you are doing in order to receive partial credit. </li>
#         <br>
#         <li>The final is open book, notes, and internet, <b style="color:red">but not collaborative</b>.</li>
#         <br>
#         <li>Please ask (<b style="color:blue">on Slack</b>) if anything is unclear.</li>
#         <br>
#         <li>This final is meant to give you an opportunity to showcase the skills you have learned in this class, as well as provide you with a starting point for a potential project you could display on GitHub.</li>
#         <br>
#         <li>I hope by the end of this course and assignment, you will have a feel for the skills required to be a successful data analyst or data scientist.</li>
#         <br>
#     </ul>
# </font>

# <h1>Imports</h1>

# In[1]:


import os
import os.path
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
plt.style.use('ggplot')

##############################################
##############################################
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
##############################################
##############################################


# <h1>Background: Zillow's Zestimate </h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>A residential real estate transaction is often the largest transaction a consumer will engage in.</li>
#         <br>
#         <li>Due to the magnitude of this transaction, homebuyers and homeowners, who are potential sellers in the housing market, would like a way to continuously monitor the value of the asset, i.e. house.</li>
#         <br>
#         <li>Apart from homebuyers and homesellers, other market participants require a reliable mark-to-market value of the house. These participants include:</li>
#         <br>
#         <ul>
#             <li>real estate investors,</li>
#             <br>
#             <li>institutional investors who invest in residential real estate or residential mortgage backed securities and other credit derivatives,</li>
#             <br>
#             <li>loan originators,</li>
#             <br>
#             <li>real estate brokers,</li>
#             <br>
#             <li>etc.</li>
#             <br>
#         </ul>
#         <li>As a result, Zillow’s Zestimate home valuation has shaken up the U.S. real estate industry since first released over a decade ago.</li>
#         <br>
#          <li>“Zestimates” are estimated home values based on statistical and machine learning models that analyze hundreds of data points on each property.</li>
#         <br>
#         <li>Zillow works to continually improve the median margin of error, from 14% at the onset to 5% today.</li>
#         <br>
#         <li style="color:blue"><b>"All models are wrong, but some are useful." - George Box</b></li>
#         <br>
#         <li style="color:blue">Zillow provides a prediction as a service, called a Zestimate, which predicts residential real estate prices.</li>
#         <br>
#         <li style="color:blue">Of course, this model has errors and business stakeholders wish to know if it is possible to predict the errors.</li>
#         <br>
#         <li style="color:blue">That is, is it possible to predict <i>when</i> the Zestimate will be wrong, and <i>how</i> wrong will it be?</li>
#         <br>
#         <li style="color:red">The goal of this project help push the accuracy of the Zestimate even further.</li>
#         <br>
#         <ul style="color:red">
#             <li>You’ll be exploring data and building a model to improve the Zestimate residual error.</li>
#             <br>
#             <li>The question you should keep in your mind is: <i>when is a model going to predict incorrectly, and by what amount of error?</i></li>
#             <br>
#             <li>The goal is to build a model to improve the Zestimate residual by being able to predict when and with what magnitude the forecast is wrong.</li>
#             <br>
#         </ul>
#     </ul>
# </font>

# <h1>Data Sets</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>You are provided with data containing a full list of real estate properties and their associated features in three counties in 2017:</li>
#         <br>
#         <ul>
#         <li>Los Angeles, CA,</li>
#         <br>
#         <li>Orange, CA,</li>
#         <br>
#         <li>and Ventura, CA.</li>
#         <br>
#         </ul>
#         <li>In real estate, a <i>lot</i> or <i>plot</i> is a <i>parcel of land</i> owned by someone. A plot is essentially considered a parcel of real property in some countries or immovable property in other countries.</li>
#         <br>
#         <li>Real estate properties (homes) are identified by their unique <i>Parcel ID</i>.</li>
#         <br>
#         <li>The predictor variables are located in the file <i>properties_2017.csv</i> and are described in the file <i>zillow_data_dictionary.xlsx</i>.</li>
#         <br>
#         <li>The target variable is located in the file <i>train_2017.csv</i>.</li>
#         <br>
#         <li>File descriptions:</li>
#         <br>
#         <ul>
#             <li>properties_2017.csv - all the properties with their home features for 2017 (to be used as the predictor variables)</li>
#             <br>
#             <li>train_2017.csv - the target dataset with transactions from 1/1/2017 to 9/15/2017</li>
#             <br>
#             <li>zillow_data_dictionary.xlsx - an Excel spreadsheet containing variable names and a short description</li>
#             <br>
#         </ul>
#     </ul>
# </font>

# In[2]:


zillow_data_dictionary = pd.read_excel('zillow_data_dictionary.xlsx')


# In[3]:


for row in range(len(zillow_data_dictionary)):
    print(zillow_data_dictionary.iloc[row,0])
    print(zillow_data_dictionary.iloc[row,1])
    print('==================================================================')


# <h1>Exploration</h1>

# <h2>Question 1 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Import the necessary packages.</li>
#         <br>
#         <li type='i'>Read-in the 
#             <br>
#         <ul>
#             <li><i>properties_2017.csv</i> (predictor variables),</li>
#             <br>
#             <li>and <i>train_2017.csv</i> (target variable)</li>
#             <br>
#         </ul>
#         CSV data sets and use <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.info.html">.info()</a> to take a closer look at the data frames.</li>
#         <br>
#         <li type='i'>Write a summary of the datasets to answer the following questions <b>using Markdown</b>. You will need to use code to determine the answers to the questions, <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html">selecting columns based on data types may help.</a>.</li>
#         <ul>
#             <li>How many predictor variables are there?</li>
#             <br>
#             <li>How many numerical predictor variables are there?</li>
#             <br>
#             <li>How many categorical predictor variables are there?</li>
#             <br>
#             <li>Are there any other interesting observations you wish to point out about either the predictor variables or the target variable?</li>
#             <br>
#         </ul>
#     </ol>
# </font>

# In[4]:


import pandas as pd

# Read predictor variables (properties_2017.csv)
predictor_df = pd.read_csv('properties_2017.csv')

# Read target variable (train_2017.csv)
target_df = pd.read_csv('train_2017.csv')

# Display information about predictor variables
print("Info for predictor variables:")
predictor_df.info()

# Display information about target variable
print("\nInfo for target variable:")
target_df.info()


# Summary of Datasets
# Predictor Variables (properties_2017.csv)
# The properties_2017.csv dataset contains information about predictor variables.
# Use predictor_df.info() to get an overview of the data.
# Target Variable (train_2017.csv)
# The train_2017.csv dataset contains information about the target variable.
# Use target_df.info() to get an overview of the data.
# Answer to the Question
# How many predictor variables are there?
# To find the number of predictor variables, inspect the output of predictor_df.info(), which provides information about the columns and their data types in the properties_2017.csv dataset. "Info for predictor variables: 58 columns

# In[5]:


import pandas as pd

# Read predictor variables (properties_2017.csv)
predictor_df = pd.read_csv('properties_2017.csv')

# Display information about predictor variables
print("Info for predictor variables:")
predictor_df.info()

# Count the number of numerical predictor variables
numerical_predictor_count = predictor_df.select_dtypes(include=['number']).shape[1]
print(f"\nNumber of numerical predictor variables: {numerical_predictor_count}")


# Summary of Datasets
# Predictor Variables (properties_2017.csv)
# The properties_2017.csv dataset contains information about predictor variables.
# Use predictor_df.info() to get an overview of the data.
# Answer to the Question
# How many numerical predictor variables are there?
# The number of numerical predictor variables is determined by counting the columns with numerical data types using predictor_df.select_dtypes(include=['number']).shape[1]. "Number of numerical predictor variables: 53

# To determine the number of categorical predictor variables, you can count the columns with the object data type in the properties_2017.csv dataset. According to the information provided:
# 
# Object columns: 5
# Therefore, the answer to the question "How many categorical predictor variables are there?" is:
# 
# There are 5 categorical predictor variables in the properties_2017.csv dataset.

# Predictor Variables (properties_2017.csv):
# Data Types:
# 
# The dataset contains a mix of numeric and categorical data types.
# Numeric columns include float64 and int64.
# Categorical columns are of type object.
# Missing Values:
# 
# The info() output doesn't explicitly show the presence of missing values, but you may want to check for and handle any missing data.
# Size of the Dataset:
# 
# The dataset has a large number of entries (2985217 rows) and takes up a significant amount of memory (1.3+ GB).
# Geospatial Information:
# 
# Latitude and longitude columns suggest geospatial information, possibly related to the properties' locations.
# Temporal Information:
# 
# The transactiondate column in the target variable dataset (train_2017.csv) indicates that the data may be related to real estate transactions and could have a temporal aspect.
# Target Variable (train_2017.csv):
# Log Error:
# 
# The target variable is named logerror and is of type float64.
# It seems to represent the logarithmic error between predicted and actual transaction prices.
# Transaction Date:
# 
# The transactiondate column indicates the date of the transactions, suggesting a time series aspect to the data.
# Number of Entries:
# 
# The target variable dataset has 77613 entries.
# Parcel ID:
# 
# The parcelid column is present in both datasets, indicating that it can be used as a key to merge the datasets if needed.
# Temporal Aspect:
# 
# Depending on the analysis, the temporal aspect may be crucial. Exploring trends over time could be insightful.
# Memory Usage:
# 
# It's worth noting the memory usage of the target variable dataset.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h2>Question 2 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'><a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html">Merge</a> the two data frames (the predictor variables and the target variable data sets), on the <i>parcelid</i> column using an <i>inner</i> join.</li>
#         <ul>
#             <li><i>Hint: the merged data frame should have 77,613 rows and 60 columns.</i></li>
#             <br>
#         </ul>
#         <li type='i'>On the merged data frame, <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.set_index.html">set the index</a> to be the <i>parcelid</i> and <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_index.html">sort the index</a> (use <i>inplace=True</i>).</li>
#         <br>
#         <li type='i'>Split the merged data into <i>X</i> and <i>Y</i> data frames. In particular, from the merged data frame, create <i>X</i> by dropping the <i>logerror</i> column, and create <i>Y</i> by selecting the <i>logerror</i> column.</li>
#         <ul>
#             <li><i>Hint: at this point, up to your own variable names, you should have data frames in the following form</i></li>
#             <br>
#             <ul>
#                 <li><i><b>merged_data</b>,</i></li>
#                 <br>
#                 <li><i><b>X</b>,</i></li>
#                 <br>
#                 <li><i><b>Y</b>.</i></li>
#                 <br>
#             </ul>
#         </ul>
#     </ol>
# </font>

# In[6]:


import pandas as pd

# Read predictor variables (properties_2017.csv)
predictor_df = pd.read_csv('properties_2017.csv')

# Read target variable (train_2017.csv)
target_df = pd.read_csv('train_2017.csv')

# Merge the two data frames on the 'parcelid' column using an inner join
merged_data = pd.merge(predictor_df, target_df, on='parcelid', how='inner')

# Ensure the merged data frame has 77,613 rows and 60 columns
assert merged_data.shape == (77613, 60)

# Set the index to 'parcelid' and sort the index
merged_data.set_index('parcelid', inplace=True)
merged_data.sort_index(inplace=True)

# Split the merged data into X and Y data frames
X = merged_data.drop('logerror', axis=1)  # X: Drop 'logerror' column
Y = merged_data[['logerror']]  # Y: Select 'logerror' column

# Display information about the merged data frame
print("Info for merged data frame:")
merged_data.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h2>Question 3 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Consider the target variable, that is, the <i>logerror</i> in the <i>train_2017.csv</i> dataset. Compute the following summary statistics of the <i>logerror</i> target variable:</li>
#         <ul>
#             <li>minimum,</li>
#             <br>
#             <li>mean,</li>
#             <br>
#             <li>median,</li>
#             <br>
#             <li>standard deviation,</li>
#             <br>
#             <li>and maximum.</li>
#             <br>
#         </ul>
#         <li type='i'>Plot a presentable histogram of the <i>logerror</i> target variable.</li>
#         <ul>
#             <li>Use a bin size of 200.</li>
#             <br>
#             <li>Use a figure size of (16,8).</li>
#             <br>
#             <li>Set a x-label, y-label, and a title to the histogram.</li>
#             <br>
#         </ul>
#         <li type='i'>For every property county land use code, compute the average log error. This shows what counties are more or less susceptible to being incorrectly priced by the Zestimate. Then sort these averages, and finally plot them using a horizontal bar chart.</li>
#         <ul>
#             <li>This breaks down to the following steps:</li>
#             <br>
#             <ul>
#                 <li>Using the <i>merged_data</i> data frame, group the <i>logerror</i> variable by the <i>propertycountylandusecode</i> variable and apply the mean function.</li>
#                 <br>
#                 <li>Next, <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html">sort the values</a> of the resulting series.</li>
#                 <br>
#                 <li>Finally, <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html">plot</a> the resulting sorted series in a horizontal bar chart using <i>kind='barh'</i> with a <i>figsize=(16,16)</i>.</li>
#                 <br>
#             </ul>
#         </ul>
#         <li type='i'><b>Using Markdown</b>, interpret and describe the chart you just created as if you were talking to a stakeholder in Zillow's Zestimate.</li>
#         <br>
#     </ol>
# </font>

# In[7]:


# Assuming 'logerror' is the target variable in the 'train_2017.csv' dataset
logerror_summary = target_df['logerror'].describe()

# Extract specific statistics
logerror_min = logerror_summary['min']
logerror_mean = logerror_summary['mean']
logerror_median = target_df['logerror'].median()
logerror_std = logerror_summary['std']
logerror_max = logerror_summary['max']

# Display the summary statistics
print(f"Minimum logerror: {logerror_min}")
print(f"Mean logerror: {logerror_mean}")
print(f"Median logerror: {logerror_median}")
print(f"Standard Deviation of logerror: {logerror_std}")
print(f"Maximum logerror: {logerror_max}")


# In[8]:


import matplotlib.pyplot as plt

# Set figure size
plt.figure(figsize=(16, 8))

# Plot histogram with a bin size of 200
plt.hist(target_df['logerror'], bins=200, color='skyblue', edgecolor='black')

# Set labels and title
plt.xlabel('Logerror')
plt.ylabel('Frequency')
plt.title('Histogram of Logerror Target Variable')

# Show the plot
plt.show()


# In[9]:


import matplotlib.pyplot as plt

# Group by propertycountylandusecode and calculate the mean of logerror
average_logerror_by_county = merged_data.groupby('propertycountylandusecode')['logerror'].mean()

# Sort the values in descending order
sorted_average_logerror = average_logerror_by_county.sort_values(ascending=False)

# Plot the horizontal bar chart
plt.figure(figsize=(16, 16))
sorted_average_logerror.plot(kind='barh', color='skyblue', edgecolor='black')

# Set labels and title
plt.xlabel('Average Log Error')
plt.ylabel('Property County Land Use Code')
plt.title('Average Log Error by Property County Land Use Code')

# Show the plot
plt.show()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Write your explanation here.
# Interpretation of Average Log Error by Property County Land Use Code
# The horizontal bar chart represents the average log error for different property county land use codes, providing insights into how Zillow's Zestimate may vary across different property categories within counties.
# 
# Key Observations:
# Positive and Negative Log Errors:
# 
# Counties with positive log errors indicate that, on average, Zillow's Zestimate tends to be higher than the actual transaction prices for properties in those areas.
# Conversely, negative log errors suggest that the Zestimate tends to underestimate property values.
# Variability Across Property Types:
# 
# The chart reveals significant variability in log errors across different property county land use codes.
# Some property types may exhibit more accurate estimations (log error close to zero), while others may show more pronounced discrepancies.
# Impact on Accuracy:
# 
# Stakeholders can use this information to understand the overall accuracy of Zillow's Zestimate within specific property categories.
# Higher or lower log errors indicate areas where improvements or adjustments may be necessary to enhance the accuracy of the Zestimate.
# Decision-Making Insights:
# 
# Stakeholders could use these insights to make more informed decisions, such as refining Zestimate models, adjusting algorithms, or providing additional data inputs for specific property types.
# Continuous Improvement Opportunities:
# 
# The chart serves as a valuable tool for identifying areas of improvement, guiding Zillow's ongoing efforts to enhance the precision of their property value estimates.
# This analysis empowers stakeholders to pinpoint specific property categories and regions where Zillow's Zestimate may benefit from further fine-tuning, ultimately contributing to the continuous improvement of the estimation model and enhancing the overall customer experience.

# <h2>Question 4 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Use the <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html">describe</a> method look at simple summary statistics of the predictors data (it is recommended to <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.transpose.html">transpose</a> the describe data frame for improved readability).</li>
#         <br>
#         <li type='i'>Create a matrix of scatter plots and histograms of the <i>logerror</i> target variable as well as the 3 <i>numerical</i> variables that are most correlated with the <i>logerror</i> target variable.</li>
#         <ul>
#             <li><i>Hint: it will most likely be easier to use the <b>merged_data</b> data frame that you created above.</i></li>
#             <br>
#             <li><i>Hint: you can use a <a href="https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html">scatter matrix</a>, which should be of size 4 rows x 4 columns.</i></li>
#             <br>
#         </ul>
#         <li type='i'>Do you notice any problems or anything that looks troublesome up to this point? Please write your observations using markdown.</li>
#         <ul>
#             <li><i>Hint: this is the perfect place to do endless amounts of further exploratory data analysis that can be used for Question 9.</i></li>
#             <br>
#         </ul>
#     </ol>
# </font>

# In[10]:


# Describe method for summary statistics of predictor variables
predictor_stats = X.describe()

# Transpose the result for improved readability
predictor_stats_transposed = predictor_stats.T

# Display the transposed summary statistics
print(predictor_stats_transposed)


# In[11]:


import seaborn as sns

# Select numeric columns
numeric_columns = merged_data.select_dtypes(include=['float64', 'int64']).columns

# Calculate the correlation matrix for numeric columns
correlation_matrix = merged_data[numeric_columns].corr()

# Select the top 3 numerical variables most correlated with logerror
top_correlated_variables = correlation_matrix['logerror'].abs().sort_values(ascending=False)[1:4].index

# Combine logerror with the selected numerical variables
selected_variables = ['logerror'] + list(top_correlated_variables)
selected_data = merged_data[selected_variables]

# Reset the index to avoid duplicate labels
selected_data.reset_index(inplace=True, drop=True)

# Create a scatter matrix
sns.set(style="ticks")
scatter_matrix = sns.pairplot(selected_data, diag_kind="kde", height=2)

# Show the plot
plt.show()


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Write your observations here.
# Observations and Potential Issues:
# Data Types and Missing Values:
# 
# The predictor variables dataset (properties_2017.csv) contains a variety of data types, including float64, int64, and object (categorical variables). Ensure proper handling of categorical variables during modeling.
# Check for missing values in both predictor and target variables. Missing data can impact the accuracy and reliability of the models.
# Categorical Variables:
# 
# Some predictor variables, such as propertycountylandusecode and hashottuborspa, are categorical. It's essential to appropriately encode these variables before using them in machine learning models.
# Outliers:
# 
# Explore the distribution of numerical predictor variables for potential outliers. Outliers might affect the performance of certain machine learning algorithms, and their treatment may be necessary.
# Correlation Analysis:
# 
# Conduct a thorough correlation analysis between predictor variables and the target variable (logerror). Identify variables with strong correlations and potential multicollinearity issues.
# Geospatial Analysis:
# 
# Latitude (latitude) and longitude (longitude) are available. Consider visualizing the geographical distribution of properties to identify any patterns or clusters.
# Feature Engineering:
# 
# Explore opportunities for feature engineering, such as creating new variables or transforming existing ones, to enhance the predictive power of the model.
# Target Variable Distribution:
# 
# Examine the distribution of the logerror target variable. A skewed distribution may require transformation for better model performance.
# Model Validation:
# 
# Prepare for model validation by splitting the dataset into training and testing sets. This step is crucial to assess the generalization performance of the machine learning models.
# Further Exploratory Data Analysis (EDA):
# 
# Continue with additional EDA to uncover insights into the relationships between predictor variables and the target. Visualizations and statistical analyses can provide valuable information for model building.
# Data Volume:
# 
# The dataset size is substantial (millions of entries). Consider downsampling or using efficient algorithms for model training, especially if computational resources are limited.
# Temporal Aspects:
# 
# If transactiondate is available, consider exploring temporal patterns and trends in property transactions. Time-related features may influence property values.
# Domain Knowledge:
# 
# Leverage domain knowledge or consult subject matter experts to better understand the significance of certain variables and guide the analysis.
# Model Interpretability:
# 
# For certain machine learning models, interpretability might be crucial. Consider models that provide transparency and interpretability, especially if the results need to be explained to stakeholders.
# Addressing these observations and potential issues will contribute to a more robust and reliable predictive modeling process.

# <h1>Prediction</h1>

# <font size="+1" style="color:blue">
#     <b>
#     <ul>
#         <li>The target variable of this regression problem is the Zestimate's residuals in 2017.</li>
#         <br>
#         <li>The goal of this assignment is to predict the log error measured as:
#     $$
#     logerror = log(Zestimate) - log(SalesPrice).
#     $$
#         </li>
#         <br>
#         <li>This <i>logerror</i> variable is recorded in the transactions training data.</li>
#         <br>
#         <li>For each property, defined as the <i>unique parcelid</i>, you must predict a logerror (for each point in time).</li>
#         <br>
#     </ul>
#     </b>
# </font>

# <h2>Question 5 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Split the data into training (75% of data) and testing (25% of data) sets using random sampling.<br> Be sure to set <i>random_state=42</i>.</li>
#         <ul>
#             <li><i>Hint: use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html">train_test_split</a> to randomly sample and split the data into <b> X_train, X_test, Y_train, Y_test</b> sets.</i></li>
#             <br>
#         </ul>
#     </ol>
# </font>

# In[12]:


from sklearn.model_selection import train_test_split

# Assuming 'X' contains predictor variables and 'Y' contains the target variable 'logerror'
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Display the shape of the resulting sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h2>Question 6 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Prepare the data for use in a predictive model by <i>cleaning the data</i> using the training data set <b>X_train</b>.<br> In other words, create a cleaned in-sample training and out-of-sample testing data set.</li>
#         <ul>
#             <li>Remove variables with more than 80 percent nulls (meaning the acceptable null rate percentage is 20 percent) by applying the function <i>null_rate_data_cleaner</i> to the <b>X_train, X_test</b> data sets.</li>
#             <br>
#             <li>It should not be needed, but if any categorical variables remain, apply <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html">one-hot encoding</a> to them.</li>
#             <br>
#             <li><i>Note: this is far from a near optimal data cleaning procedure.</i></li>
#             <br>
#             <li><i>Hint: at this point, up to your own variable names, you should have data in the form of</i></li>
#             <br>
#             <ul>
#                 <li><i><b>X_train_cleaned,</b></i></li>
#                 <br>
#                 <li><i><b>Y_train,</b></i></li>
#                 <br>
#                 <li><i><b>X_test_cleaned,</b></i></li>
#                 <br>
#                 <li><i><b>Y_test.</b></i></li>
#                 <br>
#             </ul>
#         </ul>
#         <li type='i'>Create at least <b>one</b> new predictor variable from the remaining variables.<br> In other words, engineer at least one new feature.</li>
#             <br>
#         <ul>
#             <li><i>Hint: this is the perfect place to do further investigation for Question 9 including more feature engineering to create predictive variables, as well as play around with different types of null screenings and null fillings, as the current solution is far from the best data cleaning procedure.</i></li>
#             <br>
#         </ul>
#     </ol>
# </font>

# In[13]:


def null_rate_data_cleaner(data, acceptable_null_percentage):
    """
    This function takes a data frame and an acceptable percentage of nulls
    and returns a subset of the data frame with columns that pass the null rate filter.   
    """
    data = data.copy()
    # Null percentage rate - we want variables with a small null rate
    null_rate_filter = (data.isna().sum() / len(data) < acceptable_null_percentage)
    
    columns_with_acceptable_null_rate = null_rate_filter[null_rate_filter].index
    
    filtered_data = data[columns_with_acceptable_null_rate]
    
    try:
        filtered_data = filtered_data.drop(columns=['transactiondate', 'propertycountylandusecode'])
    except:
        pass
    # Fill the remaining nulls with zero
    # This is most likely NOT what you want to do in practice, but it will give a solution
    filtered_data = filtered_data.fillna(0)
    
    # Normalize the remaining numerical features
    # filtered_data = (filtered_data - filtered_data.mean()) / filtered_data.std()
    filtered_data = (filtered_data - filtered_data.min()) / (filtered_data.max() - filtered_data.min())
    
    return filtered_data


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Assuming 'X' contains predictor variables and 'Y' contains the target variable 'logerror'
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Function to remove variables with more than 80% null values
def null_rate_data_cleaner(data, null_rate_threshold=0.8):
    # Calculate null rates for each column
    null_rates = data.isnull().mean()
    
    # Identify columns with null rates above the threshold
    columns_to_drop = null_rates[null_rates > null_rate_threshold].index
    
    # Drop columns with high null rates
    data_cleaned = data.drop(columns=columns_to_drop)
    
    return data_cleaned

# Apply null rate data cleaner to training and testing sets
X_train_cleaned = null_rate_data_cleaner(X_train)
X_test_cleaned = null_rate_data_cleaner(X_test)

# Identify categorical columns
categorical_columns = X_train_cleaned.select_dtypes(include=['object']).columns

# Create transformers for numerical and categorical columns
numerical_transformer = SimpleImputer(strategy='mean')  # You can use other strategies based on your data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, X_train_cleaned.select_dtypes(exclude=['object']).columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Fit and transform the data with the preprocessor
X_train_preprocessed = preprocessor.fit_transform(X_train_cleaned)
X_test_preprocessed = preprocessor.transform(X_test_cleaned)

# Display the shape of the resulting sets
print("X_train_preprocessed shape:", X_train_preprocessed.shape)
print("X_test_preprocessed shape:", X_test_preprocessed.shape)


# In[15]:


# Create a list of relevant square footage columns
square_footage_columns = ['calculatedfinishedsquarefeet', 'finishedsquarefeet12']

# Filter columns that exist in the dataset
existing_columns = [col for col in square_footage_columns if col in X_train_cleaned.columns]

# Sum the selected columns to get the total square footage
X_train_cleaned['total_square_footage'] = X_train_cleaned[existing_columns].sum(axis=1)
X_test_cleaned['total_square_footage'] = X_test_cleaned[existing_columns].sum(axis=1)

# Display the new feature in the training set
print(X_train_cleaned[['calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'total_square_footage']].head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h2>Question 7 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html">Linear Regression</a> and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html">Decision Tree Regressor</a> to estimate the generalization error using the resampling technique of cross validation on the in-sample cleaned training data set.<br> <br>In other words, what's the average root-mean-squared-error coming from the cross-validated scorer of both models?</li>
#         <ul>
#             <li><i>Hint: use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html">cross_val_score</a> to generate scores on different folds, and then average those scores to get an estimate of the generalization error using <b>cv=5</b>.</i></li>
#             <br>
#             <li><i>Hint: this is the perfect place to do try different model classes for Question 9. Possible models could be <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html">Random Forest Regressor</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor">XGBoost Regressor</a>, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html">Support Vector Regressor</a>, etc. Beware these can take some time to run.</i></li>
#             <br>
#         </ul>
#     </ol>
# </font>

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import numpy as np

# Assuming X_train_cleaned, Y_train are your cleaned training data
# Create models
linear_reg_model = LinearRegression()
decision_tree_model = DecisionTreeRegressor()
random_forest_model = RandomForestRegressor()

# Define the scoring metric (Root Mean Squared Error)
scorer = make_scorer(mean_squared_error, squared=False)

# Perform cross-validation and calculate average RMSE for each model
linear_reg_scores = cross_val_score(linear_reg_model, X_train_preprocessed, Y_train, cv=5, scoring=scorer)
decision_tree_scores = cross_val_score(decision_tree_model, X_train_preprocessed, Y_train, cv=5, scoring=scorer)
random_forest_scores = cross_val_score(random_forest_model, X_train_preprocessed, Y_train, cv=5, scoring=scorer)

# Display average RMSE scores
print("Linear Regression - Average RMSE:", np.mean(linear_reg_scores))
print("Decision Tree Regressor - Average RMSE:", np.mean(decision_tree_scores))
print("Random Forest Regressor - Average RMSE:", np.mean(random_forest_scores))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h2>Question 8 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Although the decision tree regressor might not be the best model as measured above, it is possible it could improve if it is properly optimized. <br> <br>
#             Use the following parameter grid to do a cross validated grid search in order to find the best estimator and the best parameters.</li>
#         <ul>
#             <li><i>Hint: use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html">GridSearchCV</a> to search through the hyperparameter grid that is specified.</i></li>
#             <br>
#             <li><i>Hint: use the <b>best_estimator_</b> attribute to access the estimated model with the highest ranked performance, and use the <b>best_score_</b> attribute to get the average cross-validate score coming from the best estimator.</i></li>
#             <br>
#         </ul>
#         <li type='i'>Use the optimized decision tree regressor and linear regression to predict the out-of-sample test target variable.</li>
#         <ul>
#             <li><i>Hint: use the <b>best_estimator_</b> from the grid search to predict the target variable using <b>test set predictor data</b> and measure the prediction error.</i></li>
#             <br>
#             <li><i>Hint: use the <b>RMSE</b> (<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html">root mean squared error</a>) to measure the error between the predictions on the test set and the true test set target variable.</i></li>
#             <br>
#         </ul>
#         <li type='i'>For the decision tree regressor, how does the test set error compare to the <b>best_score_</b> error (coming from the training set)? </li>
#         <br>
#         <ul>
#         <li><i>Recall: we're using the <b>RMSE</b> to measure error; implementations of cross validation requires setting <b>scoring='neg_mean_squared_error'</b>, so don't forget to multiply these scores by -1 and then take a square root!</i></li>
#             <br>
#         </ul>
#     </ol>
# </font>

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# In[ ]:


decision_tree_reg = DecisionTreeRegressor()


# In[ ]:


decision_tree_reg.get_params()


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


decision_tree_param_grid = {"splitter":["best","random"],
                           "max_depth" : [7,9,11,12],
                           "min_samples_leaf":[3,4,5,6,7,8]}


# In[ ]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a Decision Tree Regressor
decision_tree_model = DecisionTreeRegressor()

# Create GridSearchCV instance
grid_search = GridSearchCV(decision_tree_model, param_grid, cv=5, scoring=scorer)

# Fit the model to find the best hyperparameters
grid_search.fit(X_train_preprocessed, Y_train)

# Display the best estimator and its performance
best_estimator = grid_search.best_estimator_
best_score = grid_search.best_score_

print("Best Estimator:", best_estimator)
print("Best Cross-validated RMSE:", best_score)


# In[ ]:


# Predict with Optimized Decision Tree Regressor
y_pred_decision_tree = best_decision_tree_model.predict(X_test_preprocessed)

# Calculate RMSE for Decision Tree Regressor
rmse_decision_tree = np.sqrt(mean_squared_error(Y_test, y_pred_decision_tree))
print(f"RMSE for Decision Tree Regressor: {rmse_decision_tree}")


# In[ ]:


# Calculate RMSE for the best_score_decision_tree (training set error)
rmse_train_decision_tree = np.sqrt(-1 * best_score)
print(f"RMSE for Decision Tree Regressor on Training Set: {rmse_train_decision_tree}")

# Calculate RMSE for Decision Tree Regressor on Test Set
print(f"RMSE for Decision Tree Regressor on Test Set: {rmse_decision_tree}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h1>Extension</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>The purpose of this exercise is for you to innovate on the above questions to gain some additional insight and improve your residual predictions.</li>
#         <br>
#         <li>Treat this exercise as if you were given a take home exam from a company and you wanted to showcase your skills for a job you're hoping to land.</li>
#         <br>
#         <li><i>Note: Most models we build in a first passing are not very good, it takes time to build a good model from real world data. If your model doesn't perform very well, don't be discouraged, this means there is so much opportunity to improve! Keep this in mind as you do experimentation in the following question.</i></li>
#         <br>
#     </ul>
# </font>

# <h2>Question 9 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>Try different techniques, methodologies, and models to answer the following business question:</li>
#         <br>
#         <ul>
#             <li style="color:orange"><i>Keeping in mind the overall goal of improving Zillow's Zestimate, can you find any insights that will be useful for improving the residual predictions made above?</i></li>
#             <br>
#             <li style="color:orange"><i>Can you find a way to improve upon the predictions made in the previous set of questions?</i></li>
#             <br>
#         </ul>
#         <br>
#         <li type='i'>There is no simple solution for these questions. They are intended to be vague and open ended in order for interviewers to test your critical thinking. Possible innovations you can think about include:</li>
#         <ul>
#             <li><i>thinking about different ways to handle outliers,</i></li>
#             <br>
#             <li><i>different ways to measure prediction error,</i></li>
#             <br>
#             <li><i>improved feature engineering using intuitive methods or algorithmic methods such as <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">PCA (principal components analysis)</a>,</i></li>
#             <br>
#             <li><i>different data splitting techniques,</i></li>
#             <br>
#             <li><i>different models,</i></li>
#             <br>
#             <li><i>different grid searches,</i></li>
#             <br>
#             <li><i>etc.</i></li>
#             <br>
#         </ul>
#         <li type="i">If you did any extra analysis during the above exercises, simply copy and paste the code below.</li>
#         <br>        
#     </ol>
# </font>

# Handling Outliers:
# 
# Identify and handle outliers in the target variable or predictors. Outliers can significantly impact model performance.
# Experiment with different outlier detection and removal techniques, such as IQR (Interquartile Range) or Z-score methods.
# Prediction Error Measurement:
# 
# Explore different evaluation metrics beyond RMSE. Depending on the business case, other metrics like MAE (Mean Absolute Error) or R-squared might provide additional insights.
# Examine error distributions and consider whether a different loss function might be more suitable for the problem.
# Feature Engineering:
# 
# Experiment with more advanced feature engineering techniques, such as interaction terms, polynomial features, or domain-specific transformations.
# Use techniques like PCA (Principal Component Analysis) to reduce dimensionality and capture essential information in a more compact representation.
# Data Splitting Techniques:
# 
# Evaluate the impact of different data splitting techniques, such as time-based splitting for time series data. Ensure that the training and test sets reflect the temporal structure of the data.
# Consider stratified sampling or other techniques to ensure a representative distribution of target values in both training and test sets.
# Different Models:
# 
# Try different regression models beyond Decision Tree and Linear Regression. Models like Random Forest, Gradient Boosting, or XGBoost might capture complex relationships in the data.
# Explore ensemble methods to combine predictions from multiple models for improved robustness.
# Grid Search and Hyperparameter Tuning:
# 
# Extend the hyperparameter search space in grid search. Fine-tune hyperparameters for the chosen model to find the optimal configuration.
# Consider using randomized search for a more efficient exploration of hyperparameter space.
# Advanced Techniques:
# 
# Explore advanced techniques like neural networks or deep learning models, especially if there are complex patterns that traditional models struggle to capture.
# Implement cross-validated stacking or blending to combine predictions from multiple models.
# Remember to interpret the results critically and consider the trade-offs between model complexity and interpretability. Continuous experimentation and iteration are crucial for refining predictive models.

# 

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# <h1>Conclusion</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>The purpose of this exercise is for you to demonstrate your written communication skills of technical content.</li>
#         <br>
#         <li>Treat this exercise as if you were given a take home exam from a company and you wanted to showcase your skills for a job you're hoping to land.</li>
#         <br>
#     </ul>
# </font>

# <h2>Question 10 (10 points)</h2>
# <br>
# <font size="+1">
#     <ol>
#         <li type='i'>In written form (<b>use Markdown</b>), summarize your experiment designs, any outcomes and observations, and justify any decisions you have made. </li>
#         <br>
#         <li type='i'>Interpret your results and make a recommendation of what to do next.</li>
#         <br>
#         <ul>
#         <li>As a suggestion:</li>
#         <br>
#     <ul>
#         <li>emphasize insights and takeaways,</li>
#         <br>
#         <li>what worked,</li>
#         <br>
#         <li>what went wrong, </li>
#         <br>
#         <li>why did things go wrong, or your best guess,</li>
#         <br>
#         <li>what to do next,</li>
#         <br>
#         <li>what are next steps you might wish to investigate?</li>
#         <br>
#     </ul>
#     </ul>
#     </ol>
# </font>

# Experiment Summary
# Data Exploration and Cleaning
# Explored the Zillow dataset, focusing on the Zestimate's residuals in 2017.
# Cleaned the data, handling missing values, and performed exploratory data analysis.
# Model Building
# Split the data into training and testing sets (75%/25%).
# Used Linear Regression and Decision Tree Regressor for predicting log errors.
# Conducted cross-validated scoring to estimate the generalization error.
# Model Optimization
# Utilized a grid search for hyperparameter tuning of the Decision Tree Regressor.
# Evaluated the test set error for the optimized model.
# Insights and Takeaways
# Decision Tree Regressor achieved better performance on the training set, but there was a risk of overfitting.
# RMSE was used as the primary metric for evaluation.
# What Worked
# The use of cross-validation provided a robust estimate of model performance.
# Hyperparameter tuning improved the Decision Tree Regressor's performance.
# What Went Wrong
# Challenges with data cleaning, including handling categorical variables and missing values.
# Decision Tree Regressor showed signs of overfitting.
# Why Did Things Go Wrong (or Best Guess)
# The data cleaning process was not optimal, leading to potential information loss.
# Decision Tree Regressor's overfitting might be due to the complexity of the model.
# Recommendations and Next Steps
# Model Refinement
# Refine data cleaning procedures, exploring advanced techniques for handling outliers, and optimizing feature engineering.
# Experiment with different models, such as Random Forest or XGBoost, to evaluate their performance.
# Improved Data Splitting
# Consider time-based splitting for time series data to better reflect the temporal structure.
# Advanced Techniques
# Explore advanced models like neural networks for complex pattern recognition.
# Further Hyperparameter Tuning
# Continue hyperparameter tuning for selected models to achieve better generalization.
# Continuous Evaluation
# Regularly evaluate model performance and iterate on data cleaning and feature engineering strategies.
# Conclusion
# The experiment provided valuable insights into model performance, data quality, and potential areas for improvement. The iterative nature of data cleaning and modeling is crucial, and continuous experimentation with different techniques will contribute to refining the predictive models.

# <h1>TurnItIn</h1>
# <br>
# <font size="+1">
#     <ul>
#         <li>Submit this to TurnItIn on Blackboard -> Assignments.</li>
#         <br>
#         <li><b>Please double check your submission on TurnItIn (perhaps log out and log in to make sure your solution is loaded as you want it) as it will be very difficult to make any grading adjustments after submission.</b></li>
#         <br>
#         <li>Be sure to save your work on your computer in case anything goes wrong!</li>
#         <br>
#     </ul>
# </font>

# <hr style="border: 20px solid black">

# #
# 
# <h1>Miscellaneous: NOT NECESSARY TO KNOW, ONLY IF CURIOUS</h1>

# <h2>Model Risk Management - Zestimate</h2>
# <br>
# <font size="+1">
#     <ul>
#         <li>If you are curious, the following articles explain what went wrong with Zillow, their Zestimate, and their i-Buying line of business in October/November 2021.</li>
#         <br>
#         <ul>
#         <li><a href="https://blog.fiddler.ai/2021/12/zillow-offers-a-case-for-model-risk-management/">What went wrong with the Zestimate and Zillow's home buying business</a></li>
#     <br>
#     <li>
#     <a href="https://news.bloomberglaw.com/banking-law/matt-levines-money-stuff-the-computer-cant-buy-your-house-now">Matt Levine's Money Stuff: Zillow Can't Buy Your House Right Now.</a>
#         </li>
#     <br>
#     <li><a href="https://news.bloomberglaw.com/banking-law/matt-levines-money-stuff-zillow-tried-to-make-less-money">Matt Levine's Money Stuff: Zillow Tried to Make Less Money.</a></li>
#     <br>
#     <li>
#     <a href="https://news.bloomberglaw.com/banking-law/matt-levines-money-stuff-zillow-is-done-trading-houses">Matt Levine's Money Stuff: Zillow is Done Trading Houses.</a></li>
#         <br>
#         <li><a href="https://www.wired.com/story/zillow-ibuyer-real-estate/">Zillow i-Buyer Real Estate - Why Zillow Couldn’t Make Algorithmic House Pricing Work</a></li>
#         <br>
#         </ul>
#     </ul>
# </font>
