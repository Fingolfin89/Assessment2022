
# coding: utf-8

# In[1]:


import pandas as pd
import os
from datetime import datetime, timezone
import datetime as dt
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# # Read Data

# In[2]:


os.chdir("C:/Users/Θανάσης/Desktop/Efood")


# In[3]:


data = pd.read_csv('Assessment exercise dataset - orders.csv')


# In[4]:


data.head() 


# # Data Preprocessing

# ### Fix the type of timestamp variable

# In[5]:


data.dtypes


# In[6]:


data['order_timestamp'] = pd.to_datetime(data['order_timestamp'])


# In[7]:


data['order_timestamp'] = data['order_timestamp'].dt.tz_localize(None) 


# ##### Add day names

# In[8]:


my_date = data['order_timestamp']
day = pd.to_datetime(my_date).dt.day_name()
data['day'] = day
data.head()


# In[9]:


data.dtypes


# ### Check unique values of key columns

# In[10]:


data.user_id.nunique()


# In[11]:


data.city.nunique()


# In[12]:


data.order_id.nunique()


# In[13]:


def unique_counts(data):
   for i in data.columns:
       count = data[i].nunique()
       print(i, ": ", count)
unique_counts(data)


# ### Drop Duplicates if needed

# In[14]:


city_user_order=data[['city','user_id','order_id']].drop_duplicates()


# ### Check top cities and users per order

# In[16]:


city_user_order.groupby(['user_id'])['order_id'].aggregate('count').reset_index().sort_values('order_id', ascending=False).head()


# In[17]:


city_user_order.groupby(['city',])['order_id','user_id'].aggregate('nunique').reset_index().sort_values('order_id', ascending=False).head()


# ### Check for Null values

# In[18]:


data.isnull().sum(axis=0)


# # RFM Customer Segmentation

# In[19]:


data['order_timestamp'].max()


# In[21]:


next_date = dt.datetime(2022,2,1)
next_date


# ### Create a RFM table

# In[22]:


rfmTable = data.groupby('user_id').agg({'order_timestamp': lambda x: (next_date - x.max()).days, 'order_id': lambda x: len(x), 'amount': lambda x: x.sum()})
rfmTable['order_timestamp'] = rfmTable['order_timestamp'].astype(int)
rfmTable.rename(columns={'order_timestamp': 'recency', 
                         'order_id': 'frequency', 
                         'amount': 'monetary_value'}, inplace=True)


# In[23]:


rfmTable.head()


# ### Let’s check the details of the customer with the highest monetary value

# In[24]:


rfmTable.sort_values(by=['monetary_value'], ascending=False).head()


# In[25]:


customer1 = data[data['user_id'] == 485537911656]
customer1


# In[26]:


customer1["amount"].mean()


# ### Split the metrics

# In[27]:


quantiles = rfmTable.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()


# In[28]:


segmented_rfm = rfmTable


# ### The lowest recency, highest frequency and monetary amounts are our best customers.

# In[29]:


def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1


# ### Add segment numbers to the newly created segmented RFM table

# In[30]:


segmented_rfm['r_quartile'] = segmented_rfm['recency'].apply(RScore, args=('recency',quantiles,))
segmented_rfm['f_quartile'] = segmented_rfm['frequency'].apply(FMScore, args=('frequency',quantiles,))
segmented_rfm['m_quartile'] = segmented_rfm['monetary_value'].apply(FMScore, args=('monetary_value',quantiles,))
segmented_rfm.head()


# ### Add a new column to combine RFM score: 111 is the highest score as we determined earlier.

# In[31]:


segmented_rfm['RFMScore'] = segmented_rfm.r_quartile.map(str) + segmented_rfm.f_quartile.map(str) + segmented_rfm.m_quartile.map(str)

segmented_rfm.sort_values(by=['RFMScore','monetary_value'], ascending=True).head(10)


# In[32]:


Top_Customers = segmented_rfm[segmented_rfm['RFMScore']=='111']
Top_Customers.RFMScore.count()


# # Plot

# In[33]:


rfmTable2 = rfmTable.reset_index()
rfmTable2 = rfmTable2[['RFMScore']]
rfmTable2.head()


# In[34]:


grouped_by_rfmscore = rfmTable2.groupby(['RFMScore']).size().reset_index(name = 'count').sort_values('count', ascending=False)
grouped_by_rfmscore.head()


# In[35]:


data_plot = [go.Bar(x=grouped_by_rfmscore['RFMScore'], y=grouped_by_rfmscore['count'])]

layout = go.Layout(
    title=go.layout.Title(
        text='Customer RFM Segments'
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='RFM Segment'
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='Number of Customers'
        )
    )
)

fig = go.Figure(data=data_plot, layout=layout)
iplot(fig, filename='rfm_Segments')


# In[36]:


segmented_rfm.describe()


# # Create Matrix

# In[37]:


segmented_rfm.head()


# In[38]:


BestCustomers = ['111']
BigSpenders = ['331','121','221','321','131','231','341','241','141']
LoyalCustomers = ['212','112','211','312','313','113','213','114','214','314']
Promising = ['222','223','123','122']
Worrying = ['311']
LostBestCustomers = ['411']
LostCustomers = ['422','412','421','432','442','431','441']
LostCheapCustomers = ['433','444','434','424','413','423','443','414']


# In[39]:



def f(row):

    if row['RFMScore'] in BestCustomers:
        val = 'Best Customers'
    elif row['RFMScore'] in BigSpenders:
        val = 'Big Spenders'
    elif row['RFMScore'] in LoyalCustomers:
        val = 'Loyal Customers'
    elif row['RFMScore'] in Promising:
        val = 'Promising'
    elif row['RFMScore'] in Worrying:
        val = 'Worrying'        
    elif row['RFMScore'] in LostBestCustomers:
        val = 'Lost Best Customers'   
    elif row['RFMScore'] in LostCustomers:
        val = 'Lost Customers'
    elif row['RFMScore'] in LostCheapCustomers:
        val = 'Lost Cheap Customers'
    else:
        val = 'Need Attention'
    return val


# In[40]:


segmented_rfm['segment'] = segmented_rfm.apply(f, axis=1)


# In[41]:


segmented_rfm.head()


# In[42]:


grouped_by_segment = segmented_rfm.groupby(['segment']).size().reset_index(name = 'count').sort_values('count', ascending=False)
grouped_by_segment


# In[43]:


ax = grouped_by_segment.plot.barh(x='segment',y='count', align='center')
ax.invert_yaxis()
ax.set_xlabel('Segment')
ax.set_title('Customers per Segment')
plt.show()


# ### Per City

# In[44]:


segmented_rfm.head().reset_index()


# In[45]:


data_new = data.merge(segmented_rfm, how="left",on = 'user_id')


# In[46]:


data_new.head()


# In[47]:


grouped_by_city = data_new.groupby(['city','segment']).agg({'RFMScore': lambda x: x.count()})
grouped_by_city.head()


# # Breakfast

# In[49]:


data_new3 = data_new.drop(['order_id'], axis=1)


# In[50]:


data_new2 = data_new3.drop_duplicates('user_id')


# In[51]:


grouped_by_cuisine = data_new.groupby(['cuisine','segment']).agg({'user_id': lambda x: x.nunique()}).sort_values('user_id', ascending=False)
grouped_by_cuisine = grouped_by_cuisine.reset_index()
grouped_by_cuisine


# In[52]:


plt.figure(figsize=(10, 6))
sns.barplot(x="cuisine", hue="segment", y="user_id", data=grouped_by_cuisine)
plt.show()


# ### Create Frequency pivot

# In[53]:


data_new.pivot_table(values='user_id', index='cuisine', columns='segment', aggfunc=lambda x: len(x.unique()))


# # Association Analysis

# In[55]:


basket = (data_new.groupby(['user_id', 'cuisine'])['order_id']
          .nunique().unstack().reset_index().fillna(0)
          .set_index('user_id'))


# In[56]:


basket.head()


# ### Structure the data

# In[57]:


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
basket_sets.head()


# In[58]:


frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)


# In[59]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules


# #### Support vs Confidence

# In[62]:


#plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
#plt.xlabel('support')
#plt.ylabel('confidence')
#plt.title('Support vs Confidence')
#plt.show()


# #### Support vs Lift

# In[ ]:


#plt.scatter(rules['support'], rules['lift'], alpha=0.5)
#plt.xlabel('support')
#plt.ylabel('lift')
#plt.title('Support vs Lift')
#plt.show()


# #### Lift vs Confidence

# In[ ]:


#fit = np.polyfit(rules['lift'], rules['confidence'], 1)
#fit_fn = np.poly1d(fit)
#plt.plot(rules['lift'], rules['confidence'], 'yo', rules['lift'], 
#fit_fn(rules['lift']))


# In[64]:


rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

# Transform antecedent, consequent, and support columns into matrix
support_table = rules.pivot(index='consequents', columns='antecedents', values='lift')

plt.figure(figsize=(10,6))
sns.heatmap(support_table, annot=True, cbar=False)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.yticks(rotation=0)
plt.show() 


# In[65]:


# Transform antecedent, consequent, and support columns into matrix
support_table = rules.pivot(index='consequents', columns='antecedents', values='support')

plt.figure(figsize=(10,6))
sns.heatmap(support_table, annot=True, cbar=False)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.yticks(rotation=0)
plt.show() 

