#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[776]:


import re
import time


# In[640]:


import torch
import torch.nn as nn


# In[642]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[274]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[5]:
"""

import kagglehub

# Download latest version
path = kagglehub.dataset_download("berkanoztas/synthetic-transaction-monitoring-dataset-aml")

print("Path to dataset files:", path)


# In[323]:


df = pd.read_csv("Data/SAML-D.csv")


# In[7]:


df.head()


# In[8]:


df["Sender_bank_location"].value_counts()


# In[9]:


df["Date"].value_counts()


# In[325]:


df = df.drop("Sender_account", axis=1)


# In[327]:


df = df.drop("Receiver_account", axis=1)


# In[12]:


df["Is_laundering"].value_counts()


# In[13]:


df.head()


# In[ ]:





# In[15]:


#sns.barplot(data=df, x="Date", y="Is_laundering")


# In[16]:


df["Is_laundering"].value_counts()


# In[329]:


df = df.sample(frac = 1)


# In[330]:


non_laundering_indices = df[df["Is_laundering"] == 0].index
drop_indices = np.random.choice(non_laundering_indices, size=int(0.9 * len(non_laundering_indices)), replace=False)
df = df.drop(drop_indices)
print(df["Is_laundering"].value_counts(normalize=True))


# In[19]:


df["Is_laundering"].value_counts()


# In[331]:


def datetime_to_month(date):
    match = re.search("^([0-9]+)-([0-9]+)-([0-9]+)$", date)
    if len(match.group(2)) == 2:
        return int(match.group(2))


# In[21]:


df.info()


# In[333]:


df["Month"] = df["Date"].apply(datetime_to_month)


# In[23]:


df.head(5)


# In[24]:


plt.figure(figsize=(12,4))
sns.countplot(data=df, x="Month", hue="Is_laundering")
plt.show()


# In[336]:


for m in df["Month"].unique():
    proportion = (len(df[(df["Month"] == m) & (df["Is_laundering"] == 1)]))/len(df[df["Month"] == m])
    print("proportion: ", proportion)


# In[100]:


#need to actually have this when i do the graph below


# In[102]:


df.info()


# In[106]:


plt.figure(figsize=(12,4))
sns.barplot(data=df, y="Is_laundering", x="Month")
plt.ylim(0, 0.015)
plt.show()


# In[ ]:





# In[112]:


df["Receiver_bank_location"].value_counts()


# In[339]:


for receive_loc in df["Receiver_bank_location"].unique():
    proportion = (len(df[(df["Receiver_bank_location"] == receive_loc) & (df["Is_laundering"] == 1)]))/len(df[df["Receiver_bank_location"] == receive_loc])
    print(f"proportion when receiver location is {receive_loc}: {(proportion*100):.2f}%") 


# In[123]:


df["Sender_bank_location"].value_counts()


# In[341]:


for send_loc in df["Sender_bank_location"].unique():
    proportion = (len(df[(df["Sender_bank_location"] == send_loc) & (df["Is_laundering"] == 1)]))/len(df[df["Sender_bank_location"] == send_loc])
    print(f"proportion when sender location is {send_loc}: {(proportion*100):.2f}%") 


# In[129]:


df.columns


# In[343]:


pivot_table = df.pivot_table(
    index='Sender_bank_location', 
    columns='Receiver_bank_location', 
    values='Is_laundering', 
    aggfunc='mean'
)

plt.figure(figsize=(12, 12))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=0, vmax=0.3)

plt.title('Proportion of Fraudulent Transactions Between Sender and Receiver Countries')
plt.show()


# In[345]:


df.columns


# In[ ]:





# In[347]:


pivot_table = df.pivot_table(
    index='Payment_currency', 
    columns='Received_currency', 
    values='Is_laundering', 
    aggfunc='mean'
)

plt.figure(figsize=(12, 12))
sns.heatmap(pivot_table, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=0, vmax=0.3)

plt.title('Proportion of Fraudulent Transactions Between Sender and Receiver Countries')
plt.show()


# In[181]:


df["Payment_type"].value_counts()


# In[189]:


plt.figure(figsize=(12,4))
sns.barplot(data=df, y="Is_laundering", x="Payment_type")
plt.ylim(0, 0.07)
plt.show()


# In[191]:


df["Time"]


# In[349]:


def time_to_hour(time):
    matches = re.search("^([0-9][0-9]):.*$", time)
    return int(matches.group(1))


# In[351]:


df["Hour"] = df["Time"].apply(time_to_hour)


# In[215]:


plt.figure(figsize=(12,4))
sns.barplot(data=df, y="Is_laundering", x="Hour", order=sorted(df["Hour"].unique()))
plt.ylim(0, 0.017)
plt.show()


# In[224]:


df.columns


# In[ ]:





# In[353]:


df = df.drop(["Time", "Date", "Month"], axis=1)


# In[355]:


df.head(10)


# In[357]:


df["amount_bin"] = pd.qcut(df["Amount"], q=100)  
bin_stats = df.groupby("amount_bin")["Is_laundering"].mean().reset_index()

bin_stats["amount_mid"] = bin_stats["amount_bin"].apply(lambda x: x.mid)

plt.figure(figsize=(10, 5))
sns.lineplot(data=bin_stats, x="amount_mid", y="Is_laundering", marker="o")

plt.xlabel("Transaction Amount")
plt.ylabel("Fraud Rate")
plt.title("Fraud Rate vs Transaction Amount")
plt.xscale("log")
plt.ylim(0, bin_stats["Is_laundering"].max() * 1.1) 
plt.show()


# In[389]:


df = pd.get_dummies(data=df, columns=['Payment_currency', 'Received_currency', 'Sender_bank_location', 'Receiver_bank_location', 'Payment_type'], drop_first=True)


# In[417]:


df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)


# In[439]:


df = df.drop("amount_bin", axis=1)


# In[425]:


y = df["Is_laundering"]
y_type = df["Laundering_type"]


# In[ ]:





# In[427]:


X = df.drop(["Is_laundering", "Laundering_type"], axis=1)


# In[504]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)


# In[506]:


to_drop = y_train[y_train == 0].index
drop_indices = np.random.choice(to_drop, size=int(0.8 * len(to_drop)), replace=False)
X_train = X_train.drop(drop_indices)
y_train = y_train.drop(drop_indices)
print(y_train.value_counts(normalize=True))


# In[508]:


log_model = LogisticRegression()


# In[524]:


param_grid = {"penalty": ["l1", "l2"], "C": np.logspace(0, 3, 8)}


# In[526]:


grid_log_model = GridSearchCV(log_model, param_grid=param_grid, verbose=2)


# In[528]:


scaler = StandardScaler()


# In[530]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[532]:


grid_log_model.fit(scaled_X_train,y_train)


# In[534]:


grid_log_model.best_params_


# In[536]:


y_pred = grid_log_model.predict(scaled_X_test)


# In[538]:


y_pred


# In[540]:


accuracy_score(y_test,y_pred)


# In[542]:


confusion_matrix(y_test,y_pred)


# In[544]:


print(classification_report(y_test,y_pred))


# In[546]:


y_train.sum()


# In[548]:


len(y_train)


# In[550]:


y_pred.sum()


# In[560]:


knn_model = KNeighborsClassifier()
grid_knn_model = GridSearchCV(knn_model, verbose=2, param_grid={"n_neighbors": [2, 3, 5, 8, 11]})


# In[562]:


grid_knn_model.fit(scaled_X_train,y_train)


# In[574]:


y_pred = grid_knn_model.predict(scaled_X_test)


# In[575]:


confusion_matrix(y_test,y_pred)


# In[578]:


print(classification_report(y_test,y_pred))


# In[598]:


svm = SVC()
param_grid = {'C':[0.01,0.1,1],'kernel':['rbf','poly']}
grid_svm_model = GridSearchCV(svm,param_grid, verbose=2)


# In[600]:


grid_svm_model.fit(scaled_X_train, y_train)


# In[602]:


grid_svm_model.best_params_


# In[ ]:


y_pred = grid_svm_model.predict(scaled_X_test)


# In[ ]:


accuracy_score(y_test,y_pred)


# In[ ]:


confusion_matrix(y_test,y_pred)


# In[ ]:


print(classification_report(y_test,y_pred))


# In[604]:


n_estimators=[64,100,128,200]
max_features= [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]
param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             'oob_score':oob_score}


# In[614]:


rfc = RandomForestClassifier()


# In[618]:


grid_rfc_model = GridSearchCV(rfc, param_grid=param_grid, verbose=2)


# In[622]:


grid_rfc_model.fit(scaled_X_train, y_train)


# In[626]:


y_pred = grid_rfc_model.predict(scaled_X_test)


# In[628]:


accuracy_score(y_test,y_pred)


# In[630]:


confusion_matrix(y_test,y_pred)


# In[632]:


print(classification_report(y_test,y_pred))


# In[624]:


grid_rfc_model.best_params_


# In[634]:


param_grid = {"n_estimators":[1,10,40,100],'max_depth':[3,4,5, 6]}


# In[644]:


gb_model = GradientBoostingClassifier()


# In[646]:


grid_gb_model = GridSearchCV(gb_model,param_grid, verbose=2)


# In[648]:


grid_gb_model.fit(scaled_X_train, y_train)


# In[652]:


y_pred = grid_gb_model.predict(scaled_X_test)


# In[654]:


accuracy_score(y_test,y_pred)


# In[656]:


confusion_matrix(y_test,y_pred)


# In[658]:


print(classification_report(y_test,y_pred))


# In[650]:


grid_gb_model.best_params_


# In[ ]:





# In[669]:


df = pd.read_csv("Data/SAML-D.csv")
df = df.drop("Sender_account", axis=1)
df = df.drop("Receiver_account", axis=1)
non_laundering_indices = df[df["Is_laundering"] == 0].index
drop_indices = np.random.choice(non_laundering_indices, size=int(0.9 * len(non_laundering_indices)), replace=False)
df = df.drop(drop_indices)
print(df["Is_laundering"].value_counts(normalize=True))
df["Hour"] = df["Time"].apply(time_to_hour)
df = df.drop(["Time", "Date"], axis=1)


# In[679]:


df.info()


# In[ ]:





# In[681]:


cat_cols = ["Payment_currency", "Received_currency", "Sender_bank_location", "Receiver_bank_location", "Payment_type"]
cont_cols = ["Amount", "Hour"]
y_col = ["Is_laundering"]


# In[683]:


for cat in cat_cols:
    df[cat] = df[cat].astype('category')


# In[689]:


cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)


# In[ ]:





# In[693]:


cats = torch.tensor(cats, dtype=torch.int64)


# In[695]:


cats


# In[707]:


conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)


# In[709]:


conts.type()


# In[715]:


y = torch.tensor(df[y_col].values).flatten()


# In[721]:


y.shape


# In[723]:


cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
emb_szs


# In[1053]:
"""

class TabularModel(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz=2, layers=[120, 84], p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        layerlist = []
        n_emb = sum((nf for ni,nf in emb_szs))
        n_in = n_emb + n_cont
        
        for i in layers:
            layerlist.append(nn.Linear(n_in,i)) 
            layerlist.append(nn.LeakyReLU(negative_slope=0.02, inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1],out_sz))
            
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)
        
        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x

""" ---Credit to Pierian Data Inc. for network structure """
"""

# In[803]:


df.info()


# In[1286]:


model = TabularModel(emb_szs, conts.shape[1], 2, [120, 84], p=0.4)


# In[1288]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# In[1338]:


test_size=100000
batch_size = 700000

cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]


# In[ ]:





# In[1294]:


zero_indices = torch.where(y_train == 0)[0] 
one_indices = torch.where(y_train == 1)[0]   

num_zeros_to_keep = int(len(zero_indices) * 0.1)
selected_zero_indices = zero_indices[torch.randperm(len(zero_indices))[:num_zeros_to_keep]]

final_indices = torch.cat([selected_zero_indices, one_indices])

final_indices = final_indices[torch.randperm(len(final_indices))]

cat_train = cat_train[final_indices]
con_train = con_train[final_indices]
y_train = y_train[final_indices]


# In[1296]:


start_time = time.time()

epochs = 100
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = criterion(y_pred, y_train)
    losses.append(loss)
    
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') 
print(f'\nDuration: {time.time() - start_time:.0f} seconds') 


# In[1298]:


with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = criterion(y_val, y_test)
print(f'CE Loss: {loss:.8f}')


# In[1300]:


(torch.argmax(y_val, dim=1) == y_test).sum() / test_size


# In[1302]:


(torch.argmax(y_val, dim=1)).sum()


# In[1304]:


y_test.sum()


# In[1306]:


confusion_matrix(y_test, (torch.argmax(y_val, dim=1)))


# In[1308]:


indices = torch.where(y_test == 1)[0]
y_pred_for_class_1 = y_val[indices]


# In[1340]:


((y_pred_for_class_1).diff(dim=1) > -1.5).sum() 


# In[1342]:


torch.sum(y_val.diff(dim=1) > -1.5).item()


# In[933]:


len(y_val)


# In[939]:


y_val[:30]


# In[943]:


torch.sum(y_val[:2].diff(dim=1) > -1).item()


# In[1172]:


y_pred_for_class_1


# In[1188]:


((y_pred_for_class_1[:10]).diff(dim=1) > -1.5).sum()


# In[1194]:


y_pred_for_class_1.shape[0]


# In[1344]:


torch.save(model, "ML_predictor.pkl")


# In[ ]:

"""


