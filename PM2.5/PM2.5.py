#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier


# In[23]:


#新竹_2019.csv
#中文編碼encoding='ISO-8859-1'
#跳過兩列skiprows = 2
#重新編標頭names
test = pd.read_csv(r"D:\data mining\HW3\HW3.csv",encoding='ISO-8859-1',skiprows = 2,names = ['place','date','case','h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23'] )


# In[4]:


#轉換日期
test['date'] = pd.to_datetime(test['date'])


# In[5]:


test = test[test['date']>=pd.datetime(2019,10,1)]
train_data = test[test['date']<=pd.datetime(2019,11,30)] #訓練集
test_data = test[test['date']>=pd.datetime(2019,12,1)] #測試集


# In[6]:


train_data = train_data[['h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23']]
test_data = test_data[['h0','h1','h2','h3','h4','h5','h6','h7','h8','h9','h10','h11','h12','h13','h14','h15','h16','h17','h18','h19','h20','h21','h22','h23']]


# In[7]:


#製作時序資料: 將資料形式轉換為行代表18種屬性，欄代表逐時數據資料
AMB_TEMP = np.array(train_data[0::18])
AMB_TEMP = AMB_TEMP.reshape((1,1464))
CH4 = np.array(train_data[1::18])
CH4 = CH4.reshape((1,1464))
CO = np.array(train_data[2::18])
CO = CO.reshape((1,1464))
NMHC = np.array(train_data[3::18])
NMHC = NMHC.reshape((1,1464))
NO = np.array(train_data[4::18])
NO = NO.reshape((1,1464))
NO2 = np.array(train_data[5::18])
NO2 = NO2.reshape((1,1464))
NOx = np.array(train_data[6::18])
NOx = NOx.reshape((1,1464))
O3 = np.array(train_data[7::18])
O3 = O3.reshape((1,1464))
PM10 = np.array(train_data[8::18])
PM10 = PM10.reshape((1,1464))
PM2_5 = np.array(train_data[9::18])
PM2_5 = PM2_5.reshape((1,1464))
RAINFALL = np.array(train_data[10::18])
RAINFALL = RAINFALL.reshape((1,1464))
RH = np.array(train_data[11::18])
RH = RH.reshape((1,1464))
SO2 = np.array(train_data[12::18])
SO2 = SO2.reshape((1,1464))
THC = np.array(train_data[13::18])
THC = THC.reshape((1,1464))
WD_HR = np.array(train_data[14::18])
WD_HR = WD_HR.reshape((1,1464))
WIND_DIREC = np.array(train_data[15::18])
WIND_DIREC = WIND_DIREC.reshape((1,1464))
WIND_SPEED = np.array(train_data[16::18])
WIND_SPEED = WIND_SPEED.reshape((1,1464))
WS_HR = np.array(train_data[17::18])
WS_HR = WS_HR.reshape((1,1464))
train_data = np.vstack([AMB_TEMP,CH4,CO,NMHC,NO,NO2,NOx,O3,PM10,PM2_5,RAINFALL,RH,SO2,THC,WD_HR,WIND_DIREC,WIND_SPEED,WS_HR])


# In[8]:


test_AMB_TEMP = np.array(test_data[0::18])
test_AMB_TEMP = test_AMB_TEMP.reshape((1,744))
test_CH4 = np.array(test_data[1::18])
test_CH4 = test_CH4.reshape((1,744))
test_CO = np.array(test_data[2::18])
test_CO = test_CO.reshape((1,744))
test_NMHC = np.array(test_data[3::18])
test_NMHC = test_NMHC.reshape((1,744))
test_NO = np.array(test_data[4::18])
test_NO = test_NO.reshape((1,744))
test_NO2 = np.array(test_data[5::18])
test_NO2 = test_NO2.reshape((1,744))
test_NOx = np.array(test_data[6::18])
test_NOx = test_NOx.reshape((1,744))
test_O3 = np.array(test_data[7::18])
test_O3 = test_O3.reshape((1,744))
test_PM10 = np.array(test_data[8::18])
test_PM10 = test_PM10.reshape((1,744))
test_PM2_5 = np.array(test_data[9::18])
test_PM2_5 = test_PM2_5.reshape((1,744))
test_RAINFALL = np.array(test_data[10::18])
test_RAINFALL = test_RAINFALL.reshape((1,744))
test_RH = np.array(test_data[11::18])
test_RH = test_RH.reshape((1,744))
test_SO2 = np.array(test_data[12::18])
test_SO2 = test_SO2.reshape((1,744))
test_THC = np.array(test_data[13::18])
test_THC = test_THC.reshape((1,744))
test_WD_HR = np.array(test_data[14::18])
test_WD_HR = test_WD_HR.reshape((1,744))
test_WIND_DIREC = np.array(test_data[15::18])
test_WIND_DIREC = test_WIND_DIREC.reshape((1,744))
test_WIND_SPEED = np.array(test_data[16::18])
test_WIND_SPEED = test_WIND_SPEED.reshape((1,744))
test_WS_HR = np.array(test_data[17::18])
test_WS_HR = test_WS_HR.reshape((1,744))
test_data = np.vstack([test_AMB_TEMP,test_CH4,test_CO,test_NMHC,test_NO,test_NO2,test_NOx,test_O3,test_PM10,test_PM2_5,test_RAINFALL,test_RH,test_SO2,test_THC,test_WD_HR,test_WIND_DIREC,test_WIND_SPEED,test_WS_HR])


# In[14]:


#處理無效值
#訓練集無效值
for i in range(18):
    for j in range(1464):
        a = train_data[i][j]
        ans = a.replace('.', '', 1).isdigit()
        if(ans == False):
            #print("a = ",a)
            m = j-1
            n = j+1
            while True:
                if ((train_data[i][m].replace('.', '', 1).isdigit()==True) & (train_data[i][n].replace('.', '', 1).isdigit()==True)):
                    train_data[i][j] = str((float(train_data[i][m])+float(train_data[i][n]))/2)
                    break
                if (train_data[i][m].replace('.', '', 1).isdigit()==False):
                    m = m-1
                if (train_data[i][n].replace('.', '', 1).isdigit()==False):
                    n = n+1 
#測試集無效值
for i in range(18):
    for j in range(744):
        a = test_data[i][j]
        ans = a.replace('.', '', 1).isdigit()
        if(ans == False):
            #print("b = ",a)
            m = j-1
            n = j+1
            while True:
                if ((test_data[i][m].replace('.', '', 1).isdigit()==True) & (test_data[i][n].replace('.', '', 1).isdigit()==True)):
                    test_data[i][j] = str((float(test_data[i][m])+float(test_data[i][n]))/2)
                    break
                if (test_data[i][m].replace('.', '', 1).isdigit()==False):
                    m = m-1
                if (test_data[i][n].replace('.', '', 1).isdigit()==False):
                    n = n+1 


# In[15]:


#轉回float
for i in range(18):
    for j in range(1464):
        train_data[i][j] = float(train_data[i][j])

for i in range(18):
    for j in range(744):
        test_data[i][j] = float(test_data[i][j])


# In[20]:


#預測目標
#type 1 Y:將未來第一個小時當預測目標
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

a_sum=0
b_sum=0
c_sum=0
d_sum=0

for i in range(1458):
    #---X取PM2.5---
    type1_X_train = train_data[9:10,i:i+6].T
    type1_X_test = train_data[9:10,i+6:i+7].T
    if i>737:
        j = i%738
    else:
        j=i
    type1_Y_train = test_data[9:10,j:j+6].T.ravel()
    type1_Y_test = test_data[9:10,j+6:j+7].T
    
    #type a 線性回歸
    lm = LinearRegression().fit(type1_X_train, type1_Y_train)
    a_y_predict = lm.predict(type1_X_test)
    a_sum = a_sum + np.abs(type1_Y_test - a_y_predict)
    #print("a_sum = ",a_sum)
    #type b 隨機森林
    rfc = RandomForestClassifier(n_estimators=100).fit(type1_X_train, type1_Y_train.astype(int))
    b_y_predict = rfc.predict(type1_X_test)
    b_sum = b_sum + np.abs(type1_Y_test - b_y_predict)
    #print("b_sum = ",b_sum)
    
    
    #---X取18種屬性---
    type2_X_train = train_data[:,i:i+6].T
    type2_X_test = train_data[:,i+6:i+7].T
    if i>737:
        j = i%738
    else:
        j=i
    type2_Y_train = test_data[:,j:j+6].T
    type2_Y_test = test_data[:,j+6:j+7].T
    
    #type c 線性回歸
    lm = LinearRegression().fit(type2_X_train, type2_Y_train)
    c_y_predict = lm.predict(type2_X_test)
    c_sum = c_sum + np.abs(type2_Y_test - c_y_predict)
    #print("c_sum = ",c_sum)
    #type d 隨機森林
    rfc = RandomForestClassifier(n_estimators=100).fit(type2_X_train, type2_Y_train.astype(int))
    d_y_predict = rfc.predict(type2_X_test)
    d_sum = d_sum + np.abs(type2_Y_test - d_y_predict)
    #print("d_sum = ",d_sum)
   
        
a_mae = a_sum/1458
b_mae = b_sum/1458
c_mae = c_sum/1458
d_mae = d_sum/1458


# In[16]:


#type 2 Y:將未來第六個小時當預測目標
e_sum=0
f_sum=0
g_sum=0
h_sum=0


for i in range(1453):    
    #---X取PM2.5---
    type1_X_train = train_data[9:10,i:i+6].T
    type1_X_test = train_data[9:10,i+11:i+12].T
    if i>732:
        j = i%733
    else:
        j=i
    type1_Y_train = test_data[9:10,j:j+6].T.ravel()
    type1_Y_test = test_data[9:10,j+11:j+12].T.ravel()
    
    #type e 線性回歸
    lm = LinearRegression().fit(type1_X_train, type1_Y_train)
    e_y_predict = lm.predict(type1_X_test)
    e_sum = e_sum + np.abs(type1_Y_test - e_y_predict)
    #print("e_sum = ",e_sum)
    #type f 隨機森林
    rfc = RandomForestClassifier(n_estimators=100).fit(type1_X_train, type1_Y_train.astype(int))
    f_y_predict = rfc.predict(type1_X_test)
    f_sum = f_sum + np.abs(type1_Y_test - f_y_predict)
    #print("f_sum = ",f_sum)    
    
    #---X取18種屬性---
    type2_X_train = train_data[:,i:i+6].T
    type2_X_test = train_data[:,i+11:i+12].T
    if i>732:
        j = i%733
    else:
        j=i
    type2_Y_train = test_data[:,j:j+6].T
    type2_Y_test = test_data[:,j+11:j+12].T   
    
    #type g 線性回歸
    lm = LinearRegression().fit(type2_X_train, type2_Y_train)
    g_y_predict = lm.predict(type2_X_test)
    g_sum = g_sum + np.abs(type2_Y_test - g_y_predict)
    #print("g_sum = ",g_sum)
    #type h 隨機森林
    rfc = RandomForestClassifier(n_estimators=100).fit(type2_X_train, type2_Y_train.astype(int))
    h_y_predict = rfc.predict(type2_X_test)
    h_sum = h_sum + np.abs(type2_Y_test - h_y_predict)
    #print("h_sum = ",h_sum)
    
e_mae = e_sum/733
f_mae = f_sum/733
g_mae = g_sum/733
h_mae = h_sum/733


# In[17]:


'''
type a  線性回歸 x:PM2.5      y:未來第一個小時當預測目標
type b  隨機森林 x:PM2.5      y:未來第一個小時當預測目標
type c  線性回歸 x:18種屬性    y:未來第一個小時當預測目標
type d  隨機森林 x:18種屬性    y:未來第一個小時當預測目標

type e  線性回歸 x:PM2.5      y:未來第六個小時當預測目標
type f  隨機森林 x:PM2.5      y:未來第六個小時當預測目標
type g  線性回歸 x:18種屬性    y:未來第六個小時當預測目標
type h  隨機森林 x:18種屬性    y:未來第六個小時當預測目標

'''


# In[21]:


print("a_mae = ",a_mae)
print("b_mae = ",b_mae)
print("c_mae = ",c_mae)
print("d_mae = ",d_mae)


# In[19]:


print("e_mae = ",e_mae)
print("f_mae = ",f_mae)
print("g_mae = ",g_mae)
print("h_mae = ",h_mae)


# In[ ]:





# In[ ]:




