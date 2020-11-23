
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot
import math
import itertools
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
import scipy.stats
from datetime import datetime
import csv
import quandl
import quandl
import functools
from matplotlib.colors import LinearSegmentedColormap
pd.set_option("display.max_rows",1000)
import os.path


# In[1118]:


def process_factors():
    if os.path.exists("FactorLibrary.csv"):
        factors = pd.read_csv("FactorLibrary.csv",index_col=None)
    else:
        fcdata = pd.read_csv("ZACKS_FC.csv")


        frdata = pd.read_csv("ZACKS_FR.csv")

        mtdata = pd.read_csv("ZACKS_MT.csv")

        mtdata = mtdata[mtdata['country_code'] == 'US']


        mvdata = pd.read_csv("ZACKS_MKTV.csv")

        shsdata = pd.read_csv("ZACKS_SHRS.csv")

        prices = pd.read_csv("EOD_20200501.csv",header=None)
        prices = prices[[0,1,12]]
        prices[1]  = pd.to_datetime(prices[1])

        prices = prices[(prices[1] >= pd.to_datetime('2006-10-31')) & (prices[1] < pd.to_datetime('2020-05-01'))]
        prices = prices.sort_values([1])

        prices["ret"] = prices[12]/prices.groupby([0])[12].shift(1)-1
        prices["uret"] = prices.groupby(1)["ret"].transform(lambda x:x.mean())

        prices["vol1m"] = prices.groupby([0])['ret'].transform(lambda x:x.rolling(20).std())
        prices["vol3m"] = prices.groupby([0])['ret'].transform(lambda x:x.rolling(60).std())
        prices["vol6m"] = prices.groupby([0])['ret'].transform(lambda x:x.rolling(120).std())

        prices = prices.groupby([prices[1].dt.month,prices[1].dt.year,prices[0]]).tail(1) 
        prices[1] = prices[1] + pd.offsets.MonthEnd(0) 

        prices = prices[prices[0].isin(mtdata['m_ticker'])] 

        fcdata['filing_date'] = fcdata['filing_date'].astype('str') 
        fcdata = fcdata[fcdata['filing_date'] != "nan"]
        fcdata['filing_date']
        fcdata['filing_date'] = pd.to_datetime(fcdata['filing_date'])
        prices = prices.sort_values(1)

        fcdata = fcdata.sort_values('filing_date')

        prices = pd.merge_asof(prices,fcdata,allow_exact_matches=False,direction = 'backward',left_on = 1,
                                  right_on = 'filing_date',
                               left_by = 0,right_by  = "m_ticker")

        prices['per_end_date'] = pd.to_datetime(prices['per_end_date'])
        frdata['per_end_date'] = pd.to_datetime(frdata['per_end_date'])
        frdata1  = frdata[frdata["per_type"] == "Q"]
        prices1 = prices[prices[0].isin(frdata1['m_ticker'])]
        prices = prices.merge(frdata1,on = ['per_end_date','ticker'],how = 'left')
        shsdata['per_end_date'] = pd.to_datetime(shsdata['per_end_date'])


        prices = prices.merge(shsdata,on = ['per_end_date','ticker'],how = 'left',)
        prices = prices.merge(mtdata[['ticker','zacks_x_sector_desc']],on = ['ticker'],how = 'left',)

        mvdata['per_end_date'] = pd.to_datetime(mvdata['per_end_date'])

        prices = prices.merge(mvdata,on = ['per_end_date','ticker'],how = 'left',)

        prices = prices[[0,1,12,"per_end_date","filing_date","tot_debt_tot_equity","ret_invst","shares_out","net_lterm_debt","tot_lterm_debt","ticker","mkt_val","basic_net_eps","eps_diluted_net","zacks_x_sector_desc","profit_margin","asset_turn","free_cash_flow_per_share","book_val_per_share","tot_comm_pref_stock_div_paid","tot_revnu","vol1m","vol3m","vol6m"]]

        prices = prices.sort_values("per_end_date")
        prices = prices[np.isnan(prices['per_end_date']) == False]
        prices2 = prices.copy()
        prices2 = prices2.sort_values(1)
        prices = pd.merge_asof(prices,prices2[[0,1,12]],allow_exact_matches=True,direction = 'backward',
                               left_on = "per_end_date",right_on = 1,left_by = 0,right_by = 0)



        prices['tot_debt_tot_equity'] = prices['tot_debt_tot_equity']*( prices['12_y']/prices['12_x'])

        prices['Debt']  = prices['net_lterm_debt']

        prices['EPS'] = prices['eps_diluted_net']
        prices['EPS'][np.isnan(prices['eps_diluted_net'])] = prices['basic_net_eps'][np.isnan(prices['eps_diluted_net'])]
        prices  = prices.sort_values('1_x')
        prices['PE'] = prices['12_x']/prices['EPS']
        prices['12-1 Month Mom'] = prices['12_x']/prices.groupby([0])['12_x'].transform(lambda x:x.shift(12))-prices['12_x']/prices.groupby([0])['12_x'].transform(lambda x:x.shift(1))
        prices['3-Month Mom'] = prices['12_x']/prices.groupby([0])['12_x'].transform(lambda x:x.shift(3))-1
        prices['6 Month Mom'] =  prices['12_x']/prices.groupby([0])['12_x'].transform(lambda x:x.shift(6))-1
        prices['Rev-Gr'] = prices['tot_revnu']/prices.groupby([0])['tot_revnu'].transform(lambda x:x.shift(3))-1
        prices['DY'] = prices['tot_comm_pref_stock_div_paid']/prices['mkt_val']*(prices['12_y']/prices['12_x'])
        prices['PB'] = prices['12_x']/prices['book_val_per_share']
        prices['retinv'] = prices['ret_invst']*(prices['mkt_val']+prices['Debt'])/((prices['mkt_val']*(prices['12_x']/prices['12_y']))+prices['Debt'])
        prices['fcfy'] = prices["free_cash_flow_per_share"]/prices["12_x"]
        prices = prices[prices["zacks_x_sector_desc"] != "Unclassified"]
        prices2 = prices[[0,"1_x","12_x","PE","retinv","tot_debt_tot_equity","asset_turn","profit_margin","fcfy","Rev-Gr","DY","PB","vol1m","vol3m","vol6m","12-1 Month Mom","3-Month Mom","6 Month Mom","zacks_x_sector_desc"]]
        prices2.columns = ["Ticker","Date","Price","Price to Earnings","Return on Investment","Debt To Market Cap","Asset Turnover","Profit Margin","Free Cash Flow Yield","Revenue Growth","Dividend Yield","Price To Book Ratio","1 Month Volatility","3 Month Volatility","6 Month Volatility","12-1 Momentum","3 Month Momentum","6 Month Momentum","Sector"]
        prices2 = prices2.set_index('Ticker', append=True)
        prices2 = prices2.sort_values("Date")
        prices2 = prices2.groupby(level=1).ffill()
        prices2 = prices2.reset_index()
        prices2 = prices2.dropna(axis='rows',how='any')

        prices2 = prices2.melt(id_vars=['Ticker','Date','Price','Sector'], var_name='Factor', value_name='Factor Value')
        prices2 = prices2[prices2['Factor']!='level_0']
        factors = prices2.copy()
    factors["Date"] = pd.to_datetime(factors["Date"])
    return factors




# In[1120]:


def ranking(df, reb,n):
    df_temp = pd.DataFrame()
    df_data = df.copy()
    if 'tick_return' in df_data.columns:
        pass
    else:
       
        df_data['tick_return'] = df_data.groupby(['Ticker','Factor'])['Price'].transform(lambda x: x.pct_change())
        df_data['tick_return'] = df_data.groupby(['Ticker','Factor'])['tick_return'].fillna(method='ffill')
   
    df_data['Mkt'] = df_data.groupby(['Date','Factor'])['tick_return'].transform(lambda x:x.mean())
    fac_inv = ["Price to Earnings","Debt To Market Cap","Price To Book Ratio","1 Month Volatility","3 Month Volatility","6 Month Volatility"]
    df_data['Factor Value'][df_data['Factor'].isin(fac_inv)] = 1/df_data.loc[df_data['Factor'].isin(fac_inv)]['Factor Value']
   
    df_data2 =df_data.copy()
    df_data1 =df_data.copy()
    df_data2['Factor Value'] = df_data2.groupby(['Date','Sector','Factor'])['Factor Value'].transform(lambda x:(x-x.mean())/x.std())
    df_data2['Factor Value'] = df_data2.groupby(['Ticker','Factor'])['Factor Value'].fillna(method='ffill')
    df_data2['Factor'] = 'SN-'+df_data2['Factor']
    df_data1 = df_data1.append(df_data2)
   
    df_data['sector_score'] = df_data.groupby(['Date','Sector','Factor'])['Factor Value'].transform(lambda x:(x-x.mean())/x.std())
    df_data['sector_score'] = df_data.groupby(['Ticker','Factor'])['sector_score'].fillna(method='ffill')
    if reb == '1M':
        q = df_data.groupby(['Date','Factor'])['Factor Value'].rank(method='first',pct=True)
        df_data['rank'] = (q*n).apply('ceil')
        sec_q = df_data.groupby(['Date','Factor'])['sector_score'].rank(method='first',pct=True)
        df_data['sector_rank'] = (sec_q*n).apply('ceil')
       
    else:
        df_temp = df_data.groupby(['Ticker','Factor']).resample(reb,on='Date').last().reset_index(level=[0,1],drop=True).reset_index(drop=True)
        q = df_temp.groupby(['Date','Factor'])['Factor Value'].rank(method='first',pct=True)
        df_temp['rank'] = (q*n).apply('ceil')
        sec_q = df_temp.groupby(['Date','Factor'])['sector_score'].rank(method='first',pct=True)
        df_temp['sector_rank'] = (sec_q*n).apply('ceil')
                         
        df_data = df_data.merge(df_temp.reset_index()[['Date','Ticker','Factor','rank']],on=['Date','Ticker','Factor'],how='left')
   
        df_data = df_data.merge(df_temp.reset_index()[['Date','Ticker','Factor','sector_rank']],on=['Date','Ticker','Factor'],how='left')
        df_data['rank'] = df_data.groupby(['Ticker','Factor'])['rank'].fillna(method='ffill')
        df_data['sector_rank'] = df_data.groupby(['Ticker','Factor'])['sector_rank'].fillna(method='ffill')
   
           
    df_data['rank2']  = df_data.groupby(['Ticker','Factor'])['rank'].transform(lambda x:x.shift())
    df_ret = df_data.reset_index().groupby(['Date','Factor','rank2'])['tick_return'].mean().unstack().reset_index(level=[0,1])
   
    df_data['sector_rank2']  = df_data.groupby(['Ticker','Factor'])['sector_rank'].transform(lambda x:x.shift())
    sec_ret = df_data.reset_index().groupby(['Date','Factor','sector_rank2'])['tick_return'].mean().unstack().reset_index(level=[0,1])
   
    df_sec_data = df_data.copy()
    df_sec_data['Factor'] = "SN-"+df_sec_data['Factor']
    df_data = df_data.append(df_sec_data)
    sec_ret['Factor'] = "SN-"+sec_ret['Factor']
   
    df_ret = df_ret.append(sec_ret)
    df_ret = df_ret.merge(df_data.groupby(['Date','Factor'])['Mkt'].mean().reset_index()[['Date','Factor','Mkt']],how='left',on=['Date','Factor'])
   
    df_ret[11.0] = df_ret[10.0]-df_ret[1.0]
    return df_ret,df_data1


# In[1121]:
# In[1199]:


def phase_chart(df1_beta_new):
    reg_factors = ['1 Month Volatility','Price To Book Ratio','Revenue Growth','12-1 Momentum','Return on Investment']
    codes = [{reg_factors[i]:i} for i in range(len(reg_factors))]

    phase_chart = df1_beta_new[df1_beta_new['Factor'].isin(reg_factors)]
    phase_chart = phase_chart.groupby([phase_chart.Date.dt.year,phase_chart['Factor']])[11.0].agg(np.mean).reset_index()

    phase_chart = phase_chart[["Date","Factor",11.0]]
    phase_chart['Rank'] = phase_chart.groupby("Date").rank(ascending=False)

    phase_chart['cat'] = phase_chart['Factor'].apply(lambda x: reg_factors.index(x)) 
    phase_chart = phase_chart.sort_values(["Date","Rank"])
    pc1 = pd.pivot_table(phase_chart,index = ["Rank"],columns = "Date",values=["cat"])
    pc2 = pd.pivot_table(phase_chart,index = ["Rank"],columns = "Date",values=[11.0])

    myColors = ((0.8, 0.0, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0),(0.4, 0.3, 0.7, 0.3),(0.5, 0.6, 0.7, 0.8))
    cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.9, n_colors=5)

    labels =  np.array(pc2)
    fig = plt.figure(figsize = (30,20))
    ax = sns.heatmap(pc1,annot = labels,cmap=cmap,linewidths=.5,square=True,annot_kws={"size": 20},cbar_kws = dict(use_gridspec=False,location="top"))
    ax.set_xticklabels([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
    ax.set_xlabel('Year')
    sns.set(font_scale=2)
    ax.figure.axes[-1].yaxis.label.set_size(20)
    colorbar = ax.collections[0].colorbar
    colorbar.set_ticks([0.4,1.2,2,2.8,3.6])
    colorbar.set_ticklabels(reg_factors)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    return None


# In[1198]:

# [back to top](#0)

# In[1122]:


def import_indicators(start_date,end_date):
    
    #Data Sets with Monthly data
    df_umcsen= pd.read_csv('UofMich_consumer_sent.csv', header= 0)
    df_houst= pd.read_csv('HOUST.csv', header= 0)
    df_PMI= pd.read_csv('ISM-MAN_PMI.csv', header= 0)
    df_PMI.columns=['DATE','PMI']
    df_PMI['DATE'] = df_houst['DATE']
    df_non_farm= pd.read_csv('Non-farm.csv', header= 0)
    df_non_farm.columns=['DATE','NONFARM']
    df_T10Y2YM= pd.read_csv('T10Y2YM.csv', header= 0)
    df_UNRATE= pd.read_csv('UNRATE.csv', header= 0)
    df_USSLIND= pd.read_csv('US_Leading_Index.csv', header= 0)
    df_HOMEPRICE= pd.read_csv('US_National_Home_Price_Index.csv', header= 0)
    df_HOMEPRICE.columns=['DATE','HOMEPRICE']
    df_WTI= pd.read_csv('WTI_prices.csv', header= 0)
    df_WTI.columns=['DATE','WTI']
    
    #Converting Daily to Monthly
    df_VIX= pd.read_csv('VIXCLS.csv', header= 0)
    df_VIX.columns=['DATE','VIX']
    df_VIX=df_VIX.replace('.',np.NaN)
    df_VIX=df_VIX.dropna()
    df_VIX=df_VIX.groupby(pd.DatetimeIndex(df_VIX.DATE).to_period('M')).nth(0)
    df_VIX=df_VIX.set_index('DATE')
    df_VIX=df_VIX.reset_index()
    df_VIX['DATE'] = pd.to_datetime(df_VIX['DATE'])
    df_VIX = df_VIX[:-1]

    df_TEDRATE= pd.read_csv('TEDRATE.csv', header= 0)
    df_TEDRATE=df_TEDRATE.replace('.',np.NaN)
    df_TEDRATE=df_TEDRATE.dropna()
    df_TEDRATE=df_TEDRATE.groupby(pd.DatetimeIndex(df_TEDRATE.DATE).to_period('M')).nth(0)
    df_TEDRATE=df_TEDRATE.set_index('DATE')
    df_TEDRATE=df_TEDRATE.reset_index()
    df_TEDRATE['DATE'] = pd.to_datetime(df_TEDRATE['DATE'])
    df_TEDRATE = df_TEDRATE[:-1]

    df_list = [df_PMI,df_non_farm,df_T10Y2YM,df_UNRATE,df_USSLIND,df_HOMEPRICE,df_WTI]
    
    df_indicators = pd.merge(left=df_umcsen, right=df_houst, how ='inner', on ='DATE')
    df_indicators = pd.merge(left=df_indicators, right=df_TEDRATE, how ='inner', left_index=True, right_index=True)
    df_indicators = pd.merge(left=df_indicators, right=df_VIX, how ='inner', left_index=True, right_index=True)
    df_indicators  = df_indicators [df_indicators .columns.drop('DATE_y')]
    df_indicators  = df_indicators [df_indicators .columns.drop('DATE')]
    df_indicators.columns=['DATE','UMCSENT','HOUST','TEDRATE','VIX']
    
    for i in df_list:
        df_indicators = pd.merge(left=df_indicators, right=i, how ='inner', on ='DATE')
    
    df_indicators=df_indicators.set_index('DATE')
    df_indicators['TEDRATE'] = pd.to_numeric(df_indicators['TEDRATE'])
    df_indicators['VIX'] = pd.to_numeric(df_indicators['VIX'])
    
    df_indicators_delta = df_indicators-df_indicators.shift(1)
    
    
    df_indicators_short = df_indicators[(df_indicators.index>start_date) & (df_indicators.index<end_date)]
    df_indicators_delta_short = df_indicators_delta[(df_indicators_delta.index>start_date) & (df_indicators_delta.index<end_date)]

    return df_indicators_short, df_indicators_delta_short 


# In[1123]:

# In[1126]:


from sklearn.preprocessing import StandardScaler 
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn import svm


def select_features(X_train, y_train, X_test,max_features):
    # configure to select a subset of features
    fs = SelectFromModel(RandomForestClassifier(n_estimators=200,random_state=0), max_features=max_features)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def class_feature_selection(func, func_name, df_indicators_delta_short,df_indicators_short,df_mon_returns, train_len, num_features):
    #Removing last row so as to match length, to regress returns against one month before indicators
    df_indicators_delta_short = df_indicators_delta_short.reset_index()
    df_indicators_delta_short= df_indicators_delta_short[:-1]

    df_indicators_short = df_indicators_short.reset_index()
    df_indicators_short = df_indicators_short[:-1]

    #Combined indicator df, all level changes except TED and 10-2
    df_indicators_comb_short = df_indicators_delta_short.copy()
    df_indicators_comb_short['TEDRATE']=df_indicators_short['TEDRATE']
    df_indicators_comb_short['T10Y2YM']=df_indicators_short['T10Y2YM']
    
    #Feature scaling to make sure no one feature dominates due to range in which it lies
    sc_x = StandardScaler() 
    df_indicators_comb_short_temp = pd.DataFrame(sc_x.fit_transform(df_indicators_comb_short.iloc[:,1:]))
    df_indicators_comb_short_temp.columns = ['UMCSENT','HOUST','TEDRATE','VIX', 'PMI','NONFARM','T10Y2YM','UNRATE','USSLIND','HOMEPRICE','WTI']
    df_indicators_comb_short_temp['DATE']=df_indicators_comb_short['DATE']
    
    #Matching rows
    df_mon_returns= pd.DataFrame(df_mon_returns)
    df_mon_returns = df_mon_returns[1:]
    df_mon_returns = df_mon_returns.reset_index()
    df_mon_returns.columns=['DATE','Returns']
    df_mon_returns['ret_ind'] = np.where(df_mon_returns['Returns']>=0,1,0 )

    #Merge df
    df_merge_comb = pd.merge(left = df_mon_returns, right = df_indicators_comb_short_temp , how ='inner', left_index = True, right_index = True)
    df_merge_comb
     
    ss_list =['UMCSENT','HOUST','TEDRATE','VIX', 'PMI','NONFARM','T10Y2YM','UNRATE','USSLIND','HOMEPRICE','WTI']
    
    
    #Split to train and test set
    
    yhat_pred=[]
    yhat_prob_list=[]
    df_merge_comb_test_full = df_merge_comb.loc[train_len:,:]

    for i in range (len(df_merge_comb.index)-train_len):
        df_merge_comb_train = df_merge_comb.loc[i:train_len+i-1,:]
        df_merge_comb_test = df_merge_comb.loc[train_len+i:train_len+i,:]

        X_train_fs, X_test_fs, fs = select_features(df_merge_comb_train[ss_list], df_merge_comb_train['ret_ind'], df_merge_comb_test[ss_list],num_features)

        # fit the model
        if func_name=='SVM':
            
            model = func(kernel='linear',probability=True,random_state = 0)
        else:
            model = func(solver='liblinear',random_state = 0)
        
        model.fit(X_train_fs, df_merge_comb_train['ret_ind'])
        # evaluate the model
        yhat = model.predict(X_test_fs)
        yhat_prob = model.predict_proba(X_test_fs)

        yhat_pred.append(yhat[0])
        yhat_prob_list.append(yhat_prob[0][1])

    accuracy = accuracy_score(df_merge_comb_test_full['ret_ind'], yhat_pred)
    precision = precision_score(df_merge_comb_test_full['ret_ind'], yhat_pred)
    recall = recall_score(df_merge_comb_test_full ['ret_ind'], yhat_pred)
    F1 = 2 * (precision * recall) / (precision + recall)

    print('Accuracy: %.2f' % (accuracy*100))
    print('Precision: %.2f' % (precision*100))
    print('Recall: %.2f' % (recall*100))
    print('F1: %.2f' % (F1*100))
    
    return yhat_prob_list, df_merge_comb, df_merge_comb_test_full


# In[1127]:


def create_prob_table (df1,train_period,df_indicators_delta_short,df_indicators_short,num_features):
    if os.path.exists("prob_table.csv"):
        table_prob_comb = pd.read_csv("prob_table.csv",index_col=None)
    else:
    #To create Table of Probabilities
        df1 = df1.reset_index(drop=True)
        fact = df1['Factor'].unique()
        model_name = {0:'SVM',1:'Logistic'}
        model = [svm.SVC,LogisticRegression]
        df_mon_returns_date = df1[df1['Factor']=='1 Month Volatility']['Date']
        df_mon_returns_date_test = df_mon_returns_date.iloc[train_period+1:]
        prob_list_SVM = pd.DataFrame(index=df_mon_returns_date_test[:-2], columns = fact) #drop last two as macroeconomic indicators only up till Feb 2020
        prob_list_Log = pd.DataFrame(index=df_mon_returns_date_test[:-2], columns = fact)

        for f in fact:
            df_mon_returns = pd.DataFrame(columns=['DATE','Returns'])
            df_mon_returns['DATE'] = df1[df1['Factor']==f]['Date']
            df_mon_returns['Returns'] = df1[df1['Factor']==f][10.0]-df1[df1['Factor']==f][1.0]
            df_mon_returns = df_mon_returns.set_index('DATE') #passed in dates index as well so that we can check alignment of returns with macroeconomic indicators (should be one month lag); use y,z to check
            #train_len = int(np.size(df_mon_returns.index)/2) #Ian to Manoj: What does this length represent?

            print(f,':::')
            prob_list_SVM[f],y1,y2 = class_feature_selection(model[0],model_name[0],df_indicators_delta_short,df_indicators_short,df_mon_returns, train_period ,num_features)
            print()
            prob_list_Log[f],z1,z2 = class_feature_selection(model[1],model_name[1],df_indicators_delta_short,df_indicators_short,df_mon_returns, train_period ,num_features)
            print()

        prob_list_SVM1=prob_list_SVM.reset_index()
        table_prob_SVM =prob_list_SVM1.melt(id_vars =['Date'])
        table_prob_SVM.columns=['Date','Factor','SVM_Prob']

        prob_list_Log1=prob_list_Log.reset_index()
        table_prob_log =prob_list_Log1.melt(id_vars =['Date'])
        table_prob_log.columns=['Date','Factor','Log_Prob']

        table_prob_comb=table_prob_SVM.copy()
        table_prob_comb['Log_Prob']=table_prob_log['Log_Prob']

    return table_prob_comb


# In[1128]:


# In[1129]:


def best_fac(prob_table,df1):
    rets = df1.copy()
    rets[11.0] = rets[10.0] -rets[1.0]
    rets = rets[["Factor","Date",11.0]]
    rets = pd.pivot_table(rets,index= "Date",columns ="Factor")
    corr = rets.rolling(48).corr().reset_index()
    corr.columns = ["Date","Level","Factor"] + [x[1] for x in corr.columns[3:]]
    
    ranker  = prob_table.copy()
    ranker['SVM_Rank'] = ranker.groupby(['Date'])['SVM_Prob'].rank(method = 'first',ascending=False)
    ranker['Log_Rank'] = ranker.groupby(['Date'])['Log_Prob'].rank(method = 'first',ascending=False)
    SVMmodel = ranker[['Date','Factor','SVM_Prob','SVM_Rank']]
    Logmodel = ranker[['Date','Factor','Log_Prob','Log_Rank']]
    SVMmodel = SVMmodel.sort_values(by = ["Date","SVM_Rank"],ascending = [True,True])
    SVMmodel['List'] = SVMmodel.groupby('Date')['Factor'].apply(lambda x: (x + ',').cumsum().str.split(','))
    SVMmodel['List'] = SVMmodel['List'].apply(lambda x: x[:-2])
    def corrm(date,factor,listo):
        x = np.max(corr[listo][(corr['Date'] == date) & (corr['Factor'] == factor)].values)
        return x
    SVMmodel['Corr'] = 0
    SVMmodel['Corr'][SVMmodel['List'].apply(len)!=0] = SVMmodel[SVMmodel['List'].apply(len)!=0].apply(lambda x: corrm(x['Date'],x['Factor'],x['List']),axis=1)
    SVMmodel = SVMmodel[SVMmodel["Corr"]<0.8]
    SVMmodel['SVM_Rank'] = SVMmodel.groupby(['Date'])['SVM_Prob'].rank(method = 'first',ascending=False)
    SVMmodel= SVMmodel[SVMmodel['SVM_Rank']<=4]
    
    Logmodel = Logmodel.sort_values(by = ["Date","Log_Rank"],ascending = [True,True])
    Logmodel['List'] = Logmodel.groupby('Date')['Factor'].apply(lambda x: (x + ',').cumsum().str.split(','))
    Logmodel['List'] = Logmodel['List'].apply(lambda x: x[:-2])
    Logmodel['Corr'] = 0
    Logmodel['Corr'][Logmodel['List'].apply(len)!=0] = Logmodel[Logmodel['List'].apply(len)!=0].apply(lambda x: corrm(x['Date'],x['Factor'],x['List']),axis=1)
    Logmodel = Logmodel[Logmodel["Corr"]<0.8]
    Logmodel['Log_Rank'] = Logmodel.groupby(['Date'])['Log_Prob'].rank(method = 'first',ascending=False)
    Logmodel= Logmodel[Logmodel['Log_Rank']<=4]
    
    SVMmodel = SVMmodel[["Date","Factor","SVM_Prob","SVM_Rank"]]
    Logmodel = Logmodel[["Date","Factor","Log_Prob","Log_Rank"]]
    return SVMmodel,Logmodel,corr
    


# In[1130]:



def combinedBacktest(scores,SVMmodel,Logmodel,factors):
    Logmodel['Date'] = pd.to_datetime(Logmodel["Date"])
    SVMmodel['Date'] = pd.to_datetime(SVMmodel["Date"])
    Logmodel = Logmodel.merge(scores,how = "left",on = ["Date","Factor"])
    SVMmodel = SVMmodel.merge(scores,how = "left",on = ["Date","Factor"])
    Logmodel = Logmodel[["Ticker","Date","Price","Sector","Factor","Factor Value"]]
    SVMmodel = SVMmodel[["Ticker","Date","Price","Sector","Factor","Factor Value"]]
    Logmodel['Factor Value'] =  Logmodel.groupby(['Date','Factor'])['Factor Value'].transform(lambda x:(x-x.mean())/x.std())
    SVMmodel['Factor Value'] =  SVMmodel.groupby(['Date','Factor'])['Factor Value'].transform(lambda x:(x-x.mean())/x.std())
    Logmodel['Factor Value2'] = Logmodel.groupby(["Date","Ticker"])['Factor Value'].transform(np.mean)
    SVMmodel['Factor Value2'] = SVMmodel.groupby(["Date","Ticker"])['Factor Value'].transform(np.mean)
    Logmodel["Factor"] = "Log Model Combination"
    SVMmodel["Factor"] = "SVM Combination"
    Logmodel = Logmodel.groupby(['Date','Ticker']).first().reset_index()
    SVMmodel = SVMmodel.groupby(['Date','Ticker']).first().reset_index()
    Logmodel['Factor Value'] = Logmodel['Factor Value2']
    SVMmodel['Factor Value'] = SVMmodel['Factor Value2']
    Logmodel  = Logmodel[factors.columns]
    SVMmodel  = SVMmodel[factors.columns]
    comb = Logmodel.append(SVMmodel)
    comb.reset_index(drop=True,inplace=True)
    comb1,comb2 = ranking(comb,'1M',10)
    comb1 = comb1[comb1["Factor"].isin(["Log Model Combination","SVM Combination"])]
    return comb1,comb2
    
    
    
    
    


# In[1132]:



def plot_quantile(df1):
    df1 = df1.copy()
    df1 = df1[df1['Date'] >= "2012-01-31"]
    fig = plt.figure(figsize = (20,100))
    k=1
    for i in list(set(df1["Factor"])):
        ax1 = fig.add_subplot(15,2,k)    
        fac = df1[df1['Factor']==i]
        mean_fac = fac.mean()[0:10]
        plt.bar(height = mean_fac,x = list(range(1,len(mean_fac)+1)))
        plt.title(i)
        k+=1
        None
None




# In[1137]:


def plot_returns(df1):
    df2 = df1.copy()
    df2 = df2[df2['Date'] >= "2012-01-31"]
    fig = plt.figure(figsize = (20,100))
    k=1
    for i in list(set(df1["Factor"])):
        ax1 = fig.add_subplot(15,2,k)    
        fac = df2[df2['Factor']==i]
        fac2 = fac[fac.columns[2:]]
        cum_fac = fac2.copy()
        cum_fac = 1+cum_fac
        cum_fac = cum_fac.apply(np.cumprod)
        plt.plot(fac["Date"],cum_fac[1.0],label = "Short Decile")
        plt.plot(fac["Date"],cum_fac[10.0],label  = "Long Decile")
        plt.plot(fac["Date"],cum_fac[11.0],label = "Long Short Spread")
        plt.title(i)
        plt.legend()
        k+=1
    


# In[1138]:


import pandas_datareader as pdr
famafrenchdata = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv',index_col = 'Date')
famafrenchdata = famafrenchdata.reset_index()
famafrenchdata ['Date'] = pd.to_datetime(famafrenchdata['Date'])


# In[1141]:


import statsmodels.api as sm

def reg_LS_spread_vs_FFM5 (df1,famafrenchdata):
    df_reg = pd.DataFrame(index = df1.index, columns =['Date','Factor',11.0])
    df_reg['Date'] = df1['Date']
    df_reg['Factor'] = df1['Factor']
    df_reg = df_reg[df_reg['Date']>="2012-01-31"]
    df_reg[11.0] = df1[11.0]
    df_reg = pd.merge(df_reg, famafrenchdata[['Date','Mkt-RF','SMB','HML','RMW','CMA',]],on='Date', how='left')
    group_factor=df_reg.groupby('Factor')
    fact_list = df_reg['Factor'].unique()
    ff_list=['Mkt-RF','SMB','HML','RMW','CMA']

    reg_beta = pd.DataFrame(index=fact_list, columns= ff_list)
    reg_resid = pd.DataFrame(index=df1['Date'].unique(), columns=fact_list)
    reg_resid = reg_resid[reg_resid.index>="2012-01-31"]

    for j in fact_list:
        fact = group_factor.get_group(j)

        rhs = sm.add_constant(fact[ff_list])
        lhs = fact[11]
        res = sm.OLS(lhs, rhs, missing='drop').fit()
        for i in range (5):
            reg_beta.loc[j][i]=res.params[i+1]

        reg_resid[j]= np.array(res.resid+res.params[0])

    residual_formatted=reg_resid.T.reset_index()
    residual_formatted=residual_formatted.melt(id_vars =['index'])
    residual_formatted.columns=['Factor','Date','Residual']
    
    return residual_formatted, reg_beta
    


# In[1142]:


#Incorporates leverage, transaction cost (0.2%), funding cost (market rf) and eqmarket stats and reg residual of LS-spread vv FFM-5 stats 
def summaryStats(df2,famafrenchdata,residual_formatted):
    
    df1=df2.copy()
    df1 = df1[df1['Date']>="2012-01-30"]
    #Apply 5x leverage
    df1.iloc[:,2:12]=df1.iloc[:,2:12]*5
    
    #Account for transaction cost of 0.2%
    df1[10.0]=df1[10.0]-0.2/100
    
    df1 = pd.merge(df1, famafrenchdata[['Date','RF']],on='Date', how='left')
    df1 = pd.merge(df1, residual_formatted[['Residual']],left_index=True, right_index=True, how='left')
    
    #Account for funding cost of 0.2%
    df1[10.0]=df1[10.0]-(df1['RF']/100*.8)
    
    #Residual calculations
    df1[11.0]=df1[11.0]*5
    df1['Residual']=df1['Residual']*5
    df1['Residual']=df1['Residual']-0.2/100
    df1['Residual']=df1['Residual']-(df1['RF']/100*.8)
    
    meanResidual = df1.groupby('Factor')['Residual'].mean()*12
    stdResidual = df1.groupby('Factor')['Residual'].std()*(12**0.5)
    sharpeResidual = meanResidual/stdResidual
    hitRateResidual = df1[df1['Residual']>0].groupby('Factor').agg('count')/(df1[df1['Residual']>0].groupby('Factor').agg('count')+df1[df1['Residual']<0].groupby('Factor').agg('count'))
    hitRateResidual = hitRateResidual['Residual'].tolist()
    winnerToLoserResidual = -df1[df1['Residual']>0].groupby('Factor').agg('mean')/df1[df1['Residual']<0].groupby('Factor').agg('mean')
    winnerToLoserResidual = winnerToLoserResidual['Residual'].tolist()
    
    #mkt calculations
    df1.iloc[:,12]=df1.iloc[:,12]*5
    df1['Mkt']=df1['Mkt']-0.2/100
    df1['Mkt']=df1['Mkt']-(df1['RF']/100*.8)
    
    meanmkt = df1.groupby('Factor')['Mkt'].mean()*12
    stdmkt = df1.groupby('Factor')['Mkt'].std()*(12**0.5)
    sharpemkt = meanmkt/stdmkt
    hitRatemkt = df1[df1['Mkt']>0].groupby('Factor').agg('count')/(df1[df1['Mkt']>0].groupby('Factor').agg('count')+df1[df1['Mkt']<0].groupby('Factor').agg('count'))
    hitRatemkt = hitRatemkt['Mkt'].tolist()
    winnerToLosermkt = -df1[df1['Mkt']>0].groupby('Factor').agg('mean')/df1[df1['Mkt']<0].groupby('Factor').agg('mean')
    winnerToLosermkt = winnerToLosermkt['Mkt'].tolist()
    
    #df1=df1.drop(['Mkt','RF'], axis =1)
    
    
    meanRet = df1.groupby('Factor')[10.0].mean()*12
    meanSRet = df1.groupby('Factor')[1.0].mean()*12
    #df1[11.0] = df1[10.0] - df1[1.0]

    
    meanLSRet = df1.groupby('Factor')[11.0].mean()*12
    stdRet = df1.groupby('Factor')[10.0].std()*(12**0.5)
    stdSRet = df1.groupby('Factor')[1.0].std()*(12**0.5)
    stdLSRet = df1.groupby('Factor')[11.0].std()*(12**0.5)
    
    cumdf1 = df1.copy()
    cumdf1[cumdf1.columns[2:]] = cumdf1[cumdf1.columns[2:]].apply(lambda x: 1+x)
    cumdf1[cumdf1.columns[2:]] = cumdf1.groupby('Factor')[cumdf1.columns[2:]].cumprod()
    cummax = cumdf1.copy()
    cummax[cummax.columns[2:]] = cummax.groupby('Factor')[cummax.columns[2:]].cummax()
    drawdown = cummax.copy()
    drawdown[drawdown.columns[2:]] = -(cumdf1[cumdf1.columns[2:]]/cummax[cummax.columns[2:]]-1)
    drawdownL = drawdown.groupby('Factor')[10.0].max()
    drawdownS = drawdown.groupby('Factor')[1.0].max()
    drawdownLS = drawdown.groupby('Factor')[11.0].max()
    drawdownmkt = drawdown.groupby('Factor')['Mkt'].max()
    drawdownResidual = drawdown.groupby('Factor')['Residual'].max()
    
    sharpe = meanRet/stdRet
    sharpeS = meanSRet/stdSRet
    sharpeLS = meanLSRet/stdLSRet
    hitRate = df1[df1[10.0]>0].groupby('Factor').agg('count')/(df1[df1[10.0]>0].groupby('Factor').agg('count')+df1[df1[10.0]<0].groupby('Factor').agg('count'))
    hitRate = hitRate[10.0].tolist()
    hitRateS = df1[df1[10.0]>0].groupby('Factor').agg('count')/(df1[df1[1.0]>0].groupby('Factor').agg('count')+df1[df1[1.0]<0].groupby('Factor').agg('count'))
    hitRateS = hitRateS[1.0].tolist()
    
    hitRateLS = df1[df1[11.0]>0].groupby('Factor').agg('count')/(df1[df1[11.0]>0].groupby('Factor').agg('count')+df1[df1[11.0]<0].groupby('Factor').agg('count'))
    hitRateLS = hitRateLS[11.0].tolist()
    
    winnerToLoser = -df1[df1[10.0]>0].groupby('Factor').agg('mean')/df1[df1[10.0]<0].groupby('Factor').agg('mean')
    winnerToLoser = winnerToLoser[10.0].tolist()
    winnerToLoserS = -df1[df1[1.0]>0].groupby('Factor').agg('mean')/df1[df1[1.0]<0].groupby('Factor').agg('mean')
    winnerToLoserS = winnerToLoserS[1.0].tolist()
    winnerToLoserLS = -df1[df1[11.0]>0].groupby('Factor').agg('mean')/df1[df1[11.0]<0].groupby('Factor').agg('mean')
    winnerToLoserLS = winnerToLoserLS[11.0].tolist()

    stats = pd.DataFrame(list(zip(meanRet,stdRet,sharpe,hitRate,winnerToLoser,drawdownL,meanSRet,stdSRet,sharpeS,hitRateS,winnerToLoserS,drawdownS,meanLSRet,stdLSRet,sharpeLS,hitRateLS,winnerToLoserLS,drawdownLS,meanResidual, stdResidual,sharpeResidual,hitRateResidual,winnerToLoserResidual,drawdownResidual)))
    stats.columns = ["Long Ret","Long Std","Long Sharpe","Hit Rate","Long Winner to Loser Ratio","Long Drawdown","Short Ret","Short Std","Short Sharpe","Short Hot Rate","Short Winner To Loser Ratio","Short Drawdown","Long Short Ret","Long Short Std","Long Short Sharpe","Long Short Hit Rate","Long Short Average Winner To Loser","Long Short Drawdown","Residual Ret","Residual Std","Residual Sharpe","Residual Hit Rate","Residual Average Winner To Loser","Residual Drawdown"]
    stats.index = meanRet.index
    
    stats2 = pd.DataFrame(list(zip(meanmkt,stdmkt,sharpemkt,hitRatemkt,winnerToLosermkt,drawdownmkt)))
    stats2.columns = ["Mkt Ret","Mkt Std","Mkt Sharpe","Mkt Hit Rate","Mkt Average Winner To Loser","Mkt Drawdown"]
    #stats2.index = meanRet.index[0:15]
    
    
    return stats, stats2
    


# In[1147]:


def mcsims(data):
    if os.path.exists("sims.csv"):
        sims = pd.read_csv("sims.csv",index_col = None)
        means = sims[sims.columns[0]].tolist()
        stds = sims[sims.columns[1]].tolist()
    else:
        means = []
        stds = []
        for i in range(1000):
            print(i)
            temp = data[data['Factor'] == "Price to Earnings"].copy()
            temp = temp[temp['Date']>="2011-11-30"]
            temp["Factor Value"] = np.random.normal(0, 1,  temp.shape[0])
            temp["rank"] = temp.groupby(['Date','Factor'])['Factor Value'].rank(method='first',pct=True)
            temp['rank'] = (temp['rank']*10).apply('ceil')
            mean10 = temp[temp['rank'] == 10].groupby(['Date'])['tick_return'].mean()
            mean1 = temp[temp['rank'] == 1].groupby('Date')['tick_return'].mean()
            mean = list(map(lambda x,y: x-y,mean10,mean1))
            means.append(np.mean(mean)*12)
            stds.append(np.std(mean)*(12**0.5))
    return means,stds
    


# In[1154]:

# In[1156]:


def bn_mcsims(data):
    if os.path.exists("bnsims.csv"):
        sims = pd.read_csv("bnsims.csv",index_col = None)
        means = sims[sims.columns[0]].tolist()
        stds = sims[sims.columns[1]].tolist()
    else:
        means = []
        stds = []
        for i in range(1000):
            print(i)
            temp = data[data['Factor'] == "Price to Earnings"].copy()
            temp["Factor Value"] = np.random.normal(0, 1,  temp.shape[0])
            temp["rank"] = temp.groupby(['Date','Factor'])['Factor Value'].rank(method='first',pct=True)
            temp['rank'] = (temp['rank']*10).apply('ceil')
            temp2 = temp.copy()
            temp = pd.pivot_table(temp,index = ["Date","Factor"],values="tick_return",columns = "rank").reset_index()
            r = temp2[temp2["rank"]==1].sort_values("Date")
            r = pd.DataFrame(r.groupby('Date')["Mkt"].apply(np.nanmean))
            temp['Mkt'] = r[r.columns[0]][1:].values
            temp['cov_1']=temp.groupby(['Factor'])[['Date','Mkt',1.0]].apply(lambda x: x[['Mkt',1.0]].rolling(48).cov()).groupby(level=[0,1]).last()['Mkt'].reset_index(level=1).sort_values('level_1')['Mkt'].values
            temp['var_mkt']=(temp.groupby(['Factor'])['Mkt'].apply(lambda x: x.rolling(48).std()))**2
            temp['cov_10']=temp.groupby(['Factor'])[['Mkt',10.0]].apply(lambda x: x.rolling(48).cov()).groupby(level=[0,1]).last()['Mkt'].reset_index(level=1).sort_values('level_1')['Mkt'].values
            temp['beta_1'] = temp['cov_1']/temp['var_mkt']
            temp['beta_10'] = temp['cov_10']/temp['var_mkt']
            temp[11.0]=temp[10.0]-temp[1.0]*temp['beta_10']/temp['beta_1']
            means.append(np.nanmean(temp[11.0])*12)
            stds.append(np.nanstd(temp[11.0])*(12**0.5))
    return means,stds
    


# In[1157]:
def plot_mcsims(statsLS,statsLS2,means,stds):
    meansFacs = statsLS[statsLS.columns[12:]].sort_values('Long Short Sharpe',ascending=False)['Long Short Ret']/5
    stdFacs = statsLS[statsLS.columns[12:]].sort_values('Long Short Sharpe',ascending=False)['Long Short Std']/5

    meansFacs2 = statsLS2[statsLS2.columns[12:]].sort_values('Long Short Sharpe',ascending=False)['Long Short Ret']/5
    stdFacs2 = statsLS2[statsLS2.columns[12:]].sort_values('Long Short Sharpe',ascending=False)['Long Short Std']/5
    fig,ax=plt.subplots(figsize=(18, 9))
    plt.scatter(x=stds,y=means,alpha = 0.2, label = "MC Simulated Portfolios")
    plt.scatter(x=stdFacs,y=meansFacs,alpha = 0.8,label = "Individual Factors Long Short Spreads")
    plt.scatter(x=stdFacs2,y=meansFacs2,alpha = 0.8,label = "Combined Factor Long Short Spreads")
    plt.legend()
    plt.xlabel("Volatility of the portfolio")
    plt.ylabel("Mean of the portfolio")
    plt.title("Comparison of Factor portfolios against randomly simulated portfolios")
    plt.xlim([0,0.25])
    return None
# In[1159]:


def beta_neutral(df_rank):
    df_rank['cov_1']=df_rank.groupby(['Factor'])[['Date','Mkt',1.0]].apply(lambda x: x[['Mkt',1.0]].rolling(48).cov()).groupby(level=[0,1]).last()['Mkt'].reset_index(level=1).sort_values('level_1')['Mkt'].values
    df_rank['var_mkt']=(df_rank.groupby(['Factor'])['Mkt'].apply(lambda x: x.rolling(48).std()))**2
    df_rank['cov_10']=df_rank.groupby(['Factor'])[['Mkt',10.0]].apply(lambda x: x.rolling(48).cov()).groupby(level=[0,1]).last()['Mkt'].reset_index(level=1).sort_values('level_1')['Mkt'].values
    df_rank['beta_1'] = df_rank['cov_1']/df_rank['var_mkt']
    df_rank['beta_10'] = df_rank['cov_10']/df_rank['var_mkt']
    df_rank[11.0]=df_rank[10.0]-df_rank[1.0]*df_rank['beta_10']/df_rank['beta_1']
    
    return df_rank


# In[1160]:


def class_feature_selection_bn(func, func_name, df_indicators_delta_short,df_indicators_short,df_mon_returns, train_len, num_features):
    #Removing last row so as to match length, to regress returns against one month before indicators
    df_indicators_delta_short = df_indicators_delta_short.reset_index()
    df_indicators_delta_short= df_indicators_delta_short[:-1]

    df_indicators_short = df_indicators_short.reset_index()
    df_indicators_short = df_indicators_short[:-1]

    #Combined indicator df, all level changes except TED and 10-2
    df_indicators_comb_short = df_indicators_delta_short.copy()
    df_indicators_comb_short['TEDRATE']=df_indicators_short['TEDRATE']
    df_indicators_comb_short['T10Y2YM']=df_indicators_short['T10Y2YM']
    
    #Feature scaling to make sure no one feature dominates due to range in which it lies
    sc_x = StandardScaler() 
    df_indicators_comb_short_temp = pd.DataFrame(sc_x.fit_transform(df_indicators_comb_short.iloc[:,1:]))
    df_indicators_comb_short_temp.columns = ['UMCSENT','HOUST','TEDRATE','VIX', 'PMI','NONFARM','T10Y2YM','UNRATE','USSLIND','HOMEPRICE','WTI']
    df_indicators_comb_short_temp['DATE']=df_indicators_comb_short['DATE']
    
    #Matching rows
    df_mon_returns= pd.DataFrame(df_mon_returns)
    df_mon_returns = df_mon_returns[1:]
    df_mon_returns = df_mon_returns.reset_index()
    #df_mon_returns.columns=['DATE','Returns']
    #df_mon_returns['ret_ind'] = np.where(df_mon_returns['Returns']>=0,1,0 )

    #Merge df
    #df_merge_comb1 = pd.merge(left = df_mon_returns, right = df_indicators_comb_short_temp , how ='inner', left_index = True, right_index = True)
    #df_merge_comb
     
    ss_list =['UMCSENT','HOUST','TEDRATE','VIX', 'PMI','NONFARM','T10Y2YM','UNRATE','USSLIND','HOMEPRICE','WTI']
        
    #Split to train and test set
    
    yhat_pred=[]
    yhat_prob_list=[]
    #df_merge_comb_test_full = df_merge_comb1.loc[train_len-1:,:]
    df_mon_returns['Returns'] = np.nan
    df_mon_returns['ret_ind'] = np.nan
    
    for i in range(99):
        
        df_mon_returns['Returns'].loc[i:train_len+i] = df_mon_returns[10.0].loc[i:train_len+i]- df_mon_returns[1.0].loc[i:train_len+i]                                                        *df_mon_returns['beta_10'].loc[train_len+i-1:train_len+i-1].values/df_mon_returns['beta_1'].loc[train_len+i-1:train_len+i-1].values
        
        df_mon_returns['ret_ind'].loc[i:train_len+i] = np.where(df_mon_returns['Returns'].loc[i:train_len+i]>=0,1,0 )
                          
        df_merge_comb = pd.merge(left = df_mon_returns['ret_ind'].loc[i:train_len+i], right = df_indicators_comb_short_temp.loc[i:train_len+i,:], how ='inner', left_index = True, right_index = True)
        
        df_merge_comb_train = df_merge_comb.loc[i:train_len+i-1,:]
        df_merge_comb_test = df_merge_comb.loc[train_len+i:train_len+i,:]
        
        X_train_fs, X_test_fs, fs = select_features(df_merge_comb_train[ss_list], df_merge_comb_train['ret_ind'], df_merge_comb_test[ss_list],num_features)
        
        # fit the model
        if func_name=='SVM':
            
            model = func(kernel='linear',probability=True,random_state = 0)
        else:
            model = func(solver='liblinear',random_state = 0)
        
        
        
        model.fit(X_train_fs, df_merge_comb_train['ret_ind'])
    # evaluate the model
        yhat = model.predict(X_test_fs)
        yhat_prob = model.predict_proba(X_test_fs)

        yhat_pred.append(yhat[0])
        yhat_prob_list.append(yhat_prob[0][1])

    #accuracy = accuracy_score(df_merge_comb_test_full['ret_ind'], yhat_pred)
    #precision = precision_score(df_merge_comb_test_full['ret_ind'], yhat_pred)
    #recall = recall_score(df_merge_comb_test_full ['ret_ind'], yhat_pred)
    #F1 = 2 * (precision * recall) / (precision + recall)

#     print('Accuracy: %.2f' % (accuracy*100))
#     print('Precision: %.2f' % (precision*100))
#     print('Recall: %.2f' % (recall*100))
#     print('F1: %.2f' % (F1*100))
    
    return yhat_prob_list, df_merge_comb


# In[1162]:


def create_prob_table_bn(df1_beta,train_period,df_indicators_delta_short,df_indicators_short,num_features):
    
    df1 = df1_beta.copy()
    if os.path.exists("prob_table_BN.csv"):
        table_prob_comb = pd.read_csv("prob_table_BN.csv",index_col=None)
        
    else:
    #To create Table of Probabilities
        df1 = df1.reset_index(drop=True)
        fact = df1['Factor'].unique()
        model_name = {0:'SVM',1:'Logistic'}
        model = [svm.SVC,LogisticRegression]
        df_mon_returns_date = df1[df1['Factor']=='1 Month Volatility']['Date']
        df_mon_returns_date_test = df_mon_returns_date.iloc[train_period+1:]
        prob_list_SVM = pd.DataFrame(index=df_mon_returns_date_test[:-2], columns = fact) #drop last two as macroeconomic indicators only up till Feb 2020
        prob_list_Log = pd.DataFrame(index=df_mon_returns_date_test[:-2], columns = fact)

        for f in fact:
            df_mon_returns = pd.DataFrame()
            df1['DATE'] = df1['Date']         
            df_mon_returns = df1[['DATE',1.0,10.0,'beta_1','beta_10']][df1['Factor']==f]
                    
            df_mon_returns.set_index('DATE',inplace=True) #passed in dates index as well so that we can check alignment of returns with macroeconomic indicators (should be one month lag); use y,z to check
            #train_len = int(np.size(df_mon_returns.index)/2) #Ian to Manoj: What does this length represent?

            print(f,':::')
            
            prob_list_SVM[f],y1 = class_feature_selection_bn(model[0],model_name[0],df_indicators_delta_short,df_indicators_short,df_mon_returns, train_period ,num_features)
            prob_list_Log[f],z1 = class_feature_selection_bn(model[1],model_name[1],df_indicators_delta_short,df_indicators_short,df_mon_returns, train_period ,num_features)
            
        prob_list_SVM1=prob_list_SVM.reset_index()
        table_prob_SVM =prob_list_SVM1.melt(id_vars =['Date'])
        table_prob_SVM.columns=['Date','Factor','SVM_Prob']

        prob_list_Log1=prob_list_Log.reset_index()
        table_prob_log =prob_list_Log1.melt(id_vars =['Date'])
        table_prob_log.columns=['Date','Factor','Log_Prob']

        table_prob_comb=table_prob_SVM.copy()
        table_prob_comb['Log_Prob']=table_prob_log['Log_Prob']
        table_prob_comb.to_csv('prob_table_BN.csv',index=False)
        #table_prob_comb.to_csv('Y1_BN.csv',index=False)
        #table_prob_comb.to_csv('Y2_BN.csv',index=False)
    
    return table_prob_comb


# In[1163]:

from dateutil.relativedelta import relativedelta as reld

def best_fac_beta(prob_table,df1_beta_new):
    rets = df1_beta_new.copy()
    dates = list(set(rets['Date']))
    dates.sort()
    dates = dates[48:]
    for d in dates:
        rets_temp = rets[(rets['Date']<=pd.to_datetime(d)) & (rets['Date']>=pd.to_datetime(d)-reld(months=48))]
        rets_temp = rets_temp.sort_values("Date")
        rets_temp["beta_1"] = rets_temp.groupby("Factor")["beta_1"].transform('last')
        rets_temp["beta_10"] = rets_temp.groupby("Factor")["beta_10"].transform('last')
        rets_temp[11.0] = rets_temp[10.0]-rets_temp["beta_10"]/rets_temp["beta_1"]*rets_temp[1.0]
        rets_temp = rets_temp[["Date","Factor",11.0]]
        rets_temp = pd.pivot_table(rets_temp,index= "Date",columns ="Factor")
        if d == dates[0]:
            corr = rets_temp.corr().reset_index()
            corr["Date"] = d
            l = rets_temp
        else:
            temp_corr = rets_temp.corr().reset_index()
            temp_corr["Date"] = d
            corr = corr.append(temp_corr)
    corr.columns = ["Level","Factor"] + [x[1] for x in corr.columns[2:-1]]+["Date"]
    ranker  = prob_table.copy()
    ranker['SVM_Rank'] = ranker.groupby(['Date'])['SVM_Prob'].rank(method = 'first',ascending=False)
    ranker['Log_Rank'] = ranker.groupby(['Date'])['Log_Prob'].rank(method = 'first',ascending=False)
    SVMmodel = ranker[['Date','Factor','SVM_Prob','SVM_Rank']]
    Logmodel = ranker[['Date','Factor','Log_Prob','Log_Rank']]
    SVMmodel = SVMmodel.sort_values(by = ["Date","SVM_Rank"],ascending = [True,True])
    SVMmodel['List'] = SVMmodel.groupby('Date')['Factor'].apply(lambda x: (x + ',').cumsum().str.split(','))
    SVMmodel['List'] = SVMmodel['List'].apply(lambda x: x[:-2])
    def corrm(date,factor,listo):
        x = np.max(corr[listo][(corr['Date'] == date) & (corr['Factor'] == factor)].values)
        return x
    SVMmodel['Corr'] = 0
    SVMmodel['Corr'][SVMmodel['List'].apply(len)!=0] = SVMmodel[SVMmodel['List'].apply(len)!=0].apply(lambda x: corrm(x['Date'],x['Factor'],x['List']),axis=1)
    SVMmodel = SVMmodel[SVMmodel["Corr"]<0.8]
    SVMmodel['SVM_Rank'] = SVMmodel.groupby(['Date'])['SVM_Prob'].rank(method = 'first',ascending=False)
    SVMmodel= SVMmodel[SVMmodel['SVM_Rank']<=4]
    
    Logmodel = Logmodel.sort_values(by = ["Date","Log_Rank"],ascending = [True,True])
    Logmodel['List'] = Logmodel.groupby('Date')['Factor'].apply(lambda x: (x + ',').cumsum().str.split(','))
    Logmodel['List'] = Logmodel['List'].apply(lambda x: x[:-2])
    Logmodel['Corr'] = 0
    Logmodel['Corr'][Logmodel['List'].apply(len)!=0] = Logmodel[Logmodel['List'].apply(len)!=0].apply(lambda x: corrm(x['Date'],x['Factor'],x['List']),axis=1)
    Logmodel = Logmodel[Logmodel["Corr"]<0.8]
    Logmodel['Log_Rank'] = Logmodel.groupby(['Date'])['Log_Prob'].rank(method = 'first',ascending=False)
    Logmodel= Logmodel[Logmodel['Log_Rank']<=4]
    SVMmodel = SVMmodel[["Date","Factor","SVM_Prob","SVM_Rank"]]
    Logmodel = Logmodel[["Date","Factor","Log_Prob","Log_Rank"]]
    return SVMmodel,Logmodel,corr
    


# In[1165]:

def combinedBacktest_BN(scores,SVMmodel,Logmodel):
    if os.path.exists("beta_neutral.csv"):
        comb_n = pd.read_csv("beta_neutral.csv",index_col=None)
        comb_n.columns = ["Date","Factor",1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,'Mkt', 'beta_1', 'beta_10', 11.0]
        comb_n = comb_n[comb_n['Factor'].isin(["Log Model Combination","SVM Combination"])]
        comb_n['Date'] = pd.to_datetime(comb_n['Date'])
    else:
        SVMmodel1 = SVMmodel.copy()
        Logmodel1 = Logmodel.copy()
        beta_sv = pd.DataFrame(index=list(SVMmodel1['Date'].unique()),columns=['Date','Factor',11.0,'beta_1','beta_10'])
        beta_log = pd.DataFrame(index=list(Logmodel1['Date'].unique()),columns=['Date','Factor',11.0,'beta_1','beta_10'])

        Logmodel1 = Logmodel1.merge(scores,how = "left",on = ["Date","Factor"])
        SVMmodel1 = SVMmodel1.merge(scores,how = "left",on = ["Date","Factor"])
        Logmodel1 = Logmodel1[["Ticker","Date","Price","Sector","Factor","Factor Value"]]
        SVMmodel1 = SVMmodel1[["Ticker","Date","Price","Sector","Factor","Factor Value"]]
        Logmodel1['Factor Value'] =  Logmodel1.groupby(['Date','Factor'])['Factor Value'].transform(lambda x:(x-x.mean())/x.std())
        SVMmodel1['Factor Value'] =  SVMmodel1.groupby(['Date','Factor'])['Factor Value'].transform(lambda x:(x-x.mean())/x.std())
        Logmodel1['Factor Value2'] = Logmodel1.groupby(["Date","Ticker"])['Factor Value'].transform(lambda x:x.mean())
        SVMmodel1['Factor Value2'] = SVMmodel1.groupby(["Date","Ticker"])['Factor Value'].transform(lambda x:x.mean())
        Logmodel1["Factor"] = "Log Model Combination"
        SVMmodel1["Factor"] = "SVM Combination"
        Logmodel1 = Logmodel1.groupby(['Date','Ticker']).first().reset_index()
        SVMmodel1 = SVMmodel1.groupby(['Date','Ticker']).first().reset_index()
        Logmodel1['Factor Value'] = Logmodel1['Factor Value2']
        SVMmodel1['Factor Value'] = SVMmodel1['Factor Value2']
        Logmodel1  = Logmodel1[factors.columns]
        SVMmodel1  = SVMmodel1[factors.columns]
        comb_n = Logmodel1.append(SVMmodel1)
        comb_n.reset_index(drop=True,inplace=True)
        comb_n = ranking(comb_n,'1M',10)[0]
        comb_n.drop(11.0,axis=1,inplace=True)

        dat = Logmodel1['Date'].unique()
        for d in dat:
            print(d)

            scores['z-score'] = scores.groupby(['Date','Factor'])['Factor Value'].transform(lambda x:(x-x.mean())/x.std())
            ls = SVMmodel['Factor'][SVMmodel['Date']==d].values
            ll = Logmodel['Factor'][Logmodel['Date']==d].values
            scores_sv = scores[scores['Factor'].isin(ls)]
            scores_sv = scores_sv[(scores_sv['Date']<=pd.to_datetime(d)) & (scores_sv['Date']>=pd.to_datetime(d)-reld(months=48))]
            scores_l = scores[scores['Factor'].isin(ll)]
            scores_l = scores_l[(scores_l['Date']<=pd.to_datetime(d)) & (scores_l['Date']>=pd.to_datetime(d)-reld(months=48))]
            scores_sv['Factor Value'] = scores_sv[scores_sv['Factor'].isin(ls)].groupby(['Date','Ticker'])['z-score'].transform(lambda x:x.mean())
            scores_l['Factor Value'] = scores_l[scores_l['Factor'].isin(ll)].groupby(['Date','Ticker'])['z-score'].transform(lambda x:x.mean())

            scores_sv["Factor"] = "SVM Combination"
            scores_l["Factor"] = "Log Model Combination"
            scores_l.sort_values('Date',inplace=True)
            scores_sv.sort_values('Date',inplace=True)
            scores_sv = scores_sv.groupby(['Date','Ticker']).first().reset_index()
            scores_l = scores_l.groupby(['Date','Ticker']).first().reset_index()
            scores_sv = scores_sv[factors.columns]
            scores_l = scores_l[factors.columns]

            comb1 = scores_l.append(scores_sv)

            comb1.reset_index(drop=True,inplace=True)
            comb1.sort_values('Date',inplace=True)
            comb2 = ranking(comb1,'1M',10)
            comb_n1 = beta_neutral(comb2[0])

            beta_sv[:].loc[d] = comb_n1[['Date','Factor',11.0,'beta_1','beta_10']][(comb_n1['Date']==d) & (comb_n1['Factor']=='SVM Combination')].values[0]
            beta_log[:].loc[d] = comb_n1[['Date','Factor',11.0,'beta_1','beta_10']][(comb_n1['Date']==d) & (comb_n1['Factor']=='Log Model Combination')].values[0]

        beta_full = beta_sv.append(beta_log)
        beta_full.sort_values('Date',inplace=True)
        beta_full['Date'] = pd.to_datetime(beta_full['Date'])
        comb_n['Date'] = pd.to_datetime(comb_n['Date'])
        comb_n = comb_n.merge(beta_full[['Date','Factor','beta_1','beta_10',11.0]],on=['Date','Factor'],how='left')
    return comb_n
    
    


# In[1167]:



def getcorrs():
    q2 = asas_bn[asas_bn['Factor'] == "Log Model Combination"]
    q2 = q2[["Date",11.0]]
    q2 = q2.merge(famafrenchdata)
    return q2.corr()
    


def results(df_strat,famafrenchdata,strat,beta,alp):
    df_new = df_strat.copy()
    strat = 'Log Model Combination'
    ff_data = famafrenchdata[famafrenchdata['Date']>="2012-01-30"]

    df_new = df_new[df_new['Factor']==strat]
    df_new = df_new[['Date','Factor',11.0]]

    df_new = df_new[df_new['Date']>="2012-01-30"]
    b1 = beta['Mkt-RF'].loc[strat]
    b2 = beta['SMB'].loc[strat]
    b3 = beta['HML'].loc[strat]
    b4 = beta['RMW'].loc[strat]
    b5 = beta['CMA'].loc[strat]

    ff_data['Fama French']= b1*ff_data['Mkt-RF'] + b2*ff_data['SMB'] + b3*ff_data['HML'] + b4*ff_data['RMW'] + b5*ff_data['CMA']
    df_total = ff_data.merge(df_new[['Date',11.0]],how='left',on=['Date'])

    df_total['Strategy'] = df_total[11.0]
    df_total['Excess Return'] = df_total['Strategy'] - df_total['Fama French']
    df_total = df_total[['Date','Fama French','Strategy','Excess Return']].set_index('Date')
    df_total = df_total.reset_index()
    cal_ret = df_total.groupby(df_total['Date'].dt.year).agg('sum')
   
    return df_total,cal_ret



def reg_LS_spread_vs_FFM5_alp (df1,famafrenchdata):
    df_reg = pd.DataFrame(index = df1.index, columns =['Date','Factor',11.0])
    df_reg['Date'] = df1['Date']
    df_reg['Factor'] = df1['Factor']
    df_reg = df_reg[df_reg['Date']>="2012-01-31"]
    df_reg[11.0] = df1[11.0]
    df_reg = pd.merge(df_reg, famafrenchdata[['Date','Mkt-RF','SMB','HML','RMW','CMA',]],on='Date', how='left')
    group_factor=df_reg.groupby('Factor')
    fact_list = df_reg['Factor'].unique()
    ff_list=['Mkt-RF','SMB','HML','RMW','CMA']

    reg_beta = pd.DataFrame(index=fact_list, columns= ff_list)
    reg_resid = pd.DataFrame(index=df1['Date'].unique(), columns=fact_list)
    reg_resid = reg_resid[reg_resid.index>="2012-01-31"]

    for j in fact_list:
        fact = group_factor.get_group(j)

        rhs = sm.add_constant(fact[ff_list])
        lhs = fact[11]
        res = sm.OLS(lhs, rhs, missing='drop').fit()
        for i in range (5):
            reg_beta.loc[j][i]=res.params[i+1]

        reg_resid[j]= np.array(res.resid+res.params[0])

    residual_formatted=reg_resid.T.reset_index()
    residual_formatted=residual_formatted.melt(id_vars =['index'])
    residual_formatted.columns=['Factor','Date','Residual']
   
   
    return residual_formatted, reg_beta,np.array(res.params[0])