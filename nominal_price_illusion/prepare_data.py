import pandas as pd
import numpy as np
import numpy.ma as ma
import os
from tqdm import tqdm
import warnings
from optparse import OptionParser
from collections import deque
import datetime
from dateutil.relativedelta import relativedelta
import sys 
from scipy import stats
from scipy.stats.mstats import winsorize


class price_illusion_database():
    def __init__(self,start_date, end_datef):
        self.start_date = start_date
        self.end_date = end_date
        self.trading_days_path = '/data/raw/WIND/ASHARECALENDAR.txt'
        self.trading_days = self.get_trading_days(self.start_date, self.end_date)
        self.consensus_price_pool_path = '/data-zero/raw/GOGOAL_merged/Consensus/{}/{}/{}/CON_TARGET_PRICE_STK.txt'
        self.consensus_eps_pool = '/data-zero/raw/GOGOAL_merged/Consensus/{}/{}/{}/CON_FORECAST_STK.txt'
        self.BASEDATA_PATH_TEMPLATE = '/data/cooked/BaseData/{}/{}/{}/BASEDATA.txt' 
        self.JY_BS_PATH_TEMPLATE = '/data/raw/JY/Fundamental/{}/{}/{}/LC_BALANCESHEETALL_MERGED.txt'
        self.JY_IS_PATH_TEMPLATE = '/data/raw/JY/Fundamental/{}/{}/{}/LC_INCOMESTATEMENTALL_MERGED.txt'
        self.industry_TEMPLATE='/data-platform/ccxd/dev/risk_model/industry/{}/{}/{}/industry.txt'
        self.Save_Path_Template = '/data/public_transfer/liyihan/price_illusion/prepared_data.csv'

        
    #获取交易所交易日数据
    def get_trading_days(self, start_date=None, end_date=None, back_days=0, parser=False):
        # logger.info("start get trading days, {}, {}, {}".format(start_date, end_date, back_days))
        calendar_path =self.trading_days_path
        full_calendar = pd.read_csv(calendar_path, dtype=str)
        calendar = full_calendar
        if start_date is not None:
            calendar = calendar[calendar["SECU_CODE"] >= start_date]
        if end_date is not None:
            calendar = calendar[calendar["SECU_CODE"] <= end_date]
        if back_days > 0:
            if start_date is None:
                res = calendar.loc[calendar.index[-back_days:], "SECU_CODE"]#返回calendar中最后一行里的SECU_CODE列
            else:
                back_days_list = full_calendar[full_calendar["SECU_CODE"] < start_date]
                res = back_days_list["SECU_CODE"].iloc[-back_days:].append(calendar["SECU_CODE"])
        else:
            res = calendar["SECU_CODE"]
        if parser:
            res = pd.to_datetime(res)#转换为时间序列
        # logger.info("finish get trading days, {}, {}, {}".format(start_date, end_date, back_days))
        return res.tolist()


    #获取预测eps
    def load_forecast_eps(self, start_date, end_date, back_years=1):
        start_date = str(int(start_date[:4])-back_years) + '0101'
        actual_trading_days=self.get_trading_days(start_date,end_date)
        secu_code_list=[]
        con_date_list=[]
        con_eps_list=[]
        print("读取con_eps数据：")
        for dt in tqdm(actual_trading_days):
            path=self.consensus_eps_pool.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
            forec = pd.read_csv(path, sep='|', na_values=['None'], parse_dates=['CON_DATE'])#读取该日所有基本面数据，并将公布日期设置为日期型数据
            if forec.empty:
                continue
            forec=forec[(forec['CON_OR']==0) & (forec['CON_EPS_TYPE']==1)]#CON_OR==0的数据为依据真实财务报表的预测值, CON_EPS_TYPE==1表示90天内有3家以上券商给出了预测值
            forec['CON_DATE'] = forec['CON_DATE'].dt.strftime('%Y%m')#时间数据转换回str，方便操作，日期精确到月份
            secu_code_list+=forec['STOCK_CODE'].tolist()
            con_date_list+=forec['CON_DATE'].tolist()
            con_eps_list+=forec['CON_EPS'].tolist()
        fc_df = pd.DataFrame({'STOCK_CODE':secu_code_list,'CON_DATE':con_date_list, 'CON_EPS':con_eps_list})
        fc_df = fc_df.sort_values(['STOCK_CODE','CON_DATE','CON_EPS'], ascending=[True, False, False])#排序
        #求目标eps月平均值
        res_df = fc_df.groupby(['STOCK_CODE','CON_DATE'],as_index=False).agg({'CON_EPS':'mean'})
        res_df = res_df.sort_values(['STOCK_CODE','CON_DATE','CON_EPS'], ascending=[True, False, False])#排序
        res_df.rename(columns={'STOCK_CODE':'SECU_CODE'},inplace=True)
        return res_df[['SECU_CODE','CON_DATE','CON_EPS']]

    #获取预测price
    def load_forecast_price(self, start_date, end_date, back_years=1):
        start_date = str(int(start_date[:4])-back_years) + '0101'
        actual_trading_days=self.get_trading_days(start_date,end_date)
        secu_code_list=[]
        con_date_list=[]
        con_target_price_list=[]
        print("读取con_price数据：")
        for dt in tqdm(actual_trading_days):
            path=self.consensus_price_pool_path.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
            forec = pd.read_csv(path, sep='|', na_values=['None'], parse_dates=['CON_DATE'])#读取该日所有基本面数据，并将公布日期设置为日期型数据
            if forec.empty:
                continue
            forec=forec[forec['CON_TARGET_PRICE_TYPE']==1]#CON_TARGET_PRICE_TYPE==1表示90天内有3家以上券商给出了预测值
            forec['CON_DATE'] = forec['CON_DATE'].dt.strftime('%Y%m')#时间数据转换回str，方便操作，日期精确到月份
            secu_code_list+=forec['STOCK_CODE'].tolist()
            con_date_list+=forec['CON_DATE'].tolist()
            con_target_price_list+=forec['CON_TARGET_PRICE'].tolist()
        fc_df = pd.DataFrame({'STOCK_CODE':secu_code_list,'CON_DATE':con_date_list, 'CON_TARGET_PRICE':con_target_price_list})
        fc_df = fc_df.sort_values(['STOCK_CODE','CON_DATE','CON_TARGET_PRICE'], ascending=[True, False, False])#排序
        #求目标price月平均值
        res_df = fc_df.groupby(['STOCK_CODE','CON_DATE'],as_index=False).agg({'CON_TARGET_PRICE':'mean'})
        res_df = res_df.sort_values(['STOCK_CODE','CON_DATE','CON_TARGET_PRICE'], ascending=[True, False, False])#排序
        res_df.rename(columns={'STOCK_CODE':'SECU_CODE'},inplace=True)
        return res_df[['SECU_CODE','CON_DATE','CON_TARGET_PRICE']]

    #获得dt前一个月的最后一个交易日
    def get_lag_date(self,dt):
        if dt[4:6]=='01':
            month='12'
            year=str(int(dt[:4])-1)
            month_trading_days=self.get_trading_days(year+month+'01',year+month+'31')
            lag_date=month_trading_days[-1]
        else:
            temp=str(int(dt[:6])-1)
            month_trading_days=self.get_trading_days(temp+'01',temp+'31')
            lag_date=month_trading_days[-1]
        return lag_date

    #读取某天的basedata并以dataframe形式输出
    def get_basedata(self,dt):
        basedata_path = self.BASEDATA_PATH_TEMPLATE.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
        basedata_df = pd.read_csv(basedata_path, sep='|', na_values=['None'])
        return basedata_df

    #获得股票的上市时间
    def get_age(self,dt):
        age_df=self.get_basedata(dt)
        age_df.rename(columns={'LISTEDDATE':'Age'},inplace=True)
        return age_df[['SECU_CODE','Age']]

    #计算未来股票回报的心理预期
    def cal_expect_return(self,dt,full_con_price):
        exp_return_df=self.get_basedata(self.get_lag_date(dt))
        exp_return_df=exp_return_df[['SECU_CODE','CLOSEPRICE']]
        exp_return_df.rename(columns={'CLOSEPRICE':'lag_CLOSEPRICE'},inplace=True)
        temp_df = full_con_price[full_con_price['CON_DATE']==(dt[:6])]
        exp_return_df['CON_TARGET_PRICE'] = exp_return_df['SECU_CODE'].map(dict(zip(temp_df['SECU_CODE'],temp_df['CON_TARGET_PRICE'])))
        exp_return_df['exp_RETURN'] =  exp_return_df['CON_TARGET_PRICE']/exp_return_df['lag_CLOSEPRICE']-1
        return exp_return_df[['SECU_CODE','exp_RETURN']]
           
    #计算对应滞后一个月的日期（年+月）
    def cal_lag_monthdate(self,dt):
        if dt[4:6]=='01':
            return (str(int(dt[:4])-1)+'12')
        else:
            return(str(int(dt[:6])-1))

    #得到上月的收盘价
    def get_lag_price(self,dt):
        lag_price_df=self.get_basedata(self.get_lag_date(dt))
        lag_price_df=lag_price_df[['SECU_CODE','CLOSEPRICE']]
        lag_price_df.rename(columns={'CLOSEPRICE':'PRICE'},inplace=True)
        lag_price_df['PRICE']=lag_price_df['PRICE']/100 #论文中股票价格用收盘价/100的值衡量
        return lag_price_df[['SECU_CODE','PRICE']]

    #获得滞后的流通市值参考日期,t年5月到t+1年4月末的取值基于t-1报告期都基于t年4月末的流通市值
    def get_lag_condate(self, dt):
        if int(dt[4:6]) >= 5:
            lag_condate_list=self.get_trading_days(str(int(dt[:4])-1)+'04'+'01',str(int(dt[:4])-1)+'04'+'30')
            lag_condate=lag_condate_list[-1]
        if int(dt[4:6]) <= 4:
            lag_condate_list=self.get_trading_days(str(int(dt[:4])-2)+'04'+'01',str(int(dt[:4])-2)+'04'+'30')
            lag_condate=lag_condate_list[-1]
        return lag_condate

    #得到对应的滞后流通市值
    def get_lag_stksize(self,dt):
        stksize_df = self.get_basedata(self.get_lag_condate(dt))
        stksize_df=stksize_df[['SECU_CODE','TOTALFLOATSHARES']]
        stksize_df.rename(columns={'TOTALFLOATSHARES':'lag_TOTALFLOATSHARES'},inplace=True)
        stksize_df['lnSIZE']=np.log(stksize_df['lag_TOTALFLOATSHARES'])
        return stksize_df[['SECU_CODE','lnSIZE']]

    #计算分析师对股票未来盈利的盈余修正
    def cal_revision(self,dt,full_con_price):
        di=self.get_lag_date(dt)
        #获得上个月对股票的心理预期
        return_lag=self.get_basedata(self.get_lag_date(di))
        return_lag=return_lag[['SECU_CODE','CLOSEPRICE']]        
        return_lag.rename(columns={'CLOSEPRICE':'lag_CLOSEPRICE'},inplace=True)
        temp_d1 = full_con_price[full_con_price['CON_DATE']==(di[:6])]
        return_lag['CON_TARGET_PRICE'] = return_lag['SECU_CODE'].map(dict(zip(temp_d1['SECU_CODE'],temp_d1['CON_TARGET_PRICE'])))
        return_lag['exp_RETURN_lag'] =  return_lag['CON_TARGET_PRICE']/return_lag['lag_CLOSEPRICE']-1
        return_lag=return_lag[['SECU_CODE','exp_RETURN_lag']]
        #获得当月的股票心理预期
        revision_df=self.get_basedata(di)
        revision_df=revision_df[['SECU_CODE','CLOSEPRICE']]        
        revision_df.rename(columns={'CLOSEPRICE':'lag_CLOSEPRICE'},inplace=True)
        temp_d2 = full_con_price[full_con_price['CON_DATE']==(dt[:6])]
        revision_df['CON_TARGET_PRICE'] = revision_df['SECU_CODE'].map(dict(zip(temp_d2['SECU_CODE'],temp_d2['CON_TARGET_PRICE'])))
        revision_df['exp_RETURN'] =  revision_df['CON_TARGET_PRICE']/revision_df['lag_CLOSEPRICE']-1
        #合并当月和前月的心理预期
        revision_df['exp_RETURN_lag']=revision_df['SECU_CODE'].map(dict(zip(return_lag['SECU_CODE'],return_lag['exp_RETURN_lag'])))
        #获得上月末的流通市值
        temp=self.get_lag_stksize(dt)
        revision_df['lag_lnSIZE']=revision_df['SECU_CODE'].map(dict(zip(temp['SECU_CODE'],temp['lnSIZE'])))
        revision_df['REVISION'] =  (revision_df['exp_RETURN']-revision_df['exp_RETURN_lag'])/revision_df['lag_lnSIZE']
        return revision_df[['SECU_CODE','REVISION']]


    #读取每天的dataframe，并获得每天的return和return_mkt
    def load_basedata_return(self, start_date, end_date,back_years=1):
        start_date = str(int(start_date[:4])-back_years) + '0101'
        actual_trading_days=self.get_trading_days(start_date,end_date)
        secu_code_list=[]
        tradingday_list=[]
        pre_closeprice_list=[]
        price_list=[]
        totalshare_list=[]
        print("读取basedata_return数据：")
        for dt in tqdm(actual_trading_days):
            path=self.BASEDATA_PATH_TEMPLATE.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
            base = pd.read_csv(path, sep='|', na_values=['None'], parse_dates=['TRADINGDAY'])
            if base.empty:
                continue
            base = base[~base['CLOSEPRICE'].isna()]
            base['TRADINGDAY'] = base['TRADINGDAY'].dt.strftime('%Y%m%d')#时间数据转换回str，方便操作，日期精确到月份
            secu_code_list+=base['SECU_CODE'].tolist()
            tradingday_list+=base['TRADINGDAY'].tolist()
            pre_closeprice_list+=base['PREVCLOSEPRICE'].tolist()
            price_list+=base['CLOSEPRICE'].tolist()
            totalshare_list+=base['TOTALSHARES'].tolist()
            # print('base[CLOSEPRICE]', np.nonzero(np.isnan(np.array(base['CLOSEPRICE'])))[0].shape[0] != 0)
            # print('base[PREVCLOSEPRICE]', np.nonzero(np.isnan(np.array(base['PREVCLOSEPRICE'])))[0].shape[0] != 0)    
        bd_df = pd.DataFrame({'SECU_CODE':secu_code_list,'TRADINGDAY':tradingday_list, 'PREVCLOSEPRICE':pre_closeprice_list,'CLOSEPRICE':price_list,'TOTALSHARES':totalshare_list})
        res_df = bd_df.sort_values(['SECU_CODE','TRADINGDAY','PREVCLOSEPRICE','CLOSEPRICE','TOTALSHARES'], ascending=[True, False, False, False,False])#排序
        res_df['TRADING_month']=[x[:6] for x in res_df['TRADINGDAY']]#方便操作，日期精确到月份
        res_df['RETURN_day']=res_df['CLOSEPRICE']/res_df['PREVCLOSEPRICE']-1
        # print('res_df[CLOSEPRICE]', np.nonzero(np.isnan(np.array(res_df['CLOSEPRICE'])))[0].shape[0] != 0)
        # print('res_df[PREVCLOSEPRICE]', np.nonzero(np.isnan(np.array(res_df['PREVCLOSEPRICE'])))[0].shape[0] != 0)
        # print('res_df[RETURN_day]', np.nonzero(np.isnan(np.array(res_df['RETURN_day'])))[0].shape[0] != 0)
        temp=res_df.groupby('TRADINGDAY',as_index=False).apply(lambda x: np.average(x['RETURN_day'],weights=x['TOTALSHARES']))#按日算加权平均
        temp = temp.rename(columns={None:'RETURN_mkt'})
        res_df['RETURN_mkt']=res_df['TRADINGDAY'].map(dict(zip(temp['TRADINGDAY'],temp['RETURN_mkt'])))
        return res_df[['SECU_CODE','TRADINGDAY','TRADING_month','RETURN_day','RETURN_mkt']]

    
    #得到股票上月收益率
    def get_lag_return(self,dt,full_basedata_return):
        lag_return_df=full_basedata_return[full_basedata_return['TRADING_month']==self.cal_lag_monthdate(dt)]
        lag_return_df=lag_return_df.groupby(['SECU_CODE'], as_index=False).agg({'RETURN_day':'sum'})
        lag_return_df.rename(columns={'RETURN_day':'lag_RETURN'},inplace=True)
        return lag_return_df[['SECU_CODE','lag_RETURN']]
        
    #计算股票[-12,-2]月内的收益率之和
    def get_sum_return(self,dt,full_basedata_return):
        template=full_basedata_return.groupby(['SECU_CODE', 'TRADING_month'], as_index=False).agg({'RETURN_day':'sum'})
        template.rename(columns={'RETURN_day':'RETURN'},inplace=True)
        dt=datetime.datetime.strptime(dt,"%Y%m%d")
        start=dt+relativedelta(months=-12)
        start=start.strftime('%Y%m')
        end=dt+relativedelta(months=-2)
        end=end.strftime('%Y%m')
        sum_return_df=template[(template['TRADING_month']>=start)&(template['TRADING_month']<=end)]
        sum_return_df = sum_return_df.groupby(['SECU_CODE'], as_index=False).agg({'RETURN':'sum'})
        sum_return_df.rename(columns={'RETURN':'sum_RETURN'},inplace=True)
        return sum_return_df[['SECU_CODE','sum_RETURN']]

    #计算股票过去12个月内回报率的标准差
    def get_volatility(self,dt,full_basedata_return):
        template=full_basedata_return.groupby(['SECU_CODE', 'TRADING_month'], as_index=False).agg({'RETURN_day':'sum'})
        template.rename(columns={'RETURN_day':'RETURN'},inplace=True)
        dt=datetime.datetime.strptime(dt,"%Y%m%d")
        start=dt+relativedelta(months=-12)
        start=start.strftime('%Y%m')
        end=dt+relativedelta(months=-1)
        end=end.strftime('%Y%m')
        volatility_df=template[(template['TRADING_month']>=start)&(template['TRADING_month']<=end)]
        volatility_df=volatility_df.groupby(['SECU_CODE'],as_index=False).agg({'RETURN':'std'}) 
        volatility_df.rename(columns={'RETURN':'VOLATILITY'},inplace=True)
        return volatility_df[['SECU_CODE','VOLATILITY']]

    #计算股票的beta值
    def cal_beta(self,dt,full_basedata_return):
        dt=datetime.datetime.strptime(dt,"%Y%m%d")
        start=dt+relativedelta(months=-12)
        start=start.strftime('%Y%m')
        end=dt+relativedelta(months=-1)
        end=end.strftime('%Y%m')
        beta_df=full_basedata_return[(full_basedata_return['TRADING_month']>=start)&(full_basedata_return['TRADING_month']<=end)]
        def covr(x):
            a=np.array(x['RETURN_day'])
            b=np.array(x['RETURN_mkt'])
            res = np.cov(a ,b)
            return res[0][1]
        temp=beta_df.groupby(['SECU_CODE'],as_index=False).apply(covr)
        temp=temp.rename(columns={None:'covr'})
        var_mkt=np.var(beta_df['RETURN_mkt'])
        #   def time_series_regression(df):
            #   Y = np.array([df['RETURN_day']])
            #   X = np.array([df['RETURN_mkt']])
            #   beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)  # OLS矩阵求解式((X'X)^-1)X'Y
            #   return beta
        #   temp=beta_df.groupby(['SECU_CODE'],as_index=False).apply(time_series_regression)
        #   temp=temp.rename(columns={None:'beta'})
        beta_df=beta_df.groupby(['SECU_CODE'],as_index=False).head(1)
        beta_df['covr']= beta_df['SECU_CODE'].map(dict(zip(temp['SECU_CODE'],temp['covr'])))
        #   beta_df['beta']= beta_df['SECU_CODE'].map(dict(zip(temp['SECU_CODE'],temp['beta'])))
        beta_df['beta']=beta_df['covr']/var_mkt
        return beta_df[['SECU_CODE','beta']]

    def get_lag_rptdate(self, dt):
        if int(dt[4:6]) >= 5:
            lag_rptdate=str(int(dt[:4])-1)+'12'
        else:
            lag_rptdate=str(int(dt[:4])-2)+'12'
        return lag_rptdate

    #获取资产负债表
    def load_bs(self, start_date, end_date, back_years=2):
        #滞后到back_years年前的年初
        start_date = str(int(start_date[:4])-back_years) + '0101'
        actual_trading_days = self.get_trading_days(start_date, end_date)
        #由于基本面数据是每天出一部分公司，所以这里采用list类型的数据将所有已知信息拼接，然后根据需要摘取
        secu_code_list = []
        infopubl_date_list = []
        end_date_list = []
        totalassets_list = []
        totalshareholderequity_list = []
        intangibleassets_list=[]
        print("读取balance_sheet数据：")
        for dt in tqdm(actual_trading_days):
            path = self.JY_BS_PATH_TEMPLATE.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
            funda = pd.read_csv(path, sep='|', na_values=['None'], parse_dates=['INFOPUBLDATE','ENDDATE'],low_memory=False)#读取该日所有基本面数据，并将公布日期设置为日期型数据
            if funda.empty:
                continue
            funda = funda[(funda['IFMERGED']==1) & (funda['BULLETINTYPE']==20) & (~funda['INTANGIBLEASSETS'].isna())]#取合并报表
            funda = funda[(funda['ENDDATE'].dt.month==12)&(funda['IFMERGED']==1)&(funda['IFADJUSTED']==2)]
            funda['INFOPUBLDATE'] = funda['INFOPUBLDATE'].dt.strftime('%Y%m%d')
            funda['ENDDATE'] = funda['ENDDATE'].dt.strftime('%Y%m')
            #计算无形资产占总资产比例
            secu_code_list += funda['SECU_CODE'].tolist()
            infopubl_date_list += funda['INFOPUBLDATE'].tolist()
            end_date_list += funda['ENDDATE'].tolist()
            totalshareholderequity_list += funda['TOTALSHAREHOLDEREQUITY'].tolist()
            totalassets_list += funda['TOTALASSETS'].tolist()
            intangibleassets_list += funda['INTANGIBLEASSETS'].tolist()
        res_df = pd.DataFrame({'SECU_CODE':secu_code_list,'INFOPUBLDATE':infopubl_date_list, 'ENDDATE':end_date_list, 'TOTALASSETS':totalassets_list, 'TOTALSHAREHOLDEREQUITY':totalshareholderequity_list,'INTANGIBLEASSETS':intangibleassets_list})#利用list创建dataframe
        res_df = res_df.sort_values(['SECU_CODE','ENDDATE','INFOPUBLDATE','TOTALASSETS','TOTALSHAREHOLDEREQUITY','INTANGIBLEASSETS'], ascending=[True, False, False,False,False,False])#排序
        res_df.loc[np.isnan(res_df['INTANGIBLEASSETS']),'INTANGIBLEASSETS'] = 0
        res_df['INTANGIBLESIZE'] = res_df['INTANGIBLEASSETS'] / res_df['TOTALASSETS']#获得无形资产比例
        res_df=res_df[['SECU_CODE','INFOPUBLDATE','ENDDATE','TOTALASSETS','TOTALSHAREHOLDEREQUITY','INTANGIBLESIZE']]
        return res_df

    #获得无形资产资产比重
    def get_intangible_size(self,dt,full_bs):
        di=self.get_lag_rptdate(dt)
        intangible_size_df=full_bs[full_bs['ENDDATE'] == (di)]
        return intangible_size_df[['SECU_CODE','INTANGIBLESIZE']]

    #获得账面市值比的对数
    def get_bm_df(self,dt,full_bs):
        di=self.get_lag_rptdate(dt)
        bm_df=full_bs[full_bs['ENDDATE'] == (di)]
        bm_df=bm_df[['SECU_CODE','TOTALSHAREHOLDEREQUITY']]
        temp=self.get_basedata(self.get_lag_date(dt))
        bm_df['TOTALFLOATSHARES']=bm_df['SECU_CODE'].map(dict(zip(temp['SECU_CODE'],temp['TOTALFLOATSHARES'])))
        bm_df['lnBM']=np.log(bm_df['TOTALSHAREHOLDEREQUITY']/bm_df['TOTALFLOATSHARES'])
        return bm_df[['SECU_CODE','lnBM']]

    #读取净利润
    def load_netprofit(self, start_date, end_date, back_years=2):
        #滞后到两年前的年初 确保数据充足
        start_date = str(int(start_date[:4])-back_years) + '0101'
        actual_trading_days = self.get_trading_days(start_date, end_date)
        #以list和list加的形式将基本面数据都存储进来 最后转换成dataframe
        secu_code_list = []
        infoPubl_date_list = []
        end_date_list = []
        netprofit_list = []
        adjusted_list = []
        print("读取net profit数据：")
        for dt in tqdm(actual_trading_days):
            path = self.JY_IS_PATH_TEMPLATE.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
            funda = pd.read_csv(path, sep='|', na_values=['None'], parse_dates=['INFOPUBLDATE','ENDDATE'])#读入收益表所有数据，并把公式日期和报告期设置为datetime类型
            if funda.empty:
                continue
            funda = funda[(funda['IFMERGED']==1) & (funda['BULLETINTYPE']==20 ) & (~funda['NETPROFIT'].isna())]#取合并报表
            # if len(funda) > 0:
                # print(len(funda))
                # input()
            funda = funda[(funda['ENDDATE'].dt.month==12)&(funda['IFMERGED']==1)&(funda['IFADJUSTED']==2)]
            funda['INFOPUBLDATE'] = funda['INFOPUBLDATE'].dt.strftime('%Y%m%d')
            funda['ENDDATE'] = funda['ENDDATE'].dt.strftime('%Y%m')#为方便操作 报告期精确到月份
            secu_code_list += funda['SECU_CODE'].tolist()
            infoPubl_date_list += funda['INFOPUBLDATE'].tolist()
            end_date_list += funda['ENDDATE'].tolist()
            netprofit_list += funda['NETPROFIT'].tolist()
            adjusted_list += funda['IFADJUSTED'].tolist()
        res_df = pd.DataFrame({'SECU_CODE':secu_code_list,'INFOPUBLDATE':infoPubl_date_list, "ENDDATE":end_date_list,'IFADJUSTED':adjusted_list,'NETPROFIT':netprofit_list})#dataframe数据 行为股票代码 列为每股盈利和信息公布日、报告期 
        res_df = res_df.sort_values(['SECU_CODE','ENDDATE','NETPROFIT'], ascending=[True,  False, False])#排序
        res_df = res_df[~(res_df['NETPROFIT'].isna())]#去掉空值
        return res_df


    #利用dt已知最新报告期数据计算年度总资产同比增长
    def cal_Ag(self, dt, full_bs):
        di=self.get_lag_rptdate(dt)
        di_lag=self.get_lag_rptdate(di)
        totalassets=full_bs[full_bs['ENDDATE'] <= (di)]
        Ag_df= totalassets[['SECU_CODE','TOTALASSETS']].groupby('SECU_CODE',as_index=False).head(1)
        #去掉所用的最新季度数据
        totalassets_lag=full_bs[full_bs['ENDDATE'] <= (di_lag)]
        Ag_lag_df= totalassets_lag[['SECU_CODE','TOTALASSETS']].groupby('SECU_CODE',as_index=False).head(1)
        #计算TOTALASSETS环比增长
        Ag_df['lag_TOTALASSETS'] = Ag_df['SECU_CODE'].map(dict(zip(Ag_lag_df['SECU_CODE'],Ag_lag_df['TOTALASSETS'])))
        Ag_df['Ag'] = Ag_df['TOTALASSETS']/Ag_df['lag_TOTALASSETS']-1
        return Ag_df[['SECU_CODE','Ag']]#输出当天

    #利用dt已知最新报告期数据计算ROE
    def cal_ROE(self, dt, full_netprofit,full_bs):
        #获取最新利润表的净利润
        di=self.get_lag_rptdate(dt)
        netprofit=full_netprofit[full_netprofit['ENDDATE'] <= (di)]
        np_df = netprofit[['SECU_CODE','NETPROFIT']].groupby('SECU_CODE',as_index=False).head(1)
        #获得最新资产负债表的所有者权益
        tshe=full_bs[full_bs['INFOPUBLDATE'] <= (di)]
        tshe_df = tshe[['SECU_CODE','TOTALSHAREHOLDEREQUITY']].groupby('SECU_CODE',as_index=False).head(1)
        #合并
        tshe_df['NETPROFIT'] = tshe_df['SECU_CODE'].map(dict(zip(np_df['SECU_CODE'],np_df['NETPROFIT'])))
        #计算当年总资产同比增长率
        tshe_df['ROE']=tshe_df['NETPROFIT']/tshe_df['TOTALSHAREHOLDEREQUITY']
        ROE_df = tshe_df[['SECU_CODE','ROE']]
        return ROE_df

    #读取某天的industry并以dataframe形式输出
    def get_industry_base(self, dt):
        industry_base_path = self.industry_TEMPLATE.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
        industry_base_df = pd.read_csv(industry_base_path, sep='|', na_values=['None'])
        return industry_base_df

    #获得股票的industry
    def get_industry(self,dt):
        industry_df=self.get_industry_base(dt)
        industry_df.rename(columns={'SW2021':'INDUSTRY'},inplace=True)
        return industry_df[['SECU_CODE','INDUSTRY']]

    def save_df(self, df, dt):
        date = pd.to_datetime(dt)
        save_path = self.Save_Path_Template.format(date.year,date.month,date.day)
        if not os.path.exists(os.path.dirname(save_path)):#判断文件路径所在文件夹是否存在 不存在则创建
            os.makedirs(os.path.dirname(save_path))
        df.to_csv(save_path,sep='|', na_rep='None', float_format='%.6f',index=False)

    def run(self):
        full_bs = self.load_bs(self.start_date,self.end_date)
        full_netprofit = self.load_netprofit(self.start_date, self.end_date)#读取净利润
        full_con_eps=self.load_forecast_eps(self.start_date,self.end_date)
        full_con_price=self.load_forecast_price(self.start_date,self.end_date)
        full_basedata_return=self.load_basedata_return(self.start_date,self.end_date)
        prepared_data = pd.DataFrame()
        for dt in tqdm(self.trading_days):
            exp_return_df=self.cal_expect_return(dt,full_con_price)
            lag_price=self.get_lag_price(dt)
            stksize_df=self.get_lag_stksize(dt)
            revision_df=self.cal_revision(dt,full_con_price)
            sum_return_df=self.get_sum_return(dt,full_basedata_return)
            lag_return_df=self.get_lag_return(dt,full_basedata_return)
            revision_df=self.cal_revision(dt,full_con_price)
            beta_df=self.cal_beta(dt,full_basedata_return)
            bm_df=self.get_bm_df(dt,full_bs)
            Ag_df=self.cal_Ag( dt, full_bs)
            ROE_df = self.cal_ROE(dt, full_netprofit,full_bs)
            volatility_df=self.get_volatility(dt,full_basedata_return)
            age_df=self.get_age(dt)
            intangible_size_df=self.get_intangible_size(dt,full_bs)
            industry_df=self.get_industry(dt)
            #拼接结果
            res_df = pd.merge(exp_return_df, lag_price,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, revision_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, beta_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, stksize_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, bm_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, Ag_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, ROE_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, sum_return_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, lag_return_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, volatility_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, age_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, intangible_size_df,how='outer',on=['SECU_CODE'])
            res_df = pd.merge(res_df, industry_df,how='outer',on=['SECU_CODE'])
            res_df['DT'] = dt[:6]
            res_df['date']=datetime.datetime.strptime(dt,"%Y%m%d")
            #   res_df['DT'] = res_df['DT'].apply(lambda x: str(x)[0:6])
            #   res_df['DT'] = datetime.datetime.strptime(dt,"%Y%m%d")
            #   res_df['DT'] = res_df['DT'].values.astype('datetime64[M]')
            res_df = res_df[['SECU_CODE','DT','date','exp_RETURN','PRICE','REVISION','beta','lnSIZE','lnBM','Ag','ROE','sum_RETURN','lag_RETURN','VOLATILITY','Age','INTANGIBLESIZE','INDUSTRY']]
            #   删除变量不完全的股票
            res_df.dropna(axis=0, how='any', inplace=True)
            prepared_data = pd.concat([res_df, prepared_data], ignore_index=True)
            prepared_data=prepared_data.groupby(['SECU_CODE','DT'],as_index=False).head(1)
        #对连续变量进行1%的缩尾处理
        prepared_data_list=['exp_RETURN','PRICE','REVISION','beta','lnSIZE','lnBM','Ag','ROE','sum_RETURN','lag_RETURN','VOLATILITY','INTANGIBLESIZE']#需要进行缩尾的列名
        for i in prepared_data_list:
            prepared_data[i]=winsorize(prepared_data[i],limits=[0.01, 0.01],nan_policy='omit')
        prepared_data.to_csv('prepared_data.csv',sep='|', na_rep='None', float_format='%.6f',index=False)
        #描述性统计
        variable_describe=prepared_data[['exp_RETURN','PRICE','REVISION','beta','lnSIZE','lnBM','Ag','ROE','sum_RETURN','lag_RETURN']].describe()
        variable_describe.to_csv('variable_describe.csv',sep='|', na_rep='None', float_format='%.6f',index=True)

                

#%%

if __name__ == '__main__':
    start_date = "20170101"
    end_date = "20181231"
    price_illusion_database = price_illusion_database(start_date,end_date)#创建测试类
    price_illusion_database.run()

    
# %%
