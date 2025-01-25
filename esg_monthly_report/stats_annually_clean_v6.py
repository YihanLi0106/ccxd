# %%
import pandas as pd 
import numpy as np 
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

# 换手率 calmark 基金排名 sharperank

class DataAPI(object):
    def __init__(self):
        self.exg_trading_day_path = '/data/raw/WIND/ASHARECALENDAR.txt'
        self.ib_trading_day_path = '/data/raw/JY/IndexProduct/QT_TRADINGDAYNEW.txt'
        self.ccx_fund_type_path_template = '/data-platform/ccxd/prod/fund_data/{}/{}/{}/fund_type_ccx.csv'
        self.ccx_fund_ret_path_template = '/data-platform/ccxd/prod/fund_data/{}/{}/{}/fund_data.csv'
        self.basedata_path_template = '/data/cooked/BaseData/{}/{}/{}/BASEDATA.txt'
        pass

    def get_table_from_home(self, table_path, fields=None, na_values=['None'], sep='|', dtype={}):
        df = pd.read_csv(table_path, sep=sep, usecols=fields, na_values=na_values,
                         dtype=dtype, low_memory=False, error_bad_lines=False)
        return df


    def get_tradingday_between(self, start_date=None, end_date=None, mkt='Exchange', parser=False):
        trading_days = []
        if mkt == 'IB':
            dt_path = self.ib_trading_day_path
        elif mkt == 'Exchange':
            dt_path = self.exg_trading_day_path
        else:
            raise ValueError('please check the input of mkt')
        rf = open(dt_path, "r")
        for li, line in enumerate(rf):
            if li == 0:
                continue
            if start_date is not None and line.strip() < start_date:
                continue
            if end_date is not None and line.strip() > end_date:
                continue
            else:
                trading_days.append(line.strip())
        rf.close()
        return trading_days

    def get_dt_ccx_fund_type_df(self, dt):
        fp = self.ccx_fund_type_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp, sep=',')
        return data
    
    def get_dt_ccx_fund_data_df(self, dt):
        fp = self.ccx_fund_ret_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp, sep=',')
        return data

    def get_dt_basedata_df(self, dt):
        fp = self.basedata_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

# %%
def stats_index(index_df_input, num=10, if_cumprod=False, start_date=None, end_date=None, yearly_trading_days=252):
    '''
    index_df_input: 
        dataframe: columns = ['DT', 'ret'], DT format like'20120101'
    start_date: 
        None or '20120101'
    end_date: 
        like start_date
    yearly_trading_trading: 
        the number of yearly trading days
    if_cumprod:
        False: Simple interest method, nv = cumsum(daily_ret_ts) + 1, an_ret = mean(daily_ret_ts) * yearly_trading_days, maxDD = (cumsum nv di - cumsum nv max)
        True: Cumpound interest method, nv = cumprod(daily_ret_ts + 1), an_ret = (nv[end_dt] - nv[start_dt] + 1) ^ (yearly_trading_days/actual_trading_days) - 1, maxDD = (cumprod nv di - cumprod nv max)/cumprod nv max
    num:
        top num maxDD cases
    '''
    print('if_cumprod:{}'.format(if_cumprod))
    index_df = index_df_input.copy(deep=True)
    index_df['DT'] = index_df['DT'].astype(str)
    if end_date is not None:
        index_df = index_df[index_df['DT']<=end_date]
    if start_date is not None:
        index_df = index_df[index_df['DT']>=start_date]
    index_df['year'] = index_df['DT'].apply(lambda x: str(x)[:4])
    index_df = index_df.dropna()
    # maxDD
    '''
    print('maxDD_stats')
    maxDD_stats = maxDD_analysis(index_df_input, num=num, if_cumprod=if_cumprod, start_date=start_date, end_date=end_date)
    print('ret_stats')
    ret_dis = ret_distribution_stats(index_df_input, if_cumprod=if_cumprod, rolling_window=252) 
    print('vol_stats')   
    vol_stats = vol_distribution_stats(index_df_input, if_cumprod=if_cumprod, rolling_window=252)
    print('maxDD_dis_stats')
    maxDD_dis_stats = maxDD_distribution_stats(index_df_input, if_cumprod=if_cumprod, rolling_window=252)
    print('ret_rank_by_year')
    ret_rank_by_year = stats_mfund_rank_by_year(index_df, start_date=None, end_date=None, mfund=['股票型'])
    print('tvr_stats')
    tvr_stats = stats_index_tvr(weight_df, start_date=None, end_date=None)
    print('ret_rank_by_window')
    rank_by_window = stats_fund_rank_by_window(index_df, start_date=None, end_date=None, rolling_window=252, mfund=['股票型'])
    '''
    stats_all = ret_stats(index_df, if_cumprod=if_cumprod, yearly_trading_days=yearly_trading_days) 
    print('Ret stats')
    print(stats_all)
    
def ret_stats(index_df_input, if_maxdd=True, if_all=False, if_cumprod=True, start_date=None, end_date=None, monthly_trading_days=25):
    #print('if_cumprod:{}'.format(if_cumprod))
    index_df = index_df_input.copy(deep=True)
    index_df['DT'] = index_df['DT'].astype(str)
    if end_date is not None:
        index_df = index_df[index_df['DT']<=end_date]
    if start_date is not None:
        index_df = index_df[index_df['DT']>=start_date]
    index_df['date'] = index_df['DT'].apply(lambda x: str(x)[:4]+'-'+str(x)[4:6])
    index_df = index_df.dropna()
    year_list = list(set(index_df['date'].tolist()))
    year_list.sort()
    year_list.append('all')  #['an_ret(%)','an_std(%)','an_std_semi(%)','sharpe','sharpe_semi','maxDD(%)', 'maxDD_start','maxDD_end'])
    year_list = np.unique(year_list)
    if if_all:
        year_list = ['all']
    if if_maxdd:    
        stats_all = pd.DataFrame(index=year_list, columns=['month_ret(%)','month_std(%)','sharpe','maxDD(%)', 'maxDD_start','maxDD_end'])
    else:
        stats_all = pd.DataFrame(index=year_list, columns=['month_ret(%)','month_std(%)','sharpe'])
    for year in year_list:
        if year != 'all':
            dt_end = index_df['DT'][index_df['date']==year].tail(1).iloc[0]
            dt_end_index = index_df['DT'].tolist().index(dt_end)
            if dt_end_index < monthly_trading_days:
                dt_list = index_df['DT'].tolist()[:dt_end_index]
            else:
                dt_list = index_df['DT'][dt_end_index-monthly_trading_days:dt_end_index].tolist()
            dt_len = len(dt_list)
            tmp = index_df.loc[index_df['DT'].isin(dt_list),'ret'].values
            c = pd.Series(tmp)
            value = (c + 1).cumprod()
            stats_all.loc[year, 'month_ret(%)'] = ((value.iloc[-1]) **(monthly_trading_days/dt_len)-1)* 100
            stats_all.loc[year, 'month_std(%)'] = np.nanstd(tmp) * np.sqrt(monthly_trading_days) * 100
            cum_ts = [[k,v] for k,v in enumerate(value)]
            stats_all.loc[year, 'sharpe'] = stats_all.loc[year, 'month_ret(%)'] / stats_all.loc[year, 'month_std(%)'] 
            if if_maxdd:    
                max_value = [max(cum_ts[:i+1], key=lambda item:item[1]) for i in range(len(cum_ts))] 
                drawdown = [[ma[0], cu[0], (ma[1] - cu[1])/ma[1] * 100] for ma, cu in zip(max_value, cum_ts)]
                si, di, maxdd = max(drawdown, key=lambda item:item[2])
                stats_all.loc[year, 'maxDD(%)'] = maxdd
                stats_all.loc[year, 'maxDD_start'] = dt_list[si]
                stats_all.loc[year, 'maxDD_end'] = dt_list[di]
        else:
            tmp = index_df['ret'].values
            c = pd.Series(tmp)
            value = (c + 1).cumprod()
            dt_list = index_df['DT'].tolist()
            dt_len = len(dt_list)
            tmp = index_df.loc[index_df['DT'].isin(dt_list),'ret'].values
            c = pd.Series(tmp)
            value = (c + 1).cumprod()
            stats_all.loc[year, 'month_ret(%)'] = ((value.iloc[-1]) **(monthly_trading_days/dt_len)-1)* 100
            stats_all.loc[year, 'month_std(%)'] = np.nanstd(tmp) * np.sqrt(monthly_trading_days) * 100
            cum_ts = [[k,v] for k,v in enumerate(value)]
            stats_all.loc[year, 'sharpe'] = stats_all.loc[year, 'month_ret(%)'] / stats_all.loc[year, 'month_std(%)'] 
            if if_maxdd:    
                max_value = [max(cum_ts[:i+1], key=lambda item:item[1]) for i in range(len(cum_ts))] 
                drawdown = [[ma[0], cu[0], (ma[1] - cu[1])/ma[1] * 100] for ma, cu in zip(max_value, cum_ts)]
                si, di, maxdd = max(drawdown, key=lambda item:item[2])
                stats_all.loc[year, 'maxDD(%)'] = maxdd
                stats_all.loc[year, 'maxDD_start'] = dt_list[si]
                stats_all.loc[year, 'maxDD_end'] = dt_list[di]
    return stats_all


def maxDD_analysis(index_df, num=10, if_cumprod=False, start_date=None, end_date=None):
    '''最大回撤发生的时间'''
    '''
    
    index_df: 
        dataframe: columns = ['DT', 'ret'], DT format like'20120101'
    start_date: 
        None or '20120101'
    end_date: 
        like start_date
    if_cumprod:
        False: Simple interest method, nv = cumsum(daily_ret_ts) + 1, an_ret = mean(daily_ret_ts) * yearly_trading_days, maxDD = (cumsum nv di - cumsum nv max)
        True: Cumpound interest method, nv = cumprod(daily_ret_ts + 1), an_ret = (nv[end_dt] - nv[start_dt] + 1) ^ (yearly_trading_days/actual_trading_days) - 1, maxDD = (cumprod nv di - cumprod nv max)/cumprod nv max
    num:
        num of largest maxDD cases
    
    Output res_df
        dataframe:
            label | maxDD_start | maxDD_end | maxDD | maxDD days | Higher days
    '''
    dt_list = index_df['DT'].tolist()
    stats_list = []
    next_dt = None
    for dt in dt_list:
        if (dt == next_dt)|(dt == dt_list[0]):
            stats = pd.DataFrame(columns=['maxDD_start','maxDD_end' , 'maxDD(%)' , 'maxDD_days' , 'Higher_days'])
            ret_df = index_df.loc[index_df['DT']>=dt,:]
            if if_cumprod:
                ret_df.loc[:,'cum_ret'] = (ret_df['ret']+1).cumprod() - 1
            else:
                ret_df.loc[:,'cum_ret'] = ret_df['ret'].cumsum()
            higher_df = ret_df.loc[ret_df['cum_ret']>=0,:]
            if len(higher_df) == 0:
                higher_dt = ret_df['DT'].iloc[-1]
                next_dt = higher_dt
            else:
                higher_dt = higher_df['DT'].iloc[0]
                if higher_dt != dt_list[-1]:
                    next_dt = dt_list[dt_list.index(higher_dt)+1]
                else:
                    next_dt = higher_dt
            dd_df = ret_df.loc[ret_df['DT']<=higher_dt,:]
            max_dd_end = dd_df.loc[dd_df['cum_ret'].idxmin(),'DT']
            max_dd = dd_df['cum_ret'].min() * 100
            dt_index = dt_list.index(dt)
            dd_end_index = dt_list.index(max_dd_end)
            higher_dt_index = dt_list.index(higher_dt)
            max_dd_days = dd_end_index - dt_index
            higher_days = higher_dt_index - dd_end_index
            stats.loc[:,'maxDD_start']  = [dt]
            stats.loc[:,'maxDD_end'] = [max_dd_end]
            stats.loc[:,'maxDD(%)'] = [max_dd]
            stats.loc[:,'maxDD_days'] = [max_dd_days]
            stats.loc[:,'Higher_days'] = [higher_days]
            stats_list.append(stats)       
        else:
            pass
    full_stats = pd.concat(stats_list)
    full_stats.sort_values('maxDD(%)',inplace=True)
    dd_stats = full_stats.iloc[:num,:]
    dd_stats['label'] = [i+1 for i in list(range(num))]
    dd_stats = dd_stats[['label','maxDD_start','maxDD_end' , 'maxDD(%)' , 'maxDD_days' , 'Higher_days']]
    print('MaxDD analysis')
    print(dd_stats)
    return dd_stats


def ret_distribution_stats(index_df_input, if_cumprod=False, rolling_window=252):
    
    '''
    收益率百分位[0, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 100]，后面
    '''
    index_df = index_df_input.copy(deep=True)
    if if_cumprod:
        index_df['rolling_ret'] = (index_df['ret']+1).rolling(rolling_window).apply(lambda x:x.cumprod().iloc[-1]-1)
    else:
        index_df['rolling_ret'] = index_df['ret'].rolling(rolling_window).sum()
    stats = pd.DataFrame(columns=['ret'])
    percentile_list = [0, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 100]
    #percentile_list = [0, 5, 10,11,12,13,14, 15, 20, 25,30,40, 50, 75, 80, 85, 90, 95, 100]
    for percentile in percentile_list:
        stats.loc[str(percentile),'ret'] = index_df['rolling_ret'].quantile(percentile/100)
    stats.reset_index(inplace=True)
    stats.columns = ['percentile','ret']
    print(stats.T)
    return stats

def vol_distribution_stats(index_df_input, if_cumprod=False, rolling_window=252):
    index_df = index_df_input.copy(deep=True)
    index_df['rolling_vol'] = index_df['ret'].rolling(rolling_window).std()*np.sqrt(252)
    stats = pd.DataFrame(columns=['vol'])
    percentile_list = [0, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 100]
    for percentile in percentile_list:
        stats.loc[str(percentile),'vol'] = index_df['rolling_vol'].quantile(percentile/100)
    stats.reset_index(inplace=True)
    stats.columns = ['percentile','vol']
    print(stats.T)
    return stats

def maxDD_distribution_stats(index_df_input, if_cumprod=False, rolling_window=252):
    index_df = index_df_input.copy(deep=True)
    def get_max_dd(return_series,if_cumprod):
        if if_cumprod:
            return_list = list((return_series+1).cumprod())
            i = np.argmax((np.maximum.accumulate(return_list) - return_list))
            if i == 0:
                drawdown_max = 0
            else:    
                j = np.argmax(return_list[:i])  # 开始位置
                drawdown_max = (return_list[j] - return_list[i])/return_list[j]
        else:
            return_list = list(return_series.cumsum())
            i = np.argmax((np.maximum.accumulate(return_list) - return_list))
            if i == 0:
                drawdown_max = 0
            else:    
                j = np.argmax(return_list[:i])  # 开始位置
                drawdown_max = return_list[j] - return_list[i]
        return drawdown_max
    index_df['rolling_max_dd'] = index_df['ret'].rolling(rolling_window).apply(lambda x:get_max_dd(x,if_cumprod))
    stats = pd.DataFrame(columns=['max_dd'])
    percentile_list = [0, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 100]
    for percentile in percentile_list:
        stats.loc[str(percentile),'max_dd'] = index_df['rolling_max_dd'].quantile(percentile/100)
    stats.reset_index(inplace=True)
    stats.columns = ['percentile','max_dd']
    print(stats.T)
    return stats

def stats_index_tvr(weight_df, start_date=None, end_date=None):
    '''
    weight_df: SECU_CODE | DT | WEIGHT
    output:
        year | tvr_shares | tvr_ratio 
        year: 2012, 2013, ..., average
    '''
    if end_date is not None:
        weight_df = weight_df[weight_df['DT']<=end_date]
    if start_date is not None:
        weight_df = weight_df[weight_df['DT']>=start_date]
    weight_df = weight_df.copy(deep=True)
    dt_ls = sorted(list(set(weight_df['DT'].tolist())))
    n = len(dt_ls)
    tvr_ls = []
    tvr_shrs = []
    for i in range(n-1):
    # 最后一期的tvr_ratio无法计算
        df1 = weight_df[weight_df['DT']==dt_ls[i]]
        df2 = weight_df[weight_df['DT']==dt_ls[i+1]][['SECU_CODE','WEIGHT']]
        df = pd.merge(df1 , df2, how='outer', on='SECU_CODE', suffixes=['_X','_Y'])
        df.fillna(value=0, inplace=True)
        df['secu_tvr'] = abs(df['WEIGHT_Y']-df['WEIGHT_X'])/2
        shr_ls1 = set(df1['SECU_CODE'].tolist())
        shr_ls2 = set(df2['SECU_CODE'].tolist())  
        tvr_ls.append(sum(df['secu_tvr']))
        tvr_shrs.append(1- len(shr_ls1 & shr_ls2)/len(shr_ls1 | shr_ls2))
        print(dt_ls[i])
    tvr_ls.append(0) ; tvr_shrs.append(0)#补上最后一期tvr为0
    dt_df = pd.DataFrame({'DT':dt_ls, 'tvr_shares':tvr_shrs, 'tvr_ratio':tvr_ls})
    dt_df['year'] = dt_df['DT'].map(lambda x: int(x)//10000)
    tvr_df = dt_df[['year','tvr_shares','tvr_ratio']].groupby('year').apply(sum)
    del tvr_df['year']
    print(tvr_df)
    return tvr_df

def stats_mfund_rank_by_year(index_df, start_date=None, end_date=None, mfund=['股票型']):
    '''
    input:
        DT | index | ret
    output:
        year | ret_rank | sharpe_rank | calmar_rank | all
    '''
    index_df = index_df.copy(deep=True)
    if end_date is not None:
        index_df = index_df[index_df['DT']<=end_date]
    if start_date is not None:
        index_df = index_df[index_df['DT']>=start_date]
    stats_df = ret_stats(index_df, if_cumprod=True, start_date=start_date, end_date=end_date )[['an_ret(%)','sharpe','maxDD(%)']].reset_index().rename(columns={'index':'year'})
    stats_df['fund_code'] = '0'
    api = DataAPI()
    year_ls = stats_df['year'].tolist()
    year_end = [x+'1231' for x in year_ls[:-2]] + [str(index_df['DT'].tolist()[-1])]#调整最后一年最后日期（最后一年数据未必到年底）和'all'
    fund_path = '/work/panyuqiong/data/fund_cooked/{}/{}/{}/fund_data.txt'
    s = time.time()
    for year in year_end:
        dt_ls = api.get_tradingday_between(start_date = year[0:4] + '0101', end_date = year)
        dt_start = dt_ls[0] ; dt_end = dt_ls[-1]
        df_start = pd.read_csv(fund_path.format(int(dt_start[0:4]), int(dt_start[4:6]), int(dt_start[6:8])), sep='|')
        df_end = pd.read_csv(fund_path.format(int(dt_end[0:4]), int(dt_end[4:6]), int(dt_end[6:8])), sep='|')
        df_start['if_type'] = df_start['FUNDTYPE'].map(lambda x: x in mfund)
        df_end['if_type'] = df_end['FUNDTYPE'].map(lambda x: x in mfund)
        fund_ls_start = set(df_start[(df_start['if_type'] == True) & (df_start['IS_MAIN'] == True)]['fund_code'].tolist())
        fund_ls_end = set(df_end[(df_end['if_type'] == True) & (df_end['IS_MAIN'] == True)]['fund_code'].tolist())
        fund_ls = sorted(list(fund_ls_start & fund_ls_end))
        if year == year_end[0]:
            fund_df = df_start.loc[df_start['fund_code'].isin(fund_ls)][['fund_code','tradingday','fund_ret']]
        for date in dt_ls[1:]:
            new_df = pd.read_csv(fund_path.format(int(date[0:4]), int(date[4:6]), int(date[6:8])), sep='|')   
            new_df = new_df.loc[new_df['fund_code'].isin(fund_ls)][['fund_code','tradingday','fund_ret']]
            fund_df =  pd.concat([fund_df, new_df]) 
            print(date)
    print(time.time() - s)
    fund_ls = list(set(fund_df['fund_code'].tolist()))
    for fund in fund_ls:
        single_fund_df = fund_df[fund_df['fund_code'] == fund][['tradingday','fund_ret']].rename(columns = {'tradingday':'DT','fund_ret':'ret'})
        stats_df2 = ret_stats(single_fund_df, if_cumprod=True).reset_index().rename(columns={'index':'year'})[['year','an_ret(%)','sharpe','maxDD(%)']]
        stats_df2['fund_code'] = fund
        stats_df = pd.concat([stats_df, stats_df2])
    stats_df['calmar'] = stats_df['an_ret(%)']/stats_df['maxDD(%)']
    rank_df = pd.DataFrame(index = year_ls, columns = ['ret_rank', 'sharpe_rank', 'calmar_rank' , 'all'])
    for year in year_ls:
        year_df = stats_df[stats_df['year'] == year]
        rank_df.loc[year ,'ret_rank'] = year_df['an_ret(%)'].rank(ascending=False).tolist()[0]/len(year_df)*100
        rank_df.loc[year ,'sharpe_rank'] = year_df['sharpe'].rank(ascending=False).tolist()[0]/len(year_df)*100
        rank_df.loc[year ,'calmar_rank'] = year_df['calmar'].rank(ascending=False).tolist()[0]/len(year_df)*100
        rank_df.loc[year ,'all'] = len(year_df)  
    print('Rank by year')
    print(rank_df)       
    return rank_df

def stats_fund_rank_by_window(index_df, start_date=None, end_date=None, rolling_window=252, mfund=['股票型']):
    '''
    input:
        DT | index | ret
    output:
        percentile | ret_rank | sharpe_rank | calmar_rank     
    从运行结果来看，每一次simulation需要10-20s；如果十年，测试2250次，大概9h
    '''
    index_df = index_df.copy(deep=True)
    api = DataAPI()
    if end_date is not None:
        index_df = index_df[index_df['DT']<=end_date]
    if start_date is not None:
        index_df = index_df[index_df['DT']>=start_date]
    end_date = index_df['DT'].tolist()[-1]
    start_date = index_df['DT'].tolist()[0]
    dt_ls = api.get_tradingday_between(start_date, end_date)
    s = time.time()
    fund_path = '/work/panyuqiong/data/fund_cooked/{}/{}/{}/fund_data.txt'
    fund_union = []
    fund_dict = {} # dt:fund_ls 用于储存所有用到的fund_code,免去未来重新读取
    for i in range(rolling_window - 1, len(dt_ls)):
        print('calculate fund list dt:{}'.format(dt_ls[i]))
        dt_start = dt_ls[i - rolling_window + 1]
        dt_end = dt_ls[i]         
        df_start = pd.read_csv(fund_path.format(int(dt_start[0:4]), int(dt_start[4:6]), int(dt_start[6:8])), sep='|')
        df_end = pd.read_csv(fund_path.format(int(dt_end[0:4]), int(dt_end[4:6]), int(dt_end[6:8])), sep='|')
        df_start['if_type'] = df_start['FUNDTYPE'].map(lambda x: x in mfund)
        df_end['if_type'] = df_end['FUNDTYPE'].map(lambda x: x in mfund)
        fund_ls_start = set(df_start[(df_start['if_type'] == True) & (df_start['IS_MAIN'] == True)]['fund_code'].tolist())
        fund_ls_end = set(df_end[(df_end['if_type'] == True) & (df_end['IS_MAIN'] == True)]['fund_code'].tolist())
        fund_ls = sorted(list(fund_ls_start & fund_ls_end))
        fund_dict[dt_end] = fund_ls
        fund_union = list(set(fund_ls) | set(fund_union))
    print('calculate fund list finished! time:{}'.format(time.time() - s))
    #读取所有有用的数据
    print('start loading data')
    fund_df = pd.DataFrame(columns=['fund_code','tradingday','fund_ret'])
    for i in range(len(dt_ls)):
        date = dt_ls[i]
        new_df = pd.read_csv(fund_path.format(int(date[0:4]), int(date[4:6]), int(date[6:8])), sep='|')   
        new_df = new_df.loc[new_df['fund_code'].isin(fund_union)][['fund_code','tradingday','fund_ret']]
        fund_df =  pd.concat([fund_df, new_df]) 
        print('loading data {}'.format(date))
    print('finished loading data dt:{}'.format(time.time() - s))
    ret_rank_ls = []; sharpe_rank_ls = []; calmar_rank_ls = []; all_ls = []
    for i in range(rolling_window - 1, len(dt_ls)):
        fund_ls = fund_dict[dt_ls[i]]
        start_dt = dt_ls[i-rolling_window+1] ; end_dt = dt_ls[i]
        #stats_df = ret_stats(index_df, if_cumprod=True, start_date=start_dt, end_date=end_dt )[['an_ret(%)','sharpe','maxDD(%)']].reset_index().rename(columns={'index':'year'})
        stats_df = ret_stats(index_df, if_maxdd=False, if_all=True, if_cumprod=True, start_date=start_dt, end_date=end_dt )[['an_ret(%)','sharpe']].reset_index().rename(columns={'index':'year'})
        stats_df['fund_code'] = '0'
        stats_df = stats_df[stats_df['year']== 'all']
        for fund in fund_ls:
            single_fund_df = fund_df[fund_df['fund_code'] == fund][['tradingday','fund_ret']].rename(columns = {'tradingday':'DT','fund_ret':'ret'})
            #stats_df2 = ret_stats(single_fund_df, if_cumprod=True, start_date=start_dt, end_date=end_dt).reset_index().rename(columns={'index':'year'})[['year','an_ret(%)','sharpe','maxDD(%)']]
            stats_df2 = ret_stats(single_fund_df, if_maxdd=False, if_all=True, if_cumprod=True, start_date=start_dt, end_date=end_dt).reset_index().rename(columns={'index':'year'})[['year','an_ret(%)','sharpe']]
            stats_df2['fund_code'] = fund
            stats_df = pd.concat([stats_df, stats_df2[stats_df2['year'] == 'all']])
        #stats_df['calmar'] = stats_df['an_ret(%)']/stats_df['maxDD(%)']
        ret_rank_ls.append(stats_df['an_ret(%)'].rank(ascending=False).tolist()[0]/len(stats_df)*100)
        sharpe_rank_ls.append(stats_df['sharpe'].rank(ascending=False).tolist()[0]/len(stats_df)*100)
        #calmar_rank_ls.append(stats_df['calmar'].rank(ascending=False).tolist()[0]/len(stats_df)*100)
        all_ls.append(len(stats_df))  
        print('Rank by year: dt {} time:{}'.format(dt_ls[i], time.time()-s)) 
    stats = pd.DataFrame(columns=['ret_rank','sharpe_rank'])#,'calmar_rank'
    percentile_list = [0, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 100]
    for percentile in percentile_list:
        stats.loc[str(percentile),'ret_rank'] = np.percentile(ret_rank_ls, percentile)
        stats.loc[str(percentile),'sharpe_rank'] = np.percentile(sharpe_rank_ls, percentile)
        #stats.loc[str(percentile),'calmar_rank'] = np.percentile(calmar_rank_ls, percentile)
    stats.reset_index(inplace=True)
    stats.columns = ['percentile','ret_rank','sharpe_rank']# ,'calmar_rank'
    print(stats.T)
    print(time.time()-s)    
    return stats
    pass

# %%
if __name__ == '__main__':
    # read index_df = DT, ret, DT like 20120101
    import pandas as pd
    import numpy as np
    import time
    path_dir  = '/data/public_transfer/zhuqifan/stock_index_2022/MF5_M_xczx_crsc_results/index_level/index_level_20130101__20211230.csv'
    #path_dir = '/data/public_transfer/zhuqifan/stock_index_2022/MF5_M_xczx_crsc_results/index_weight/index_weight_20130101__20211230.csv'
    df = pd.read_csv(path_dir, sep=',', na_values=['None']) 
    #df.rename(columns = {'date':'DT', 'secu_code':'SECU_CODE', 'weight':'WEIGHT'}, inplace=True)
    df.columns = ['DT', 'index', 'ret']
    df['DT'] = df['DT'].map(lambda x: x[0:4]+ x[5:7] + x[8:10])
    
    # set parameter
    if_cumprod = True
    start_date = None
    end_date = None
    days = 250
    maxDD_num_cases = 10
    #stats_index_tvr = stats_index_tvr(df, '20200101','20211231') 
    #stats_fund_rank_by_window(df, start_date = '20201201', end_date='20211231')
    stats_mfund_rank_by_year(df, start_date ='20200101', end_date='20211120')
    #stats = maxDD_distribution_stats(df,if_cumprod=True)
    #stats = stats_index(df, num=maxDD_num_cases, if_cumprod=if_cumprod, start_date=start_date, end_date=end_date, yearly_trading_days=days)

# %%
