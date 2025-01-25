import pandas as pd 
import numpy as np
from tqdm import tqdm
import os
import constant as cons

class per(object): 
    def __init__(self, start_date_m, start_date_q, start_date_hy, end_date, benchmark, output_dir = '/data/public_transfer/liyihan/esg/{}/{}月报'):
        self.exg_trading_day_path = '/data/raw/WIND/ASHARECALENDAR.txt'
        self.ib_trading_day_path = '/data/raw/JY/IndexProduct/QT_TRADINGDAYNEW.txt'
        self.out_dir = output_dir 
        self.path_template = '/data-platform/ccxd/prod/stock_index/Thematic_index/{}_xinjingbao/index_point/{}.csv'
        #self.path_weight = '/data-platform/ccxd/prod/stock_index/Thematic_index/{}_xinjingbao/weight/{}.csv'
        self.start_date = [start_date_m, start_date_q, start_date_hy]
        self.end_date = end_date
        self.benchmark = benchmark
        self.if_cumprod = True
        self.index_path_dict = cons.index_path_dict
        self.dt_list = self.get_tradingday_between(start_date_hy, self.end_date)

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

    def get_table_from_home(self, table_path, fields=None, na_values=['None'], sep='|', dtype={}):
        df = pd.read_csv(table_path, sep=sep, usecols=fields, na_values=na_values,
                         dtype=dtype, low_memory=False, error_bad_lines=False)
        return df

    def _init_dirs(self, end_date, index_name): 
        if not os.path.exists(self.out_dir.format(end_date, index_name)): 
            os.makedirs(self.out_dir.format(end_date, index_name))
        pass

    def get_benchmark_return_df(self):
        ret_df = pd.DataFrame()
        for dt in tqdm(self.dt_list, desc='加载基准数据中'):
            fp_bench = self.index_path_dict[self.benchmark].format(dt)
            df_bench = self.get_table_from_home(fp_bench)
            ret_df = pd.concat([ret_df, df_bench])
        ret_df.reset_index(drop=True,inplace=True)
        return ret_df

    def get_daily_return(self,index_name):
        daily_return = pd.DataFrame()
        for dt in tqdm(self.dt_list, desc='读取原始数据中'):
            fp = self.path_template.format(index_name, dt)
            df = self.get_table_from_home(fp)
            daily_return = pd.concat([daily_return, df])
        return daily_return

    def combine_stats_table(self):
        index_name_list = [f'{self.benchmark}','Carbon','ESG']
        i = 0
        stats_df = pd.DataFrame()
        dt_index = ['近一月','近三月','近六月']
        index_stats = pd.DataFrame(index=index_name_list,columns=['date','index_name','区间收益率(%)','年化波动率(%)','夏普比例','最大回撤(%)'])
        for start_date in self.start_date:
            dt_list = self.get_tradingday_between(start_date, end_date)
            for index_name in index_name_list:
                if index_name == f'{self.benchmark}':
                    hy_return =self.get_benchmark_return_df()
                else:
                    hy_return = self.get_daily_return(index_name)
                hy_return['date'] = hy_return['date'].astype(str)
                dt_return = hy_return[hy_return['date'].isin(dt_list)]
                value = dt_return['daily_point'].tolist()
                cum_ts = [[k,v] for k,v in enumerate(value)] #k:索引，v:累积收益率
                index_stats.loc[index_name, 'index_name'] = index_name
                index_stats.loc[index_name, '区间收益率(%)'] = (value[-1]-value[0])/value[0]*100
                index_stats.loc[index_name, '年化波动率(%)'] = dt_return['daily_ret'].std()*np.sqrt(252)*100
                index_stats.loc[index_name, '夏普比例'] = index_stats.loc[index_name, '区间收益率(%)']/index_stats.loc[index_name, '年化波动率(%)']
                max_value = [max(cum_ts[:i+1], key=lambda item:item[1]) for i in range(len(cum_ts))] 
                drawdown = [[ma[0], cu[0], (ma[1] - cu[1])/ma[1] * 100] for ma, cu in zip(max_value, cum_ts)]
                index_stats.loc[index_name, '最大回撤(%)'] = max(drawdown, key=lambda item:item[2])[2]
            index_stats['date'] = dt_index[i]
            i += 1
            stats_df = pd.concat([stats_df,index_stats])
        stats_df.set_index('date',inplace=True)
        stats_df.to_csv(self.out_dir.format(self.end_date) + '/指数半年内统计指标.csv')
        return stats_df


if __name__ == '__main__':
    start_date_m = '20221125'
    start_date_q = '20220925'
    start_date_hy = '20220625'
    end_date = '20221225'
    benchmark = 'CCX1800'
    output_dir = '/data/public_transfer/liyihan/esg/{}'
    test = per(start_date_m, start_date_q, start_date_hy, end_date, benchmark, output_dir)
    test.combine_stats_table() 

