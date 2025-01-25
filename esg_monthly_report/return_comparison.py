import pandas as pd 
from tqdm import tqdm
import os
from tqdm import tqdm
import constant as cons

class per(object): 
    def __init__(self, start_date, end_date, benchmark, output_dir = '/data/public_transfer/liyihan/esg/{}/{}月报'):
        self.exg_trading_day_path = '/data/raw/WIND/ASHARECALENDAR.txt'
        self.ib_trading_day_path = '/data/raw/JY/IndexProduct/QT_TRADINGDAYNEW.txt'
        self.out_dir = output_dir 
        self.path_template = '/data-platform/ccxd/prod/stock_index/Thematic_index/{}_xinjingbao/index_point/{}.csv'
        self.start_date = start_date
        self.end_date = end_date
        self.benchmark = benchmark
        self.index_path_dict = cons.index_path_dict
        self.dt_list = self.get_tradingday_between(self.start_date, self.end_date)
        #self.path_weight = '/data-platform/ccxd/prod/stock_index/Thematic_index/{}_xinjingbao/weight/{}.csv'.format(self.end_date)

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
        point_df = pd.DataFrame()
        ret_df = pd.DataFrame(columns = ['date','daily_ret'])
        for dt in tqdm(self.dt_list, desc='加载基准数据中'):
            fp_bench = self.index_path_dict[self.benchmark].format(dt)
            _ = self.get_table_from_home(fp_bench)
            point_daily_df = _.loc[:, ['date','daily_point']]
            ret_daily_df = _.loc[:, ['date','daily_ret']]
            point_df = pd.concat([point_df, point_daily_df])
            ret_df = pd.concat([ret_df, ret_daily_df])
        start_point = point_df.iloc[0,:]['daily_point']
        point_df['daily_point'] = point_df['daily_point']/start_point*1000
        point_df.rename(columns={'daily_point':f'{self.benchmark}'},inplace=True)
        ret_df.rename(columns={'daily_ret':f'{self.benchmark}'},inplace=True)
        return point_df, ret_df

    def get_daily_return(self,index_name):
        self.daily_return = pd.DataFrame(columns = ['date','daily_point','daily_ret'])
        for dt in tqdm(self.dt_list, desc='读取原始数据中'):
            fp = self.path_template.format(index_name, dt)
            df = self.get_table_from_home(fp)
            self.daily_return = pd.concat([self.daily_return, df])
        start_point = self.daily_return.iloc[0,:]['daily_point']
        self.daily_return['daily_point'] = self.daily_return['daily_point']/start_point*1000
        pass

    def run(self):
        index_name_list = ['Carbon','ESG']
        for index_name in index_name_list:
            #df_stat = self.combine_stats_table()
            if index_name == 'Carbon':
                self.get_daily_return(index_name)
                ben_point, ben_ret =self.get_benchmark_return_df()
                ret_compare = pd.merge(self.daily_return[['date','daily_ret']],ben_ret,on='date',how='left')
                #ret_prod_compare = ret_compare.copy(deep=True)
                #ret_prod_compare.iloc[:,1] = (ret_compare.iloc[:,1]+1).cumprod(axis=0)
                #ret_prod_compare.iloc[:,2] = (ret_compare.iloc[:,2]+1).cumprod(axis=0)
                point_compare = pd.merge(self.daily_return[['date','daily_point']],ben_point,on='date')
                point_compare.rename(columns={'daily_point':'碳中和指数'},inplace=True)
                ret_compare.rename(columns={'daily_ret':'碳中和指数'},inplace=True)
                #ret_prod_compare.rename(columns={'daily_point':'美丽中国ESG指数'},inplace=True)
            else:
                self.get_daily_return(index_name)
                point_compare = pd.merge(self.daily_return[['date','daily_point']],point_compare,on='date')
                ret_compare = pd.merge(self.daily_return[['date','daily_ret']],ret_compare,on='date')
                point_compare.rename(columns={'daily_point':'美丽中国ESG指数'},inplace=True)
                ret_compare.rename(columns={'daily_ret':'美丽中国ESG指数'},inplace=True)
                #ret_prod_compare.rename(columns={'daily_point':'碳中和指数'},inplace=True)
        self._init_dirs(self.end_date, index_name)               
        ret_compare['date'] = ret_compare['date'].apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8])
        point_compare['date'] = point_compare['date'].apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8])
        ret_compare.to_csv(self.out_dir.format(self.end_date) + '/{}--{}_ret_compare.csv'.format(self.start_date, self.end_date),index=False)
        point_compare.to_csv(self.out_dir.format(self.end_date) + '/{}--{}_point_compare.csv'.format(self.start_date, self.end_date),index=False)            
        #ret_prod_compare.to_csv(self.out_dir.format(self.end_date, index_name) + '/{}--{}_ret_prod_compare.csv'.format(self.start_date, self.end_date), sep='|', na_rep='None',index=False)
        #df_stat.to_csv(self.out_dir.format(self.end_date) + '/指数半年内统计指标.csv', sep='|', na_rep='None')
        pass


if __name__ == '__main__':
    start_date = '20221125'
    end_date = '20221225'
    benchmark = 'CCX1800'
    output_dir = '/data/public_transfer/liyihan/esg/{}'
    test = per(start_date, end_date, benchmark, output_dir)
    test.run() 

