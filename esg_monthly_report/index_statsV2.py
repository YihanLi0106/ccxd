from data_api import DATAAPI
import stats_annually_clean_v6 as stn
import pandas as pd 
import numpy as np
from tqdm import tqdm
import os
from importlib import reload
import constant as cons
reload(stn)
reload(cons)


class IndexStats:
    def __init__(self, target_index_name_list, benchmark_index_name_list, outdir, start_date=None, end_date=None, 
        if_cumprod=True, fields=None, rank_type_dict=None):
        self.outdir = outdir
        self.init_param()
        self.data_api = DATAAPI()
        self.target_index_name_list = target_index_name_list
        self.benchmark_index_name_list = benchmark_index_name_list
        self.start_date = start_date
        self.end_date = end_date
        self.if_cumprod = if_cumprod
        self.fields = fields
        self.rank_type_dict = rank_type_dict   
        self.total_field = ['turnover_ratio',]
        self.stats_chinese_name_dict = {
                'ret(%)':'年化收益率(%)',
                'std(%)': '年化波动率(%)', 
                'sharpe':'夏普比率',
                'maxDD(%)':'最大回撤',
                'turnover_ratio':'换手率',
                'calmar_rank':'calmar排名',
                'ret_rank':'收益率排名', 
                'sharpe_rank':'夏普比率排名'
            }
        self.index_path_dict = cons.index_path_dict
        self.weight_path_dict = cons.weight_path_dict

    def init_param(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

    def get_benchmark_index_ret_dict(self):
        dt_list = self.data_api.get_tradingday_between(self.start_date, self.end_date)
        bench_dict = {}
        for benchmark in self.benchmark_index_name_list:
            i = 0
            for dt in tqdm(dt_list, desc='加载基准数据中'):
                fp_bench = self.index_path_dict[benchmark].format(dt)
                _ = self.data_api.get_table_from_home(fp_bench)
                point_df = _.loc[:, ['date','daily_point']]
                ret_df = _.loc[:, ['date','daily_ret']]
                if i == 0:
                    bench_dict[f'{benchmark}_index'] = [point_df]
                    bench_dict[f'{benchmark}_ret'] = [ret_df]
                    i += 1 
                else:
                    bench_dict[f'{benchmark}_index'].append(point_df)
                    bench_dict[f'{benchmark}_ret'].append(ret_df)
            bench_dict[f'{benchmark}_index'] = pd.concat(bench_dict[f'{benchmark}_index'])
            bench_dict[f'{benchmark}_index'].columns = ['DT',f'{benchmark}']
            bench_dict[f'{benchmark}_ret'] = pd.concat(bench_dict[f'{benchmark}_ret'])
            bench_dict[f'{benchmark}_ret'].columns = ['DT',f'{benchmark}']
        return bench_dict
    
    def get_target_index_ret_df(self, factor):
        factor_index_list, factor_ret_list = [], []
        dt_list = self.data_api.get_tradingday_between(self.start_date, self.end_date)
        for dt in tqdm(dt_list, desc='读取因子数据中'):
            fp = self.index_path_dict[factor].format(dt)
            _ = self.data_api.get_table_from_home(fp)
            point_df = _[['date','daily_point']]
            ret_df = _[['date','daily_ret']]
            factor_index_list.append(point_df)
            factor_ret_list.append(ret_df)
        factor_index_df = pd.concat(factor_index_list)
        factor_index_df.columns = ['DT', f'{factor}']
        factor_ret_df = pd.concat(factor_ret_list)
        factor_ret_df.columns = ['DT', f'{factor}']
        return factor_index_df, factor_ret_df

    def get_index_ret_dict_from_daily_table(self):
        factor_return_dict = {}
        bm_dict = self.get_benchmark_index_ret_dict()
        for factor in self.target_index_name_list:
            index_df, ret_df = self.get_target_index_ret_df(factor)
            for benchmark in self.benchmark_index_name_list:
                bm_index = bm_dict[f'{benchmark}_index']
                bm_ret = bm_dict[f'{benchmark}_ret']
                index_df = pd.merge(index_df, bm_index, how='left', on='DT')
                ret_df = pd.merge(ret_df, bm_ret, how='left', on='DT')
            factor_return_dict[f'{factor}_index'] = index_df
            factor_return_dict[f'{factor}_ret'] = ret_df
        return factor_return_dict

    def get_index_ret_dict_from_table(self):
        """需满足传入的总表第一列为日期(格式为'20121231')，第二列为点位，第三列为收益率。

        Returns:
            dict: factor_ret, factor_index (for factor in target_index)
        """
        bm_dict = {}
        factor_return_dict = {}
        for benchmark in self.benchmark_index_name_list:
            temp_df = self.data_api.get_table_from_home(self.index_path_dict[benchmark].format('index_point'))
            bm_dict[benchmark] = temp_df.loc[(temp_df.iloc[:,0].astype(str) >= self.start_date) & (temp_df.iloc[:,0].astype(str) <= self.end_date)].copy()
        for factor in tqdm(self.target_index_name_list):
            df1 = self.data_api.get_table_from_home(self.index_path_dict[factor].format('index_point'))
            df = df1.loc[(df1.iloc[:,0].astype(str) >= self.start_date) & (df1.iloc[:,0].astype(str) <= self.end_date)].copy()
            index_df = df.iloc[:,[0,1]]
            index_df.columns = ['DT', factor]
            ret_df = df.iloc[:,[0,2]]
            ret_df.columns = ['DT', factor]
            for benchmark in self.benchmark_index_name_list:
                bm_index = bm_dict[benchmark].iloc[:,[0,1]]
                bm_index.columns = ['DT', benchmark]
                bm_ret = bm_dict[benchmark].iloc[:,[0,2]]
                bm_ret.columns = ['DT', benchmark]
                index_df = pd.merge(index_df, bm_index, how='left', on='DT')
                ret_df = pd.merge(ret_df, bm_ret, how='left', on='DT')
            factor_return_dict[f'{factor}_index'] = index_df
            factor_return_dict[f'{factor}_ret'] = ret_df
        return factor_return_dict

    @staticmethod
    def get_rolling_ret(ret_df, rolling_window=250):
        rolling_ret = ret_df.set_index('DT').astype(float).rolling(rolling_window).apply(lambda x:np.prod(x + 1)).shift(- rolling_window + 1)
        rolling_ret = rolling_ret.reset_index()
        return rolling_ret
    
    def get_weight_df(self, weight_fp):
        dt_list = self.data_api.get_tradingday_between(self.start_date, self.end_date)
        weight_df_list = []
        for dt in tqdm(dt_list, desc='读取调仓数据中'):
            try:
                weight_df_list.append(self.data_api.get_table_from_home(weight_fp.format(dt)))
            except FileNotFoundError:
                continue
        weight_df = pd.concat(weight_df_list).reset_index(drop=True)[['date', 'secu_code', 'weight']]
        weight_df.columns = ['调仓日', '指数代码', '权重']
        return weight_df
        
    def get_fields_df(self, factor , total_factor_dict, index):
        ret_df = total_factor_dict[f'{factor}_ret'].set_index('DT')
        n_col_level2 = len(self.benchmark_index_name_list) + 1 
        # fields_dict = dict(zip([f'{field}_df' for field in self.fields],[pd.DataFrame(index=index)]*len(self.fields)))
        tvr_df = pd.DataFrame(index=index)
        ret_rank_df = pd.DataFrame(index=index)
        calmar_rank_df = pd.DataFrame(index=index)
        sharpe_rank_df = pd.DataFrame(index=index)
        if  'turnover_ratio' in self.fields:
            for index_name in ret_df.columns:
                weight_fp = self.weight_path_dict[index_name]
                weight_df = self.get_weight_df(weight_fp)
                weight_df.columns = ['DT','SECU_CODE','WEIGHT']
                turnover_df = stn.stats_index_tvr(weight_df)
                tvr_df[index_name] = turnover_df['tvr_ratio'].to_list() + [turnover_df['tvr_ratio'].mean()]
            tvr_df.columns = [[self.stats_chinese_name_dict['turnover_ratio']]*n_col_level2, tvr_df.columns]
        if self.rank_type_dict[factor]:
            # print(f'计算{factor}及其基准指数排名！')
            for index_name in ret_df.columns:
                input_df = ret_df[index_name].reset_index()
                input_df.columns = ['DT','ret']
                rank_df = stn.stats_mfund_rank_by_year(input_df,mfund=self.rank_type_dict[factor])
                ret_rank_df[index_name] = rank_df['ret_rank'].to_list()
                calmar_rank_df[index_name] = rank_df['calmar_rank'].to_list()
                sharpe_rank_df[index_name] = rank_df['sharpe_rank'].to_list()
            ret_rank_df.columns = [[self.stats_chinese_name_dict['ret_rank']]*n_col_level2, ret_rank_df.columns]
            calmar_rank_df.columns = [[self.stats_chinese_name_dict['calmar_rank']]*n_col_level2, calmar_rank_df.columns]
            sharpe_rank_df.columns = [[self.stats_chinese_name_dict['sharpe_rank']]*n_col_level2, sharpe_rank_df.columns]
        fields_df = pd.concat([tvr_df, ret_rank_df, calmar_rank_df, sharpe_rank_df],axis=1)
        return fields_df 

    def combine_stats_table(self, factor, factor_return_dict):
        for i in tqdm(range(len(self.benchmark_index_name_list) + 1), desc='生成基础统计结果'):
            input_df = factor_return_dict[f'{factor}_ret'].set_index('DT').iloc[:, i]
            input_df = input_df.reset_index()
            name = input_df.columns[-1]
            input_df.columns = ['DT', 'ret']
            stats_df = stn.ret_stats(input_df, self.if_cumprod)
            index = stats_df.index
            if i == 0:
                an_ret_df = pd.DataFrame(index=index)
                an_std_df = pd.DataFrame(index=index)
                sharpe_df = pd.DataFrame(index=index)
                maxDD_df = pd.DataFrame(index=index)
                maxDD_start_df = pd.DataFrame(index=index)
                maxDD_end_df = pd.DataFrame(index=index)
                an_ret_df[f'{name}'] = stats_df['an_ret(%)'].copy()
                an_std_df[f'{name}'] = stats_df['an_std(%)'].copy()
                sharpe_df[f'{name}'] = stats_df['sharpe'].copy()
                maxDD_df[f'{name}'] = stats_df['maxDD(%)'].copy()
                maxDD_start_df[f'{name}'] = stats_df['maxDD_start'].copy()
                maxDD_end_df[f'{name}'] = stats_df['maxDD_end'].copy()
            else:
                an_ret_df[f'{name}'] = stats_df['an_ret(%)'].copy()
                an_std_df[f'{name}'] = stats_df['an_std(%)'].copy()
                sharpe_df[f'{name}'] = stats_df['sharpe'].copy()
                maxDD_df[f'{name}'] = stats_df['maxDD(%)'].copy()
                maxDD_start_df[f'{name}'] = stats_df['maxDD_start'].copy()
                maxDD_end_df[f'{name}'] = stats_df['maxDD_end'].copy()
        an_ret_df.columns = [['年化收益率(%)'] * (i + 1), an_ret_df.columns]
        an_std_df.columns = [['年化波动率(%)'] * (i + 1), an_std_df.columns]
        sharpe_df.columns = [['夏普比率(%)'] * (i + 1), sharpe_df.columns]
        maxDD_df.columns = [['最大回撤(%)'] * (i + 1), maxDD_df.columns]
        maxDD_start_df.columns = [['最大回撤起始时间'] * (i + 1), maxDD_start_df.columns]
        maxDD_end_df.columns = [['最大回撤结束时间'] * (i + 1), maxDD_end_df.columns]
        combine_table_df = pd.concat([an_ret_df, an_std_df, sharpe_df, maxDD_df, maxDD_start_df, maxDD_end_df], axis=1)
        fields_df = self.get_fields_df(factor, factor_return_dict, index)
        combine_table_df = pd.concat([combine_table_df, fields_df],axis=1)
        combine_table_df.rename(index={'all':'Avg'},inplace=True)
        return combine_table_df

    def table_to_excel(self, total_factor_dict):
        for factor in self.target_index_name_list:
            ret_df = total_factor_dict[f'{factor}_ret']
            rollingret_df = IndexStats.get_rolling_ret(ret_df, rolling_window=250)
            table1 = pd.merge(total_factor_dict[f'{factor}_index'], rollingret_df, suffixes=['', '_rolling_ret'], on='DT')
            table2 = self.combine_stats_table(factor, total_factor_dict,)
            weight_fp = self.weight_path_dict[factor]
            table3 = self.get_weight_df(weight_fp)
            writer = pd.ExcelWriter(self.outdir + f'{factor}_output.xlsx')
            table1.to_excel(writer, sheet_name=f'{factor}指数点位', index=False)
            table2.to_excel(writer, sheet_name='风险收益情况')
            table3.to_excel(writer, sheet_name='指数成分权重', index=False)
            writer.save()
        return

if __name__ == '__main__':
    start_date = '20220101'
    end_date = '20220831'
    outdir_temp = '/data/public_transfer/zhuwanjia/ESG/stats/'
    if_cumprod = True
    rank_type = None
    # 是否进行排序
    # rank_type=['股票型']
    rank_type_list = [[], [], [], [], ['股票型']]
    target_index_name_list = ['Growth', 'Earning', 'LowBeta', 'ResVol', 'Illiq'] # ,'Momentum', 'Dividend', 'Value']
    benchmark_index_name_list = ['CCX300', 'CCX500', 'CCX1800']
    fields = ['turnover_ratio']
    rank_type_dict = dict(zip(target_index_name_list,rank_type_list))
    st = IndexStats(target_index_name_list, benchmark_index_name_list,
                    outdir_temp, start_date, end_date, if_cumprod=if_cumprod, 
                    fields = fields, rank_type_dict=rank_type_dict)
    total_factor_dict = st.get_index_ret_dict_from_daily_table() # 每日读取慢 如果没有总表用这个
    # total_factor_dict = st.get_index_ret_dict_from_table() # 有总表用这个
    st.table_to_excel(total_factor_dict)
