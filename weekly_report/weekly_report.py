import pandas as pd
import numpy as np
import numpy.ma as ma
import os
from tqdm import tqdm
import warnings

class DataAPI(object):
    def __init__(self):
        self.exg_trading_day_path = '/data/raw/WIND/ASHARECALENDAR.txt'
        self.ib_trading_day_path = '/data/raw/JY/IndexProduct/QT_TRADINGDAYNEW.txt'
        self.ashare_basedata_path_template = '/data/cooked/BaseData/{}/{}/{}/BASEDATA.txt'
        self.ashare_industry_path_template = '/data/cooked/Industry/{}/{}/{}/Industry.txt'
        self.ccx_stock_index_dir = '/data-platform/ccxd/prod/stock_index/'
        self.stock_index_ret_path = self.ccx_stock_index_dir + '{}/index_point/index_point.csv'
        self.stock_index_weight_path = self.ccx_stock_index_dir +'{}/weight/{}.csv'
        self.factor_index_ret_path_template = self.ccx_stock_index_dir+'factor_index/{}/index_point/index_point.csv'
        self.factor_index_weight_path = self.ccx_stock_index_dir +'factor_index/{}/weight/{}.csv'
        self.strategy_index_ret_path_template = self.ccx_stock_index_dir+'strategy_index/{}/index_point/index_point.csv'
        self.strategy_index_weight_path = self.ccx_stock_index_dir +'strategy_index/{}/weight/{}.csv'
        self.mutual_fund_dir = '/work/panyuqiong/data/fund_cooked/'
        self.mutual_fund_basedata_path_template = self.mutual_fund_dir + '{}/{}/{}/fund_data.txt' 
        self.mutual_fund_pool_path_template = '/data/public_transfer/ysun/fund_test_prod/{}/{}/{}/fund_type_ccx.csv'
        self.fund_barra_path = self.mutual_fund_dir + '{}/{}/{}/fund_barra.txt' 
        self.factot_ret_path_template = '/data/raw/HUIAN/RiskModel/{}/{}/{}/FAC_RET_sw.txt'
        self.exposure_path='/data/raw/HUIAN/RiskModel/{}/{}/{}/EXPOSUREsw.txt'
        self.factor_srisk_path='/data/raw/HUIAN/RiskModel/{}/{}/{}/srisk/d_srisk_vra.txt'
        self.factor_frisk_path='/data/raw/HUIAN/RiskModel/{}/{}/{}/frisk/d_cov_vra.txt'
        self.recent_key_template = self.mutual_fund_dir + '{}/{}/{}/recent_holding_key.txt'
        self.recent_detail_template = self.mutual_fund_dir + '{}/{}/{}/recent_holding_detail.txt'
        
        
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
    
    def get_dt_ashare_basedata_df(self, dt):
        fp = self.ashare_basedata_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        data = data[~data['SECU_CODE'].str.contains('.BJ')] #去掉类似833819.BJ的代码
        return data
    
    def get_dt_ashare_industry_df(self, dt):
        fp = self.ashare_industry_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        data = data[~data['SECU_CODE'].str.contains('.BJ')]
        return data

    def get_stock_index_ret_df(self,index_name):
        fp = self.stock_index_ret_path.format(index_name)
        data = self.get_table_from_home(fp)
        return data
    
    def get_stock_index_weight_df(self,index_name,dt): #股票代码 weight weight_pre
        fp = self.stock_index_weight_path.format(index_name,dt)
        data = self.get_table_from_home(fp)
        return data
    
    def get_factor_index_ret_df(self,index_name):
        fp = self.factor_index_ret_path_template.format(index_name)
        data = self.get_table_from_home(fp)
        return data
    
    def get_factor_index_weight_df(self,index_name,dt):
        fp = self.factor_index_weight_path.format(index_name,dt)
        data = self.get_table_from_home(fp)
        return data
    
    def get_strategy_index_ret_df(self,index_name): #Aggresive
        fp = self.strategy_index_ret_path_template.format(index_name)
        data = self.get_table_from_home(fp)
        return data
    
    def get_strategy_index_weight_df(self,index_name,dt):
        fp = self.strategy_index_weight_path.format(index_name,dt)
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_mutual_fund_basedata_df(self, dt):
        fp = self.mutual_fund_basedata_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_mutual_fund_manager_basedata_df(self, dt):
        fp = self.mutual_fund_manager_basedata_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_mutual_fund_pool_df(self, dt):
        fp = self.mutual_fund_pool_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp, sep=',')
        return data
    
    def get_dt_factor_srisk_df(self, dt): #两列SECU_CODE srisk
        fp = self.factor_srisk_path.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_factor_frisk_df(self, dt): #风险大表
        fp = self.factor_frisk_path.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_exposure_df(self, dt): #第一列股票代码 后面列是所有因子
        fp = self.exposure_path.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_fund_barra_df(self, dt): #第一列基金代码 后面列是所有因子
        fp = self.fund_barra_path.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_factor_ret_df(self, dt): #三列分别为Factor因子 DlyReturn日收益率 TradeDate交易日期
        fp = self.factot_ret_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_recent_holding_detail(self, dt): #基金大表
        fp = self.recent_detail_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_recent_holding_key(self, dt):
        fp = self.recent_key_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data



class WeeklyReport(object):
    def __init__(self,total_trading_day,end_date,out_dir='/home/liyihan/weekly_report'):
        #self.start_date = start_date 
        self.end_date = end_date
        self.total_trading_day = total_trading_day
        self.out_dir = out_dir
        self.data_api = DataAPI()
        self.broad_index_list = ['CCX全A','CCX300','CCX500','CCX800','CCX1000','CCX1800']
        self.pool_list = ['CCX300','CCX500','CCX800','CCX1000','CCX1800']
        self.factor_list = ['Value','Earning','Growth','Leverage','ResVol','Mom','Illiq','Size','SizeNL','DividendYield','BETA']
        self.factor_broad_index_list = ['Dividend','Earning','EarningQuality','Growth','HighBeta','Illiq','Leverage','LowBeta','Momentum','ResVol','Value']
        self.exposure_list = ['BETA','BTOP','SIZE','SIZENL','LEVERAGE','LIQUIDTY','GROWTH','RESVOL','MOMENTUM','EARNYILD']
        self.exposure_trade_list = ['COUNTRY', 'AERODEF', 'AGRIFOREST', 'AUTO', 'BANK', 'BUILDDECO', 'CHEM', 'COMMETRADE', 'COMPUTER', 'CONGLOMERATES', 'CONMAT', 'ELECEQP', 'ELECTRONICS', 'FOODBEVER', 'HEALTH', 'HOUSEAPP', 'IRONSTEEL', 'LEISERVICE', 'LIGHTINDUS', 'MACHIEQUIP', 'MEDIA', 'MINING', 'NONBANKFINAN', 'NONFERMETAL', 'REALESTATE', 'TELECOM', 'TEXTILE', 'TRANSPORTATION', 'UTILITIES']
        self.strategy_index_list = ['Aggressive','DividendResVol','FOF3Merged','LowRisk','Quality']
        self.dt_list_all = self.data_api.get_tradingday_between()
        self.factor_broad_name_dict = {'Value':'价值','Earning':'高盈利','Growth':'高成长','EarningQuality':'高盈利质量','Leverage':'低杠杆',
                                        'ResVol':'低波','Momentum':'动量','Illiq':'低流动性','Dividend':'高分红','HighBeta':'高Beta','LowBeta':'低Beta',
                                        'DividendResVol':'红利低波','Quality':'质量'}
        self.factor_name_dict = {'ValueFactor':'价值','EarningFactor':'高盈利','GrowthFactor':'高成长','EarningQualityFactor':'高盈利质量','LeverageFactor':'低杠杆',
                                 'ResVolFactor':'低波','MomFactor':'动量','IlliqFactor':'低流动性','SizeFactor':'市值','SizeNLFactor':'中市值','DividendYieldFactor':'高分红',
                                 'BETAFactor':'高Beta'}
        self.strategy_index_name_dict = {'Aggressive':'进取型股票指数','DividendResVol':'红利低波指数','FOF3Merged':'优秀基金经理持仓指数','LowRisk':'防御型股票指数',
                                         'Quality':'高质量指数'}
        pass
    
    def _init_dir_(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        pass
    
    def get_broad_index_ret_df(self,index): #三列分别为date daily_point（股指） daily_ret（收益率）
        if index == 'CCX全A':
            index_name = 'BroadBaseIndex'
        elif index == 'CCX300':
            index_name = 'index_300'
        elif index == 'CCX500':
            index_name = 'index_500'
        elif index == 'CCX800':
            index_name = 'index_800'
        elif index == 'CCX1000':
            index_name = 'index_1000'
        elif index == 'CCX1800':
            index_name = 'index_1800'
        else:
            raise ValueError('Please check broad index name!')
        ret_df = self.data_api.get_stock_index_ret_df(index_name)
        return ret_df
    
    def get_dt_broad_index_weight_df(self,index,dt): #股票代码 weight weight_pre
        if index == 'CCX全A':
            index_name = 'BroadBaseIndex'
        elif index == 'CCX300':
            index_name = 'index_300'
        elif index == 'CCX500':
            index_name = 'index_500'
        elif index == 'CCX800':
            index_name = 'index_800'
        elif index == 'CCX1000':
            index_name = 'index_1000'
        elif index == 'CCX1800':
            index_name = 'index_1800'
        else:
            raise ValueError('Please check broad index name!')
        ret_df = self.data_api.get_stock_index_weight_df(index_name,dt)
        return ret_df
    
    def get_index_chn_name(self,index): #中文名称获取
        if index in self.broad_index_list:
            index_name = index
        elif index in self.factor_broad_index_list:
            index_name = self.factor_broad_name_dict[index]+'100'
        elif ('Stock' in index) | ('Fund' in index):
            index_name = index
        elif index in self.strategy_index_list:
            index_name = self.strategy_index_name_dict[index]
        else:
            name_list = index.split('_')
            index_name = name_list[1] + self.factor_name_dict[name_list[0]]
        return index_name
            
    
    def get_index_performance_stats(self): #股票指数涨幅/跌幅 股票（策略）指数收益-图4/21 （第一张表）
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        index_list = self.broad_index_list + self.factor_broad_index_list + self.strategy_index_list
        for pool in self.pool_list: #股票
            for factor in self.factor_list: #因子
                index_list.append('{}Factor_{}'.format(factor,pool))
        stats = pd.DataFrame(index=index_list,columns=['近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)'])
        for index in index_list:
            ret_list = []
            if index in self.broad_index_list: #获得股票指数
                index_ret_df = self.get_broad_index_ret_df(index) #三列分别为date daily_point（股指） daily_ret（收益率）
            elif index in self.strategy_index_list: #Aggresive
                index_ret_df = self.data_api.get_strategy_index_ret_df(index)
            else: #Momentum
                index_ret_df = self.data_api.get_factor_index_ret_df(index)
            index_ret_df['date'] = index_ret_df['date'].astype(str)
            for backdays in [self.total_trading_day+1,22,64,127,253]:
                dt_list = self.dt_list_all[end_date_index-backdays+1:end_date_index+1]
                ret_df = index_ret_df[index_ret_df['date'].isin(dt_list)]
                #ret = ret_df['daily_ret'].sum() * 100
                ret = (ret_df['daily_point'].iloc[-1]/ret_df['daily_point'].iloc[0] -1) * 100
                ret_list.append(round(ret,2))
            stats.loc[index,:] = ret_list
        stats.reset_index(inplace=True)
        stats.columns = ['index'] + ['近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)']
        stats['中文名'] = stats['index'].apply(lambda x : self.get_index_chn_name(x))
        stats = stats[['index','中文名','近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)']]
        stats.set_index('index',inplace=True)
        return stats
    
    def get_index_active_performance_stats(self): #因子优化股票指数超额收益-图7/10（第二张表）
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        index_list = []
        for pool in self.pool_list:
            for factor in self.factor_list:
                index_list.append('{}Factor_{}'.format(factor,pool)) #表的索引列，均为ValueFactor_CCX300格式
        stats = pd.DataFrame(index=index_list,columns=['近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)'])
        for index in index_list:
            ret_list = []
            index_ret_df = self.data_api.get_factor_index_ret_df(index)
            index_ret_df['date'] = index_ret_df['date'].astype(str)
            base_index_ret_df = self.get_broad_index_ret_df(index.split('_')[-1])
            base_index_ret_df['date'] = base_index_ret_df['date'].astype(str)
            for backdays in [self.total_trading_day+1,22,64,127,253]:
                dt_list = self.dt_list_all[end_date_index-backdays+1:end_date_index+1]
                ret_df = index_ret_df[index_ret_df['date'].isin(dt_list)]
                base_ret_df = base_index_ret_df[base_index_ret_df['date'].isin(dt_list)]
                #ret = ret_df['daily_ret'].sum() * 100 - base_ret_df['daily_ret'].sum() * 100
                ret = (ret_df['daily_point'].iloc[-1]/ret_df['daily_point'].iloc[0] -1) * 100 - (base_ret_df['daily_point'].iloc[-1]/base_ret_df['daily_point'].iloc[0] -1)*100 #超额收益计算
                ret_list.append(round(ret,2))
            stats.loc[index,:] = ret_list
        stats.reset_index(inplace=True)
        stats.columns = ['index'] + ['近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)']
        stats['中文名'] = stats['index'].apply(lambda x : self.get_index_chn_name(x))
        stats = stats[['index','中文名','近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)']]
        stats.set_index('index',inplace=True)
        return stats

    
    def get_invest_chance_stats(self): #（第三张表）
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        index_list = self.broad_index_list + self.factor_broad_index_list + self.strategy_index_list
        # for pool in self.pool_list:
        #     for factor in self.factor_list:
        #         index_list.append('{}Factor_{}'.format(factor,pool))
        stats = pd.DataFrame(index=index_list,columns=['市场广度-3M','市场广度-6M','市场广度-12M','市场广度-36M'])
        backdays_list = [63,126,252,756]
        dt_list = self.dt_list_all[end_date_index-np.max(backdays_list)+1:end_date_index+1]
        for index in index_list:
            std_list = []
            for dt in tqdm(dt_list):
                stock_basedata_df = self.data_api.get_dt_ashare_basedata_df(dt)
                if index in self.broad_index_list:
                    index_weight_df = self.get_dt_broad_index_weight_df(index, dt)
                elif index in self.strategy_index_list:
                    index_weight_df = self.data_api.get_strategy_index_weight_df(index, dt)
                else:
                    index_weight_df = self.data_api.get_factor_index_weight_df(index, dt)
                stock_basedata_df = stock_basedata_df[stock_basedata_df['SECU_CODE'].isin(index_weight_df['secu_code'].tolist())]
                stock_basedata_df['ret'] = stock_basedata_df['CLOSEPRICE']/stock_basedata_df['PREVCLOSEPRICE'] - 1
                std_list.append(stock_basedata_df['ret'].std())
            invst_chance_list = []
            for backdays in backdays_list:
                invst_chance_list.append(round(np.nanmean(std_list[-backdays:])*np.sqrt(252)*100,2))
            stats.loc[index,:] = invst_chance_list
        stats.reset_index(inplace=True)
        stats.columns = ['index'] + ['市场广度-3M','市场广度-6M','市场广度-12M','市场广度-36M']
        stats['中文名'] = stats['index'].apply(lambda x : self.get_index_chn_name(x))
        stats = stats[['index','中文名','市场广度-3M','市场广度-6M','市场广度-12M','市场广度-36M']]
        stats.set_index('index',inplace=True)
        return stats
        
    def get_factor_ret_stats(self, factor_type = 'factor'): #股票风格因子收益表现（第四张表）
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        index_list = self.factor_list + []
        #stats = pd.DataFrame(index=index_list,columns=['近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)'])
        backdays_list = [self.total_trading_day,21,63,126,252] #一周 一个月 三个月 半年 一年
        factor_ret_df_list = []
        dt_list = self.dt_list_all[end_date_index-np.max(backdays_list)+1:end_date_index+1]
        for dt in dt_list:
            factor_ret_df = self.data_api.get_dt_factor_ret_df(dt) #各因子收益率（三列分别为Factor因子 DlyReturn日收益率 TradeDate交易日期）
            factor_ret_df_list.append(factor_ret_df)
        for backdays in backdays_list:
            for i,factor_ret_df in enumerate(factor_ret_df_list[-backdays:]):
                if factor_type == 'factor':
                    factor_ret_df = factor_ret_df[factor_ret_df['Factor'].isin(self.exposure_list)]
                else:
                    factor_ret_df = factor_ret_df[factor_ret_df['Factor'].isin(self.exposure_trade_list)]
                if i == 0:
                    factor_ret_df.rename(columns={'DlyReturn':'ret'},inplace=True)
                    full_factor_ret_df = factor_ret_df.copy()
                else: #Factor ret TradeDate DlyReturn
                    full_factor_ret_df = pd.merge(full_factor_ret_df,factor_ret_df[['Factor','DlyReturn']],on='Factor',how='left')
                    full_factor_ret_df['ret'] = full_factor_ret_df['ret'] + full_factor_ret_df['DlyReturn']
                    full_factor_ret_df = full_factor_ret_df[['Factor','ret']]
            #full_factor_ret_df.sort_values('Factor',inplace=True)
            full_factor_ret_df['ret'] = full_factor_ret_df['ret'] * 10000
            full_factor_ret_df['ret'] = full_factor_ret_df['ret'].apply(lambda x:round(x))
            full_factor_ret_df.columns = ['Factor',backdays]
            if backdays == backdays_list[0]:
                stats = full_factor_ret_df.copy()
            else:
                stats = pd.merge(stats,full_factor_ret_df,on='Factor',how='outer')
        stats.columns = ['Factor','近一周','近1个月','近3个月','近半年','近一年']  
        return stats
    
    def get_risk(self,dt_list,secu_list=[],asset_type='stock',index=None): #股票/基金风险计算
        risk_list = []
        for dt in dt_list:
            total_risk_df=pd.DataFrame(columns=['SECU_CODE','total_risk'])
            factor_cov = self.data_api.get_dt_factor_frisk_df(dt)[['FACTOR']+self.exposure_list]
            factor_cov = factor_cov[factor_cov['FACTOR'].isin(self.exposure_list)]
            factor_cov = factor_cov.fillna(0)
            factor_cov = factor_cov[['FACTOR']+factor_cov['FACTOR'].tolist()]
            if asset_type in ['stock','index']:
                base_data=self.data_api.get_dt_ashare_basedata_df(dt)
                base_data['weight'] = base_data['CLOSEPRICE'].fillna(0)*base_data['TOTALSHARES'].fillna(0) #权重：收盘价*总股本
                exposure_data = self.data_api.get_dt_exposure_df(dt)[['SECU_CODE']+factor_cov['FACTOR'].tolist()].fillna(0)
            elif asset_type == 'fund':
                base_data = self.data_api.get_dt_mutual_fund_basedata_df(dt)
                base_data.rename(columns={'fund_cap':'weight','fund_code':'SECU_CODE'},inplace=True)
                exposure_data = self.data_api.get_dt_fund_barra_df(dt)[['SECU_CODE']+factor_cov['FACTOR'].tolist()].fillna(0)
            # risk' = x*F*x'(x-exposure; F-factor_cov因子协方差矩阵)为协方差矩阵加权
            total_risk_values = np.dot(exposure_data[factor_cov['FACTOR'].tolist()].values,factor_cov[factor_cov['FACTOR'].tolist()]*0.0001/252)
            total_risk_values = np.dot(total_risk_values,exposure_data[factor_cov['FACTOR'].tolist()].values.T)
            total_risk_df['SECU_CODE'] = exposure_data['SECU_CODE']
            total_risk_df['total_risk'] = np.diagonal(total_risk_values) #取risk的对角线，即为风险
            #total_risk_df['total_risk'] = np.sqrt(np.diagonal(total_risk_values)/10000)
            total_risk_df = pd.merge(total_risk_df,base_data[['SECU_CODE','weight']]) # SECU_CODE total_risk weight
            if asset_type in ['stock','index']:
                srisk_df=self.data_api.get_dt_factor_srisk_df(dt) #两列SECU_CODE srisk
                srisk_df['SRISK'] = (srisk_df['SRISK']*0.01/np.sqrt(252)).apply(lambda x:np.square(x)) #真实年化方差
                total_risk_df = pd.merge(total_risk_df,srisk_df) #SECU_CODE total_risk weight SRISK
            else:
                holding_detail_df = self.data_api.get_dt_recent_holding_detail(dt)
                holding_detail_df = holding_detail_df[['FUNDCODE','HOLDSTOCKCODE','RATIOINNV_dt']] #三列：基金代码 持有股票代码 股票收益率贴现值
                holding_detail_df.columns = ['FUND_CODE','SECU_CODE','weight']
                srisk_df=self.data_api.get_dt_factor_srisk_df(dt) #两列SECU_CODE srisk
                srisk_df['SRISK'] = (srisk_df['SRISK']*0.01/np.sqrt(252)).apply(lambda x:np.square(x)) #真实年化方差
                holding_detail_df = pd.merge(holding_detail_df,srisk_df,on='SECU_CODE',how='left') #FUND_CODE SECU_CODE weight srisk
                holding_detail_df['SRISK'] = holding_detail_df['SRISK']*holding_detail_df['weight']
                srisk_df = holding_detail_df[['FUND_CODE','SRISK']].groupby('FUND_CODE').sum()
                srisk_df.reset_index(inplace=True)
                srisk_df.columns = ['SECU_CODE','SRISK']
                total_risk_df = pd.merge(total_risk_df,srisk_df) #SECU_CODE total_risk weight SRISK
            if asset_type == 'index': #重新算权重weigh
                if index in self.broad_index_list:
                    index_weight_df = self.get_dt_broad_index_weight_df(index, dt) #股票代码 weight weight_pre
                elif index in self.strategy_index_list:
                    index_weight_df = self.data_api.get_strategy_index_weight_df(index, dt)
                else:
                    index_weight_df = self.data_api.get_factor_index_weight_df(index, dt)
                secu_list = index_weight_df['secu_code'].tolist()
                index_weight_df = index_weight_df[['secu_code','weight']]
                index_weight_df.columns = ['SECU_CODE','weight']
            total_risk_df = total_risk_df[total_risk_df['SECU_CODE'].isin(secu_list)]
            if asset_type == 'index':
                total_risk_df = pd.merge(total_risk_df[['SECU_CODE','total_risk','SRISK']],index_weight_df,on='SECU_CODE')
            # risk = x*F*x' + s; risk*权重w/权重w总和*252（还原为非年化）
            dt_total_risk=np.dot(total_risk_df['total_risk'].fillna(0)+total_risk_df['SRISK'].fillna(0),total_risk_df['weight'].fillna(0))/total_risk_df['weight'].sum()*252
            #mkt_total_risk_list.append(dt_total_risk) 
            # srisk_df=pd.merge(srisk_df,base_data[['SECU_CODE','weight']])
            # dt_srisk=np.dot(srisk_df['SRISK'].fillna(0),srisk_df['weight'].fillna(0))/srisk_df['weight'].sum()*252
            risk = dt_total_risk 
            risk_list.append(risk)
            #mkt_srisk_list.append(dt_srisk)
        port_risk = np.nanmean(risk_list) #风险均值
        return round(port_risk,2)
    
    def get_exposure(self,dt_list,secu_list=[],asset_type='stock',index=None): #返回因子暴露数组
        for dt in dt_list:
            if asset_type in ['stock','index']:
                exposure_data = self.data_api.get_dt_exposure_df(dt).fillna(0)
            elif asset_type == 'fund':
                exposure_data = self.data_api.get_dt_fund_barra_df(dt).fillna(0)
            exposure_data = exposure_data[['SECU_CODE']+self.exposure_list]
            if asset_type == 'index':
                exposure_data = exposure_data[['SECU_CODE']+self.exposure_list]
                if index in self.broad_index_list:
                    index_weight_df = self.get_dt_broad_index_weight_df(index, dt)
                elif index in self.strategy_index_list:
                    index_weight_df = self.data_api.get_strategy_index_weight_df(index, dt)
                else:
                    index_weight_df = self.data_api.get_factor_index_weight_df(index, dt)
                secu_list = index_weight_df['secu_code'].tolist()
                index_weight_df = index_weight_df[['secu_code','weight']]
                index_weight_df.columns = ['SECU_CODE','weight']
                exposure_data = pd.merge(exposure_data,index_weight_df,on='SECU_CODE')
            exposure_data = exposure_data[exposure_data['SECU_CODE'].isin(secu_list)]
            if asset_type in ['stock','fund']:
                exposure_list = exposure_data[self.exposure_list].mean().tolist() #因子暴露平均值
            else:
                for col in self.exposure_list:
                    exposure_data[col] = exposure_data[col]*exposure_data['weight']
                exposure_list = exposure_data[self.exposure_list].sum().tolist()
            if dt == dt_list[0]:
                exposure_value = np.array(exposure_list)  
            else:
                exposure_value += np.array(exposure_list) 
        exposure_value = exposure_value/len(dt_list)
        return np.around(exposure_value,2)
    
    def get_port_exposure_stats(self,backdays,num=100): #过去一周涨幅最高/最低的前100的股票/偏股混合基金组合分析-表1 股票（策略）指数风格情况-表3/4 股票策略指数风格分析（除收益列）-表5（第五张表）
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        index_list = ['TopStock{}'.format(num),'BottomStock{}'.format(num),'TopMutualFund{}'.format(num),'BottomMutualFund{}'.format(num)] + self.broad_index_list + self.factor_broad_index_list + self.strategy_index_list
        stats = pd.DataFrame(index=index_list,columns=['收益','风险']+self.exposure_list)
        dt_list = self.dt_list_all[end_date_index-backdays+1:end_date_index+1]
        for index in index_list:
            if index in ['TopStock{}'.format(num),'BottomStock{}'.format(num),'TopMutualFund{}'.format(num),'BottomMutualFund{}'.format(num)]:
                for dt in dt_list:
                    if 'Stock' in index: #算TopStock100
                        basedata_df = self.data_api.get_dt_ashare_basedata_df(dt)
                        basedata_df = basedata_df[basedata_df['SECU_CODE'].apply(lambda x:'BJ' not in x)] #删BJ
                        basedata_df['ret'] = basedata_df['CLOSEPRICE']/basedata_df['PREVCLOSEPRICE'] - 1
                        basedata_df = basedata_df[['SECU_CODE','ret']]
                    elif 'Fund' in index:
                        basedata_df = self.data_api.get_dt_mutual_fund_basedata_df(dt)
                        basedata_df = basedata_df[basedata_df['FUNDTYPE'].isin(['股票型','混合型'])]
                        basedata_df = basedata_df[['fund_code','fund_ret']]
                        basedata_df.columns = ['SECU_CODE','ret']
                    if dt == dt_list[0]:
                        ret_df = basedata_df.copy()
                    else:
                        basedata_df.rename(columns={'ret':'prev_ret'},inplace=True)
                        ret_df = pd.merge(ret_df,basedata_df,on='SECU_CODE',how='left') #SECU_CODE ret和SECU_CODE prev_ret连接->SECU_CODE ret prev_ret
                        ret_df['ret'] = (ret_df['ret']+1) * (ret_df['prev_ret']+1) - 1 #收益率
                        ret_df = ret_df[['SECU_CODE','ret']]
                ret_df.sort_values('ret',inplace=True,ascending=False) #按收益率排序
                ret_df.dropna(axis=0, subset=['ret'], how='any', inplace=True)
                top_ret_df = ret_df.iloc[:num,:] #取前100
                bottom_ret_df = ret_df.iloc[-num:,:] #取后100
                if 'Top' in index: #top表计算
                    stats.loc[index,'收益'] = round(top_ret_df['ret'].mean()*100,2) #收益率平均值
                    if 'Stock' in index: #股票top
                        stats.loc[index,'风险'] = self.get_risk(dt_list, top_ret_df['SECU_CODE'].tolist(),asset_type='stock') #得平均风险值
                        stats.loc[index,self.exposure_list] = self.get_exposure(dt_list, top_ret_df['SECU_CODE'].tolist(),asset_type='stock') #因子暴露填充
                    else: #基金top
                        stats.loc[index,'风险'] = self.get_risk(dt_list, top_ret_df['SECU_CODE'].tolist(),asset_type='fund')
                        stats.loc[index,self.exposure_list] = self.get_exposure(dt_list, top_ret_df['SECU_CODE'].tolist(),asset_type='fund')
                else: #bottom表计算
                    stats.loc[index,'收益'] = round(bottom_ret_df['ret'].mean()*100,2)
                    if 'Stock' in index:
                        stats.loc[index,'风险'] = self.get_risk(dt_list, bottom_ret_df['SECU_CODE'].tolist(),asset_type='stock')
                        stats.loc[index,self.exposure_list] = self.get_exposure(dt_list, bottom_ret_df['SECU_CODE'].tolist(),asset_type='stock')
                    else:
                        stats.loc[index,'风险'] = self.get_risk(dt_list, bottom_ret_df['SECU_CODE'].tolist(),asset_type='fund')
                        stats.loc[index,self.exposure_list] = self.get_exposure(dt_list, bottom_ret_df['SECU_CODE'].tolist(),asset_type='fund')
            else:
                if index in self.broad_index_list:
                    index_ret_df = self.get_broad_index_ret_df(index)
                elif index in self.strategy_index_list:
                    index_ret_df = self.data_api.get_strategy_index_ret_df(index)
                else:
                    index_ret_df = self.data_api.get_factor_index_ret_df(index)
                index_ret_df['date'] = index_ret_df['date'].astype(str)
                ret_df = index_ret_df[index_ret_df['date'].isin(dt_list)]
                #ret = round(ret_df['daily_ret'].sum() * 100,2)
                ret = round((ret_df['daily_point'].iloc[-1]/ret_df['daily_point'].iloc[0] -1) * 100,2)
                stats.loc[index,'收益'] = ret
                stats.loc[index,'风险'] = self.get_risk(dt_list,asset_type = 'index',index=index)
                stats.loc[index,self.exposure_list] = self.get_exposure(dt_list,asset_type = 'index',index=index)
        stats.reset_index(inplace=True)
        stats.columns = ['index'] + ['收益','风险']+self.exposure_list
        stats['中文名'] = stats['index'].apply(lambda x : self.get_index_chn_name(x))
        stats = stats[['index','中文名']+['收益','风险']+self.exposure_list]
        stats.set_index('index',inplace=True)
        return stats

    def get_exposure_trade(self,dt_list,secu_list=[],asset_type='stock',index=None): #返回因子暴露数组
        for dt in dt_list:
            if asset_type in ['stock','index']:
                exposure_data = self.data_api.get_dt_exposure_df(dt).fillna(0)
            elif asset_type == 'fund':
                exposure_data = self.data_api.get_dt_fund_barra_df(dt).fillna(0)
            exposure_data = exposure_data[['SECU_CODE']+self.exposure_trade_list]
            exposure_data = exposure_data[exposure_data['SECU_CODE'].isin(secu_list)]
            exposure_list = exposure_data[self.exposure_trade_list].mean().tolist() #因子暴露平均值
            if dt == dt_list[0]:
                exposure_value = np.array(exposure_list)  
            else:
                exposure_value += np.array(exposure_list) 
        exposure_value = exposure_value/len(dt_list)
        return np.around(exposure_value,2)
    
    def get_relative_ret(self, exposure6, exposure22, exposure64, factor_ret, factor_type = 'factor'):
        relative_ret6 = self.get_relative_specific_ret(exposure6, factor_ret, self.total_trading_day+1, factor_type)[['Factor', 'relative_ret近一周']]
        relative_ret22 = self.get_relative_specific_ret(exposure22, factor_ret, 22, factor_type)[['Factor', 'relative_ret近1个月']]
        relative_ret64 = self.get_relative_specific_ret(exposure64, factor_ret, 64, factor_type)[['Factor', 'relative_ret近3个月']]
        relative_ret = pd.merge(relative_ret6, relative_ret22, on='Factor', how='left')
        relative_ret = pd.merge(relative_ret, relative_ret64, on='Factor', how='left')
        relative_ret.rename(columns={'relative_ret近一周':'近一周', 'relative_ret近1个月':'近1个月', 'relative_ret近3个月':'近3个月'},inplace=True)
        relative_ret = relative_ret[['Factor', '近一周', '近1个月', '近3个月']]
        return relative_ret
    
    def get_relative_specific_ret(self, exposure, factor_ret, time, factor_type = 'factor'):
        factor_ret_temp = factor_ret.copy()
        if factor_type == 'factor':
            exposure_temp = exposure[self.exposure_list]
        else:
            exposure_temp = exposure[self.exposure_trade_list]
        exposure_trans = np.transpose(exposure_temp)
        exposure_trans['exposure_diff'] = exposure_trans['TopStock100'] - exposure_trans['BottomStock100']
        factor_ret_temp['exposure_diff{}'.format(time)] = list(exposure_trans['exposure_diff'])
        if time == self.total_trading_day+1:
            factor_ret_temp['relative_ret近一周'] = factor_ret_temp['近一周'] * factor_ret_temp[f'exposure_diff{time}']
        elif time == 22:
            factor_ret_temp['relative_ret近1个月'] = factor_ret_temp['近1个月'] * factor_ret_temp['exposure_diff22']
        else:
            factor_ret_temp['relative_ret近3个月'] = factor_ret_temp['近3个月'] * factor_ret_temp['exposure_diff64']
        return factor_ret_temp

    def get_port_exposure_trade_stats(self,backdays,num=100): #行业（第七张表）
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        index_list = ['TopStock{}'.format(num),'BottomStock{}'.format(num),'TopMutualFund{}'.format(num),'BottomMutualFund{}'.format(num)]
        stats = pd.DataFrame(index=index_list,columns=self.exposure_trade_list)
        dt_list = self.dt_list_all[end_date_index-backdays+1:end_date_index+1]
        for index in index_list:
            for dt in dt_list:
                if 'Stock' in index: #算TopStock100
                    basedata_df = self.data_api.get_dt_ashare_basedata_df(dt)
                    basedata_df = basedata_df[basedata_df['SECU_CODE'].apply(lambda x:'BJ' not in x)] #删BJ
                    basedata_df['ret'] = basedata_df['CLOSEPRICE']/basedata_df['PREVCLOSEPRICE'] - 1
                    basedata_df = basedata_df[['SECU_CODE','ret']]
                elif 'Fund' in index:
                    basedata_df = self.data_api.get_dt_mutual_fund_basedata_df(dt)
                    basedata_df = basedata_df[basedata_df['FUNDTYPE'].isin(['股票型','混合型'])]
                    basedata_df = basedata_df[['fund_code','fund_ret']]
                    basedata_df.columns = ['SECU_CODE','ret']
                if dt == dt_list[0]:
                    ret_df = basedata_df.copy()
                else:
                    basedata_df.rename(columns={'ret':'prev_ret'},inplace=True)
                    ret_df = pd.merge(ret_df,basedata_df,on='SECU_CODE',how='left') #SECU_CODE ret和SECU_CODE prev_ret连接->SECU_CODE ret prev_ret
                    ret_df['ret'] = (ret_df['ret']+1) * (ret_df['prev_ret']+1) - 1 #收益率
                    ret_df = ret_df[['SECU_CODE','ret']]
            ret_df.sort_values('ret',inplace=True,ascending=False)
            top_ret_df = ret_df.iloc[:num,:] #取前100
            bottom_ret_df = ret_df.iloc[-num:,:] #取后100
            if 'Top' in index: #top表计算
                if 'Stock' in index: #股票top
                    stats.loc[index,self.exposure_trade_list] = self.get_exposure_trade(dt_list, top_ret_df['SECU_CODE'].tolist(),asset_type='stock') #因子暴露填充
                else: #基金top
                    stats.loc[index,self.exposure_trade_list] = self.get_exposure_trade(dt_list, top_ret_df['SECU_CODE'].tolist(),asset_type='fund')
            else: #bottom表计算
                if 'Stock' in index:
                    stats.loc[index,self.exposure_trade_list] = self.get_exposure_trade(dt_list, bottom_ret_df['SECU_CODE'].tolist(),asset_type='stock')
                else:
                    stats.loc[index,self.exposure_trade_list] = self.get_exposure_trade(dt_list, bottom_ret_df['SECU_CODE'].tolist(),asset_type='fund')
        stats.reset_index(inplace=True)
        stats.columns = ['index'] + self.exposure_trade_list
        exposure_difference = [100]
        for i, row in stats.iteritems():
            if i != 'index':
                exposure_difference.append(row[0] - row[1])
        stats.loc[len(stats)] = exposure_difference
        stats.sort_values(axis = 1, by = 4, ascending = False, inplace=True)
        stats.drop(index = 4, inplace = True)
        exposure_trade_list_sort = stats.columns.values.tolist()
        exposure_trade_list_sort.remove('index')
        stats['中文名'] = stats['index'].apply(lambda x : self.get_index_chn_name(x))
        stats = stats[['index','中文名']+exposure_trade_list_sort]
        stats.set_index('index',inplace=True)
        return stats
    
    def get_predict_vol(self,index):
        dt = [dt for dt in self.dt_list_all if dt<=self.end_date][-1] #最近的一个交易日
        if index in self.broad_index_list:
            index_weight_df = self.get_dt_broad_index_weight_df(index, dt) #股票代码 weight weight_pre
        elif index in self.strategy_index_list:
            index_weight_df = self.data_api.get_strategy_index_weight_df(index, dt)
        else:
            index_weight_df = self.data_api.get_factor_index_weight_df(index, dt)
        secu_list = index_weight_df['secu_code'].tolist()
        index_weight_df = index_weight_df[['secu_code','weight']]
        index_weight_df.columns = ['SECU_CODE','weight']
        
        total_risk_df=pd.DataFrame(columns=['SECU_CODE','total_risk'])
        factor_cov = self.data_api.get_dt_factor_frisk_df(dt)[['FACTOR']+self.exposure_list]
        factor_cov = factor_cov[factor_cov['FACTOR'].isin(self.exposure_list)]
        factor_cov = factor_cov.fillna(0)
        factor_cov = factor_cov[['FACTOR']+factor_cov['FACTOR'].tolist()]
        exposure_data = self.data_api.get_dt_exposure_df(dt)[['SECU_CODE']+factor_cov['FACTOR'].tolist()].fillna(0)
        total_risk_values = np.dot(exposure_data[factor_cov['FACTOR'].tolist()].values,factor_cov[factor_cov['FACTOR'].tolist()]*0.0001/252)
        total_risk_values = np.dot(total_risk_values,exposure_data[factor_cov['FACTOR'].tolist()].values.T)
        total_risk_df['SECU_CODE'] = exposure_data['SECU_CODE']
        total_risk_df['total_risk'] = np.diagonal(total_risk_values)
        srisk_df = self.data_api.get_dt_factor_srisk_df(dt)
        srisk_df['SRISK'] = (srisk_df['SRISK']*0.01/np.sqrt(252)).apply(lambda x:np.square(x))
        total_risk_df = pd.merge(total_risk_df,srisk_df)
        total_risk_df = total_risk_df[total_risk_df['SECU_CODE'].isin(secu_list)]
        total_risk_df = pd.merge(total_risk_df[['SECU_CODE','total_risk','SRISK']],index_weight_df,on='SECU_CODE')
        dt_total_risk=np.dot(total_risk_df['total_risk'].fillna(0)+total_risk_df['SRISK'].fillna(0),total_risk_df['weight'].fillna(0))/total_risk_df['weight'].sum()*252
        return round(dt_total_risk*100,2)
    
    def get_index_vol_stats(self): #股票（策略）指数风险情况-图5/22（第六张表）
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        index_list = self.broad_index_list + self.factor_broad_index_list + self.strategy_index_list
        for pool in self.pool_list:
            for factor in self.factor_list:
                index_list.append('{}Factor_{}'.format(factor,pool))
        stats = pd.DataFrame(index=index_list,columns=['近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)'])
        for index in index_list:
            vol_list = []
            if index in self.broad_index_list:
                index_ret_df = self.get_broad_index_ret_df(index) #三列分别为date daily_point（股指） daily_ret（收益率）
            elif index in self.strategy_index_list:
                index_ret_df = self.data_api.get_strategy_index_ret_df(index)
            else:
                index_ret_df = self.data_api.get_factor_index_ret_df(index)
            index_ret_df['date'] = index_ret_df['date'].astype(str)
            for backdays in [self.total_trading_day,21,63,126,252]:
                dt_list = self.dt_list_all[end_date_index-backdays+1:end_date_index+1]
                ret_df = index_ret_df[index_ret_df['date'].isin(dt_list)]
                vol = ret_df['daily_ret'].std() *np.sqrt(252)* 100
                vol_list.append(round(vol,2))
            stats.loc[index,:] = vol_list
        stats.reset_index(inplace=True)
        stats.columns = ['index'] + ['近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)']
        stats['中文名'] = stats['index'].apply(lambda x : self.get_index_chn_name(x))
        stats = stats[['index','中文名','近一周(%)','近1个月(%)','近3个月(%)','近半年(%)','近一年(%)']]
        stats['predict_vol(%)'] = stats['index'].apply(lambda x:self.get_predict_vol(x)) #计算predict_vol
        stats.set_index('index',inplace=True)
        return stats

    def get_attribution(self, backdays, relative = 'no'):
        end_date_index = self.dt_list_all.index([dt for dt in self.dt_list_all if dt<=self.end_date][-1])
        columns_list = self.exposure_list+self.exposure_trade_list
        stats = pd.DataFrame(index=self.strategy_index_list,columns=columns_list)
        dt_list = self.dt_list_all[end_date_index-backdays+1:end_date_index+1]

        factor_ret_df_list = []
        for index in self.strategy_index_list:
            for dt in dt_list:
                index_weight_df = self.data_api.get_strategy_index_weight_df(index, dt)[['secu_code', 'weight']] #权重
                if relative == 'relative':
                    index_weight_df_base = self.get_dt_broad_index_weight_df('CCX1800', dt)[['secu_code', 'weight']]
                    index_weight_df_base.rename(columns={'weight':'weight_base'},inplace=True)
                    index_weight_df = pd.merge(index_weight_df,index_weight_df_base,on='secu_code',how='left').fillna(0)
                    index_weight_df['weight'] = index_weight_df['weight'] - index_weight_df['weight_base']
                    index_weight_df = index_weight_df[['secu_code', 'weight']]
                exposure_data = self.data_api.get_dt_exposure_df(dt).fillna(0)
                exposure_data.rename(columns={'SECU_CODE':'secu_code'},inplace=True)
                index_exposure = pd.merge(index_weight_df,exposure_data,on='secu_code',how='left')
                for factor in self.exposure_list+self.exposure_trade_list:
                    index_exposure[factor] = index_exposure[factor] * index_exposure['weight']
                index_exposure = index_exposure[self.exposure_list+self.exposure_trade_list]
                factor_ret_df = self.data_api.get_dt_factor_ret_df(dt)[['Factor', 'DlyReturn']]
                for i, row in index_exposure.iteritems():
                    ret = factor_ret_df.loc[factor_ret_df['Factor'] == i, 'DlyReturn'].tolist()
                    index_exposure[i] = index_exposure[i] * ret[0]
                index_exposure.loc['sum']=index_exposure.apply(lambda x:sum(x))
                factor_ret_df_list.append(index_exposure[-1:])
            for i,factor_ret_df in enumerate(factor_ret_df_list):
                if i == 0:
                    full_factor_ret_df = factor_ret_df.copy()
                else:
                    full_factor_ret_df = pd.concat([full_factor_ret_df, factor_ret_df], ignore_index=True)
            full_factor_ret_df.loc[index]=full_factor_ret_df.apply(lambda x:sum(x))
            full_factor_ret_df.loc[index] = full_factor_ret_df.loc[index] * 10000
            full_factor_ret_df.loc[index] = full_factor_ret_df.loc[index].apply(lambda x:round(x))
            stats.loc[index] = list(full_factor_ret_df.loc[index])
        stats['index'] = self.strategy_index_list
        stats['中文名'] = stats['index'].apply(lambda x : self.get_index_chn_name(x))
        stats = stats[['中文名']+self.exposure_list+self.exposure_trade_list]
        return stats
    
    def run(self):
        self._init_dir_(self.out_dir)
        writer = pd.ExcelWriter(self.out_dir+f'weekly_report{self.end_date}.xlsx')
        index_performance_stats = self.get_index_performance_stats() #第一张表
        index_active_stats = self.get_index_active_performance_stats() #第二张表
        invest_chance_stats = self.get_invest_chance_stats() #第三张表
        factor_ret_stats = self.get_factor_ret_stats() #第四张表
        factor_trade_ret_stats = self.get_factor_ret_stats('trade')
        weekly_exposure_stats = self.get_port_exposure_stats(self.total_trading_day+1) #第五张表
        #monthly_exposure_stats = self.get_port_exposure_stats(21)
        #half_yearly_exposure_stats = self.get_port_exposure_stats(126)
        index_vol_stats = self.get_index_vol_stats() #第六张表
        port_exposure_trade_stats = self.get_port_exposure_trade_stats(self.total_trading_day+1)
        relative_ret_stats = self.get_relative_ret(weekly_exposure_stats, self.get_port_exposure_stats(22), self.get_port_exposure_stats(64), factor_ret_stats)
        relative_trade_ret_stats = self.get_relative_ret(port_exposure_trade_stats, self.get_port_exposure_trade_stats(22), self.get_port_exposure_trade_stats(64), factor_trade_ret_stats, 'trade')
        strategy_attribution = self.get_attribution(self.total_trading_day)
        strategy_relative_attribution = self.get_attribution(self.total_trading_day, 'relative')

        index_performance_stats.to_excel(writer,sheet_name='index_performance_stats')
        index_active_stats.to_excel(writer,sheet_name='index_active_stats')
        invest_chance_stats.to_excel(writer,sheet_name='invest_chance_stats')
        factor_ret_stats.to_excel(writer,sheet_name='factor_ret_stats')
        factor_trade_ret_stats.to_excel(writer,sheet_name='factor_trade_ret_stats')
        weekly_exposure_stats.to_excel(writer,sheet_name='weekly_exposure_stats')
        #monthly_exposure_stats.to_excel(writer,sheet_name='monthly_exposure_stats')
        #half_yearly_exposure_stats.to_excel(writer,sheet_name='half_yearly_exposure_stats')
        index_vol_stats.to_excel(writer,sheet_name='index_vol_stats')
        port_exposure_trade_stats.to_excel(writer,sheet_name='port_exposure_trade_stats')
        relative_ret_stats.to_excel(writer,sheet_name='relative_ret_stats')
        relative_trade_ret_stats.to_excel(writer,sheet_name='relative_trade_ret_stats')
        strategy_attribution.to_excel(writer,sheet_name='strategy_attribution')
        strategy_relative_attribution.to_excel(writer,sheet_name='strategy_relative_attribution')
        writer.save()
        pass
    
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    date = '20221209' # 每周产出时改为当周最后一个交易日的日期（一般为每周五）
    total_trading_day = 5 # 当周实际交易日的数量（春节/国庆等法定假期所在周的周交易日数量可能会减少，根据官网交易日历调整）
    out_dir = '/home/liyihan/weekly_report/'
    test = WeeklyReport(test = WeeklyReport(end_date=date, total_trading_day=total_trading_day, out_dir=out_dir))
    test.run()
    
    
    