'''事件类因子用data_api'''
import pandas as pd 
import numpy as np
import datetime


class DATAAPI(object):
    def __init__(self):
        self.exg_trading_day_path = '/data/raw/WIND/ASHARECALENDAR.txt'
        self.ib_trading_day_path = '/data/raw/JY/IndexProduct/QT_TRADINGDAYNEW.txt'
        self.basedata_path_template = '/data/cooked/BaseData/{}/{}/{}/BASEDATA.txt'
        self.industry_path_template = '/data/cooked/Industry/{}/{}/{}/Industry.txt'
        self.bs_path_template = '/data/raw/JY/Fundamental/{}/{}/{}/LC_BALANCESHEETALL_MERGED.txt'
        self.is_path_template = '/data/raw/JY/Fundamental/{}/{}/{}/LC_INCOMESTATEMENTALL_MERGED.txt'
        self.cfs_path_template = '/data/raw/JY/Fundamental/{}/{}/{}/LC_CASHFLOWSTATEMENTALL_MERGED.txt'
        self.indexquote_path_template = '/data/raw/JY/IndexQuote/{}/{}/{}/QT_INDEXQUOTE_ALL.txt'
        self.hs300_weight_template = '/data/raw/JY/IndexWeight/{}/{}/{}/SA_TRADABLESHARE_HS300.txt'
        self.zz500_weight_template = '/data/raw/JY/IndexWeight/{}/{}/{}/SA_TRADABLESHARE_CSI500.txt'
        self.exposure_path_template = '/data/raw/HUIAN/RiskModel/{}/{}/{}/EXPOSUREsw.txt'
        self.factor_model_template = '/work/panyuqiong/data/factor_model/{}/{}/{}/{}/FACTOR_PREMIUM.txt'
        self.f_cov_path_template = '/data/raw/HUIAN/RiskModel/{}/{}/{}/frisk/d_cov_vra.txt'    
        self.f_srisk_path_template = '/data/raw/HUIAN/RiskModel/{}/{}/{}/srisk/d_srisk_vra.txt'    
        self.gogoal_consensus_path_template = '/data/raw/GOGOAL/Consensus/{}/{}/{}/CON_FORECAST_STK.txt'
        self.diviend_path_template = '/data/raw/JY/BaseData/{}/{}/{}/LC_DIVIDENDPROCESS.txt'
        self.kcb_diviend_path_template = '/data/raw/JY/BaseData/{}/{}/{}/LC_STIBDIVIDEND_new.txt'
        self.factor_path_template = '/work/panyuqiong/data/stock_factor_prod/{}/{}/{}/{}.txt'
        self.fof_stock_path_template = '/work/panyuqiong/data/fof_stock_factor/{}/{}/{}/{}.txt'
        self.zz800_path_template = '/data/raw/JY/IndexWeight/{}/{}/{}/INDEXWEIGHT_CSI000906.txt'
        self.zz1000_path_template = '/data/raw/JY/IndexWeight/{}/{}/{}/INDEXWEIGHT_CSI000852.txt'
        self.fund_skill_template = '/work/panyuqiong/data/fund_skill/{}.txt'
        self.fund_skill_dt_template = '/work/panyuqiong/data/fund_skill/{}/{}/{}/{}.txt'
        self.fund_cooked_dir = '/work/panyuqiong/data/fund_cooked/'
        self.recent_key_template = self.fund_cooked_dir + '{}/{}/{}/recent_holding_key.txt'
        self.recent_detail_template = self.fund_cooked_dir + '{}/{}/{}/recent_holding_detail.txt'
        self.wind_consensus_path_template = '/data/raw/WIND/Consensus/{}/{}/{}/ASHARECONSENSUSROLLINGDATA.txt'
        self.industry_factor = ['AERODEF','AGRIFOREST','AUTO','BANK','BUILDDECO','CHEM','COMMETRADE','COMPUTER',\
                                'CONGLOMERATES','CONMAT','ELECEQP','ELECTRONICS','FOODBEVER','HEALTH','HOUSEAPP',\
                                'IRONSTEEL','LEISERVICE','LIGHTINDUS','MACHIEQUIP','MEDIA','MINING','NONBANKFINAN',\
                                'NONFERMETAL','REALESTATE','TELECOM','TEXTILE','TRANSPORTATION','UTILITIES']
        self.style_factor = ['SIZE','SIZENL','BETA','BTOP','EARNYILD','GROWTH','LEVERAGE','LIQUIDTY','MOMENTUM','RESVOL']
        self.new_style_factor = ['BETAFactor','EarningFactor','GrowthFactor','IlliqFactor','LeverageFactor','MomFactor',
                            'QualityFactor','ResVolFactor','SizeFactor','SizeNLFactor','ValueFactor','DividendYield_SP']
        self.barra_factor = self.style_factor + self.industry_factor
        self.firm_charac_template = '/work/panyuqiong/data/factor_model/FF-3/{}/{}/{}/FIRM_CHARACTERISTIC.txt'
        self.analyst_dir = '/data/raw/GOGOAL/'
        self.analyst_consensus_dir = self.analyst_dir + 'Consensus/'
        self.analyst_derivative_dir = self.analyst_dir + 'Derivative/'
        self.analyst_earning_forecast_dir = self.analyst_dir + 'EarningForecast/'
        self.full_derivative_path = self.analyst_derivative_dir + '{}.txt'
        self.full_earning_forecast_path = self.analyst_earning_forecast_dir + '{}.txt'
        self.consensus_df_path_template = self.analyst_consensus_dir + '{}/{}/{}/{}.txt'
        self.derivative_df_path_template = self.analyst_derivative_dir + '{}/{}/{}/{}.txt'
        self.earning_forecast_df_path_template = self.analyst_earning_forecast_dir + '{}/{}/{}/{}.txt'
        self.sepcific_ret_path_template = '/data/raw/HUIAN/RiskModel/{}/{}/{}/SPECIFIC_RET_sw.txt'
        self.ASHARENOTICE = '/data/raw/WIND/Fundamental/{}/{}/{}/ASHAREPROFITNOTICE.txt'
        self.ASHAREPROFITEXPRESS = '/data/raw/WIND/Fundamental/{}/{}/{}/ASHAREPROFITEXPRESS.txt'
        self.calendar_path = '/data/raw/WIND/ASHARECALENDAR.txt'
        pass

    def get_table_from_database(self, table_name, fields=None, condition=None, user_name='jydb', passwd='jydb', address='117.122.203.7:1521', db_name='jydb'):
        con = create_engine('oracle+cx_oracle://' + user_name +
                            ':' + passwd + '@' + address + '/' + db_name)
        fields_str = '*' if fields is None else ','.join(fields)
        if condition is None:
            sql = 'SELECT ' + fields_str + ' FROM ' + table_name
        else:
            sql = 'SELECT ' + fields_str + ' FROM ' + table_name + ' ' + condition
        df = pd.read_sql(sql, con)
        return df

    def get_table_from_home(self, table_path, fields=None, na_values=['None'], sep='|', dtype={}):
        df = pd.read_csv(table_path, sep=sep, usecols=fields, na_values=na_values,
                         dtype=dtype, low_memory=False, error_bad_lines=False)
        return df

    def get_pre_tradingday(self, dt):
        dt_ls = self.get_tradingday_between()
        dt_ls = sorted(dt_ls + [dt])
        return(dt_ls[dt_ls.index(dt)-1])

    def get_calendar_df(self):
        fp = self.calendar_path
        data = self.get_table_from_home(fp)
        return data 
    
    def get_trading_days(self, start_date=None, end_date=None, back_days=0): # 输入为str格式,如果START_DATE为None，则回溯最近的nday
        full_calendar = self.get_calendar_df()
        calendar = full_calendar
        if start_date is not None:
            calendar = calendar[calendar["SECU_CODE"] >= int(start_date)]
        if end_date is not None:
            calendar = calendar[calendar["SECU_CODE"] <= int(end_date)]
        if back_days > 0:
            if start_date is None:
                res = calendar.loc[calendar.index[-back_days:], "SECU_CODE"]
            else:
                back_days_list = full_calendar[full_calendar["SECU_CODE"] < int(start_date)]
                res = back_days_list["SECU_CODE"].iloc[-back_days:].append(calendar["SECU_CODE"])
        else:
            res = calendar["SECU_CODE"]
        return [str(dt) for dt in res.tolist()]
    
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

    def transfer_daily2month(self, dt_daily_list, if_start=True):
        dt_all_list = self.get_tradingday_between()
        start_di = dt_all_list.index(dt_daily_list[0])
        end_di = dt_all_list.index(dt_daily_list[-1])
        dt_daily_list_merged = dt_all_list[start_di-1:end_di+1]
        dt_month_list = []
        for prev_dt, dt in zip(dt_daily_list_merged[:-1], dt_daily_list_merged[1:]):
            if int(prev_dt[:6]) - int(dt[:6]) != 0:
                if if_start:  # 取月初
                    dt_month_list.append(dt)
                else:
                    dt_month_list.append(prev_dt)
        return dt_month_list

    def transfer_daily2quarter(self, dt_daily_list, if_start=True):
        dt_all_list = self.get_tradingday_between()
        start_di = dt_all_list.index(dt_daily_list[0])
        end_di = dt_all_list.index(dt_daily_list[-1])
        dt_daily_list_merged = dt_all_list[start_di-1:end_di+1]
        dt_quarter_list = [str(pd.to_datetime(x).year) + 'Q' +
                           str(pd.to_datetime(x).quarter) for x in dt_daily_list_merged]
        map_dict = dict(zip(dt_daily_list_merged, dt_quarter_list))
        res_quarter_list = []
        for prev_dt, dt in zip(dt_daily_list_merged[:-1], dt_daily_list_merged[1:]):
            if map_dict[prev_dt] != map_dict[dt]:
                if if_start:
                    res_quarter_list.append(dt)
                else:
                    res_quarter_list.append(prev_dt)
        return res_quarter_list

    def get_dt_hs300_weight_df(self, dt):
        fp = self.hs300_weight_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_zz500_weight_df(self, dt):
        fp = self.zz500_weight_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_zz800_weight_df(self, dt):
        fp = self.zz800_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_zz1000_weight_df(self, dt):
        fp = self.zz1000_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_top70_df(self, dt):
        fp = self.basedata_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        data['cap'] = data['CLOSEPRICE']*data['TOTALSHARES']
        data['rank']=data['cap'].rank(pct=True)
        data = data[data['rank']>0.3]
        return data

    def get_dt_basedata_df(self, dt):
        fp = self.basedata_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_industry_df(self, dt):
        fp = self.industry_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_indexquote_df(self, dt):
        fp = self.indexquote_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_bs_df(self, dt, columns, if_adjust):
        fp = self.bs_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        if columns is None:
            data = self.get_table_from_home(fp)
        else:
            columns_list = ['SECU_CODE', 'IFADJUSTED', 'BULLETINTYPE', 'IFCOMPLETE', 'IFMERGED', 'INFOPUBLDATE', 'ENDDATE'] + columns
            data = self.get_table_from_home(fp, columns_list)
        data['is688'] = [code[:3]=="688" for code in data['SECU_CODE']]
        data = data[(data['BULLETINTYPE'].isin([10,20]) | data['is688']) & (data['IFCOMPLETE']==1)]
        data.drop(columns=['is688'], inplace=True)
        if if_adjust:
            data = data[data['IFMERGED']==1]
        else:
            data = data[(data['IFMERGED']==1)&(data['IFADJUSTED']==2)]
        return data

    def get_dt_is_df(self, dt, columns, if_adjust):
        fp = self.is_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        if columns is None:
            data = self.get_table_from_home(fp)
        else:
            columns_list = ['SECU_CODE', 'IFADJUSTED', 'BULLETINTYPE', 'IFCOMPLETE', 'IFMERGED', 'INFOPUBLDATE', 'ENDDATE'] + columns
            data = self.get_table_from_home(fp, columns_list)
        data['is688'] = [code[:3]=="688" for code in data['SECU_CODE']]
        data = data[(data['BULLETINTYPE'].isin([10,20]) | data['is688']) & (data['IFCOMPLETE']==1)]
        data.drop(columns=['is688'], inplace=True)
        if if_adjust:
            data = data[(data['IFMERGED']==1)&data['IFADJUSTED'].isin([1,2])]
        else:
            data = data[(data['IFMERGED']==1)&(data['IFADJUSTED']==2)]
        return data

    def get_dt_cfs_df(self, dt, columns, if_adjust):
        fp = self.cfs_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        if columns is None:
            data = self.get_table_from_home(fp, columns)
        else:
            columns_list = ['SECU_CODE', 'IFADJUSTED', 'BULLETINTYPE', 'IFCOMPLETE', 'IFMERGED', 'INFOPUBLDATE', 'ENDDATE'] + columns
            data = self.get_table_from_home(fp, columns_list)
        data['is688'] = [code[:3]=="688" for code in data['SECU_CODE']]
        data = data[(data['BULLETINTYPE'].isin([10,20]) | data['is688']) & (data['IFCOMPLETE']==1)]
        data.drop(columns=['is688'], inplace=True)
        if if_adjust:
            data = data[(data['IFMERGED']==1)&data['IFADJUSTED'].isin([1,2])]
        else:
            data = data[(data['IFMERGED']==1)&(data['IFADJUSTED']==2)]
        return data

    def get_dt_exposure_df(self, dt):
        fp = self.exposure_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_cov_df(self, dt):
        fp = self.f_cov_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_srisk_df(self, dt):
        fp = self.f_srisk_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_factor_premium_df(self, dt, factor='FF-5'):  # CARHAT FF-3 FF-5
        fp = self.factor_model_template.format(factor, int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_consensus_df(self, dt):
        fp = self.gogoal_consensus_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:8]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_dividend_df(self, dt):
        fp = self.diviend_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_kcb_dividend_df(self, dt):
        fp = self.kcb_diviend_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_factor_df(self, dt, factor_name):
        fp = self.factor_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]), factor_name)
        data = self.get_table_from_home(fp)
        return data

    def get_dt_fof_stock_factor_df(self, dt, factor_name):
        fp = self.fof_stock_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]), factor_name)
        data = self.get_table_from_home(fp)
        return data

    def get_fund_skill_df(self,skill_name='CarhartAlphaD63'):
        fp = self.fund_skill_template.format(skill_name)
        data = self.get_table_from_home(fp)
        return data
  
    def get_dt_fund_skill_df(self, dt, skill_name):
        fp = self.fund_skill_dt_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]), skill_name)
        data = self.get_table_from_home(fp)
        return data

    def get_dt_recent_holding_detail_df(self, dt):
        fp = self.recent_detail_template.format(
            int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_recent_holding_key_df(self, dt):
        fp = self.recent_key_template.format(
            int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_wind_consensus_df(self, dt):
        fp = self.wind_consensus_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_style_exposure_df(self, dt):
        fp = self.factor_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]), 'full_factor') 
        data = self.get_table_from_home(fp)
        return data

    def get_dt_firm_charac_df(self,dt):
        fp = self.firm_charac_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data

    def get_dt_analyst_derivative_df(self, dt, file_name):
        fp = self.derivative_df_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]),file_name)
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_analyst_consensus_df(self, dt, file_name):
        fp = self.consensus_df_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]),file_name)
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_analyst_earning_forecast_df(self, dt, file_name):
        fp = self.earning_forecast_df_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]),file_name)
        data = self.get_table_from_home(fp)
        return data
    
    def get_full_analyst_earning_forecast_df(self, file_name):
        fp = self.full_earning_forecast_path.format(file_name)
        data = self.get_table_from_home(fp)
        return data
    
    def get_full_analyst_derivative_df(self, file_name):
        fp = self.full_derivative_path.format(file_name)
        data = self.get_table_from_home(fp)
        return data

    def get_dt_specific_ret(self, dt):
        fp = self.sepcific_ret_path_template.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_profit_notice(self, dt):
        fp = self.ASHARENOTICE.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data
    
    def get_dt_profit_express(self, dt):
        fp = self.ASHAREPROFITEXPRESS.format(int(dt[:4]), int(dt[4:6]), int(dt[6:]))
        data = self.get_table_from_home(fp)
        return data