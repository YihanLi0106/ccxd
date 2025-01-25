from fcntl import DN_DELETE
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np
from datetime import datetime, date
from optparse import OptionParser
import os

object_position_path = '/data-platform/ccxd/prod/stock_index/Thematic_index/{}_xinjingbao/weight/{}.csv'
barra_path = '/data-zero/raw/HUIAN/RiskModel/{}/{}/{}/{}'
ccx1800_position_path = '/data-platform/ccxd/prod/stock_index/index_1800/weight/{}.csv'

q_date_huxia = {
    '20200331':'2020Q1',
    '20200630':'2020Q2',
    '20200930':'2020Q3',
    '20201231':'2020Q4',
    '20210331':'2021Q1',
    '20210630':'2021Q2',
    '20210930':'2021Q3',
    '20211231':'2021Q4',
    '20220331':'2022Q1',
    '20220630':'2022Q2',
}

def get_date_list(st, ed):
    all_date_list =  pd.read_csv('/data-zero/raw/WIND/ASHARECALENDAR.txt',dtype={'SECU_CODE':str})
    all_date_list = all_date_list[(all_date_list['SECU_CODE'] >=st) & (all_date_list['SECU_CODE'] <=ed)]['SECU_CODE']
    return all_date_list.to_list()
    #我写的
"""     rebalance_date_list = []
    rebalance_path = '/data-platform/ccxd/prod/stock_index/Thematic_index/ESG_xinjingbao/rebalance_records/'
    file_name_list = os.listdir(rebalance_path)
    for time in file_name_list:
        if len(time) == 30:
            rebalance_date_list.append(time[18:26])
    return all_date_list.to_list(), rebalance_date_list """
"""     if src == 'huaxia':
        rebalance_date_list = list(q_date_huxia.keys())
    elif src == 'yifangda':
        rebalance_date_list = list(pd.read_excel('/work/ysun/stock_index/lowrick/yifangda/优化组合持仓.xls', dtype={'调仓日期':str}, usecols=['调仓日期']).drop_duplicates()['调仓日期'].sort_values().values)
    elif src == 'fuguo':
        # rebalance_date_list = list(pd.read_excel('/work/ysun/stock_index/lowrick/fuguo/FY100指数增强组合持仓.xlsx', dtype={'Date':str}, usecols=['Date']).drop_duplicates()['Date'].sort_values().values)
        rebalance_date_list = list(pd.read_csv('/work/ysun/stock_index/lowrick/fuguo/FY100指数增强组合持仓.csv', dtype={'Date':str}, usecols=['Date']).drop_duplicates()['Date'].sort_values().values)
        rebalance_date_list = [d.replace('-', '') for d in rebalance_date_list]

    else:
        rebalance_date_list = all_date_list.to_list() """




def get_position_actual(index_name, dd, src):
    if src == 'huaxia':
        position = get_position_huaxia(dd)
    elif src=='yifangda':
        position = get_position_yifangda(dd)
    elif src=='fuguo':
        position = get_position_fuguo(dd)
    elif src[:4]=='tech':
        position = get_position_techindex(dd, src)

    else:
        position = get_actual_position_ccx(index_name, dd, object_position_path) #这个是自定义的路径

    return position


def get_position_techindex(dd, src):
    dd = datetime.strftime(datetime.strptime(dd, '%Y%m%d'), '%Y-%m-%d')
    index_type = src[5:]
    position = pd.read_csv('/work/ysun/stock_index/tech_index/res/{}/index_weight/index_weight_20130104__20220831.csv'.format(index_type))
    position = position[position['date'] == dd][['secu_code', 'weight']]
    position['weight'] = position['weight'] * 100
    position.columns=['secu_code', 'actual_weight']

    return position


def get_position_huaxia(dd):
    position = pd.read_excel('/work/ysun/stock_index/lowrisk/huaxia/sim_file.xlsx', sheet_name=q_date_huxia[dd])
    position.columns=['secu_code', 'actual_weight']
    return position

def get_position_yifangda(dd):
    position = pd.read_excel('/work/ysun/stock_index/lowrick/yifangda/优化组合持仓.xls', dtype={'调仓日期':str})
    cur_position = position[position['调仓日期']==dd][['证券代码', '权重(%)']]
    cur_position.columns=['secu_code', 'actual_weight']
    cur_position['actual_weight'] = cur_position['actual_weight'] / cur_position['actual_weight'].sum() * 100
    return cur_position

def get_position_fuguo(dd):
    # position = pd.read_excel('/work/ysun/stock_index/lowrick/fuguo/FY100指数增强组合持仓.xlsx', dtype={'Date':str})
    position = pd.read_csv('/work/ysun/stock_index/lowrick/fuguo/FY100指数增强组合持仓.csv', dtype={'Date':str})
    dd = datetime.strftime(datetime.strptime(dd, '%Y%m%d'), '%Y-%m-%d')
    cur_position = position[position['Date']==dd][['Code', 'Weight']]
    cur_position.columns=['secu_code', 'actual_weight']
    cur_position['actual_weight'] = cur_position['actual_weight'] / cur_position['actual_weight'].sum() * 100
    return cur_position

def get_actual_position_ccx(index_name, dd, path):
    position = pd.read_csv(path.format(index_name, dd), sep='|')
    cur_position = position.copy(deep=True)[['secu_code', 'weight']]
    cur_position.rename(columns={'weight':'actual_weight'}, inplace=True)
    cur_position['actual_weight'] = cur_position['actual_weight'] / cur_position['actual_weight'].sum() * 100

    return cur_position

def get_position_benchmark(dd, src, benchmark_position_path):
    if (src == 'raw') or not benchmark_position_path:
        position = pd.DataFrame(columns=['date', 'secu_code', 'weight'])
    elif src !='fuguo':
        position = pd.read_csv(benchmark_position_path.format(dd), sep='|', dtype={'date':str})
    elif src == 'raw':
        position = pd.DataFrame(columns=['date', 'secu_code', 'weight'])
    else:
        position = pd.read_csv(benchmark_position_path.format(dd), sep='|')
        #position = pd.read_csv(benchmark_position_path.format(dd.replace('-', '')), sep='|', dtype={'date':str})

    obj_position = position.copy(deep=True)[['date', 'secu_code', 'weight']]
    obj_position['weight'] = obj_position['weight']/ obj_position['weight'].sum() * 100
    obj_position.rename(columns={'weight':'obj_weight'}, inplace=True)
    return obj_position


def get_active_position(dd, benchmark_data, actual_data):
    merge_position = pd.merge(actual_data, benchmark_data, how='outer', on='secu_code')
    merge_position = merge_position.fillna(0)
    merge_position['date'] = dd
    merge_position['position'] = merge_position['actual_weight'] - merge_position['obj_weight']
    active_position = merge_position.copy(deep=True)[['date','secu_code','position']]
    # same_ratio_w = (100 - abs(merge_position['active_position']).sum() / 2)
    # same_ratio_num = len(merge_position[(merge_position['actual_weight'] >0)&(merge_position['obj_weight'] >0)]) / len(merge_position) * 100
    return active_position#, same_ratio_w, same_ratio_num


def cal_barra_exposure(dd, cur_position):
    set_name = cur_position.columns.values[-1]
    date_d = datetime.strptime(dd, '%Y-%m-%d') if '-' in dd else datetime.strptime(dd, '%Y%m%d') #调仓日下的exposure
    exp_data = pd.read_csv(barra_path.format(date_d.year, date_d.month, date_d.day, 'exposure.csv'), na_values=['None'])
    factor_col = list(exp_data.columns)[1:] #所有的归因源
    exp_data.rename(columns={'SECU_CODE': 'secu_code'}, inplace=True)
    position_exp = pd.merge(cur_position, exp_data, how='left', on='secu_code')
    exp = position_exp[factor_col].apply(lambda x : position_exp[set_name]/100 * x, axis=0).sum(axis=0) #对个股的暴露赋权求和（整个指数的风险暴露）
    style_exp = exp[0:10]
    #style_exp = style_exp/(style_exp.abs()).sum()*100 #风格因子暴露占比
    industry_exp = exp[11:]
    industry_exp = industry_exp/(industry_exp.abs()).sum()*100 #行业因子暴露占比
    return list(exp), list(style_exp), list(industry_exp), factor_col


def cal_active_reture(dd, exp, cur_position): #和调仓日无关，所有dd适用
    all_factor_ret = pd.DataFrame()
    date_d = datetime.strptime(dd, '%Y%m%d')
    factor_ret = pd.read_csv(barra_path.format(date_d.year, date_d.month, date_d.day, 'factor_return.csv'))
    factor_ret = pd.merge(factor_ret, exp, how='left', on='Factor')
    factor_ret['ret'] = factor_ret['DlyReturn'] * factor_ret['exp'] #因子收益*指数暴露
    factor_ret.index=factor_ret['Factor']
    all_factor_ret = pd.concat((all_factor_ret, factor_ret['ret']), axis=1) #指数收益（分解到因子上）->风格收益
    all_factor_ret.loc['style'] = all_factor_ret[0:10].sum()
    all_factor_ret.loc['industry'] = all_factor_ret[10:].sum()
    sret_data = pd.read_csv(barra_path.format(date_d.year, date_d.month, date_d.day, 'specific_return.csv'))
    cur_position = pd.merge(cur_position, sret_data, how='left', left_on='secu_code', right_on='SECU_CODE')
    cur_position.drop(columns='SECU_CODE',axis=1, inplace=True) 
    cur_sret = (cur_position['position'] /100 * cur_position['SPRET']).sum()
    all_factor_ret.loc['sret'] = cur_sret
    all_factor_ret.loc['date'] = dd
    return all_factor_ret.T.reset_index(drop=True)

def select_industry(df, ret):
    industry_list = df.iloc[-1,1:].sort_values().tail(5).index.tolist()+df.iloc[-1,1:].sort_values().head(5).index.tolist()
    df_top = ret[industry_list]
    return df_top

def main(st, ed, src, index_name_list):
    #same_ratio = []
    all_date_list= get_date_list(st, ed)
    for index_name in index_name_list:
        active_position = pd.DataFrame()
        actual_position = pd.DataFrame()
        all_active_style_exp = pd.DataFrame()
        all_actual_style_exp = pd.DataFrame()
        all_active_rets = pd.DataFrame()
        all_active_industry_exp = pd.DataFrame()
        all_actual_industry_exp = pd.DataFrame()
        for dd in all_date_list:
            print(dd)
            cur_actual_position = get_position_actual(index_name, dd, src) #a_p存储自定义指数的权重
            if src == 'ccx1800':
                path = ccx1800_position_path
            elif src =='raw':
                path = None
            else:
                path = object_position_path
            # path = None
            object_position = get_position_benchmark(dd, src, path) #o_p存储对标指数的权重
            cur_active_position = get_active_position(dd, object_position, cur_actual_position) #指数与基准权重差(相对)
            cur_actual_position.rename(columns={'actual_weight':'position'}, inplace=True)
            active_position = pd.concat([active_position, cur_active_position])
            actual_position = pd.concat([actual_position, cur_actual_position]) 
            #active_position = active_position.append(cur_active_position)
            #same_ratio.append([dd, cur_same_ratio, cur_same_ratio_num]) active里另外两个，已经被删掉
            exp_active_ret, cur_active_style_exp, cur_active_industry_exp,f_c = cal_barra_exposure(dd, cur_active_position) #指数相对风险暴露（单日）
            exp_actual_ret, cur_actual_style_exp, cur_actual_industry_exp, _= cal_barra_exposure(dd, cur_actual_position) #指数绝对风险暴露（单日）
            s_c = f_c[0:10]
            i_c = f_c[11:]
            df_cur_active_style_exp = pd.DataFrame([dd] + cur_active_style_exp, index=['date'] + s_c).T
            df_cur_actual_style_exp = pd.DataFrame([dd] + cur_actual_style_exp, index=['date'] + s_c).T
            df_cur_active_industry_exp = pd.DataFrame([dd] + cur_active_industry_exp, index=['date'] + i_c).T
            df_cur_actual_industry_exp = pd.DataFrame([dd] + cur_actual_industry_exp, index=['date'] + i_c).T
            all_active_style_exp = all_active_style_exp.append(df_cur_active_style_exp)
            all_actual_style_exp = all_actual_style_exp.append(df_cur_actual_style_exp)
            all_active_industry_exp = all_active_industry_exp.append(df_cur_active_industry_exp)
            all_actual_industry_exp = all_actual_industry_exp.append(df_cur_actual_industry_exp)
            cur_active_exp_pre = pd.DataFrame([dd] + exp_active_ret, index=['date'] + f_c).iloc[1:].reset_index() #收益计算需要   
            cur_active_exp_pre.columns = ['Factor', 'exp']
            cur_active_exp_pre['exp']=cur_active_exp_pre['exp'].astype(float)
            cur_actual_exp_pre = pd.DataFrame([dd] + exp_actual_ret, index=['date'] + f_c).iloc[1:].reset_index() #收益计算需要   
            cur_actual_exp_pre.columns = ['Factor', 'exp']    
            cur_actual_exp_pre['exp']=cur_actual_exp_pre['exp'].astype(float)   
            #cur_actual_rets = cal_active_reture(dd, cur_actual_exp_pre, cur_actual_position)
            cur_active_rets = cal_active_reture(dd, cur_active_exp_pre, cur_active_position)
            #返回指数收益分解（当天）
            #all_actual_rets = all_actual_rets.append(cur_actual_rets)
            all_active_rets = all_active_rets.append(cur_active_rets)
        all_active_style_exp = all_active_style_exp.append(pd.DataFrame(['average']+all_active_style_exp.iloc[:,1:].mean().tolist(),index=['date'] + s_c).T)
        all_actual_style_exp = all_actual_style_exp.append(pd.DataFrame(['average']+all_actual_style_exp.iloc[:,1:].mean().tolist(),index=['date'] + s_c).T)
        all_active_industry_exp = all_active_industry_exp.append(pd.DataFrame(['average']+all_active_industry_exp.iloc[:,1:].mean().tolist(),index=['date'] + i_c).T)
        all_actual_industry_exp = all_actual_industry_exp.append(pd.DataFrame(['average']+all_actual_industry_exp.iloc[:,1:].mean().tolist(),index=['date'] + i_c).T)
        all_active_rets['date'] = all_active_rets['date'].apply(lambda x: str(x)[0:4]+'-'+str(x)[4:6]+'-'+str(x)[6:8])
        all_active_rets.set_index('date', inplace=True)
        #all_active_rets.set_index('date', inplace=True)
        all_active_ret_separate = all_active_rets[['style','industry','sret']].cumsum()
        all_active_ret_style = pd.DataFrame(all_active_rets.iloc[:,0:10].sum(),columns=['style_ret'])
        all_active_ret_industry = pd.DataFrame(all_active_rets.iloc[:,11:-3].sum(),columns=['industry_ret'])
        all_active_industry_exp.set_index('date', inplace=True)
        all_actual_industry_exp.set_index('date', inplace=True)
        #industry_active_top = select_industry(all_active_industry_exp, all_active_rets)
        #industry_actual_top = select_industry(all_actual_industry_exp, all_actual_rets)        
        if index_name == 'ESG':
            index_name = '美丽中国ESG指数' 
        else:
            index_name = '碳中和指数'         
        save_path = '/data/public_transfer/liyihan/esg/{}/{}月报'.format(ed, index_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #输出设置：
        #①指数暴露all_actual_exp②相对暴露all_active_exp③收益拆分表（风格+总拆分）all_active_rets
        all_actual_style_exp.to_csv(save_path+'/指数风格因子暴露情况_{}_{}.csv'.format(ed, index_name, st, ed), index=False)
        all_active_style_exp.to_csv(save_path+'/指数风格相对于CCX1800的暴露情况_{}_{}.csv'.format(ed, index_name, st, ed), index=False)
        all_actual_industry_exp.to_csv(save_path+'/指数行业因子暴露情况_{}_{}.csv'.format(ed, index_name, st, ed))
        all_active_industry_exp.to_csv(save_path+'/指数行业相对于CCX1800的暴露情况_{}_{}.csv'.format(ed, index_name, st, ed))
        #(industry_active_top+1).cumprod(axis=0).to_csv(save_path+'/指数行业相对于CCX1800的收益情况_top10_{}_{}.csv'.format(ed, index_name, st, ed))
        all_active_ret_style.to_csv(save_path+'/指数相对于CCX1800的风格因子收益情况_{}_{}.csv'.format(ed, index_name, st, ed))
        all_active_ret_industry.to_csv(save_path+'/指数相对于CCX1800的行业因子收益情况_{}_{}.csv'.format(ed, index_name, st, ed))
        all_active_ret_separate.to_csv(save_path+'/指数相对于CCX1800的收益拆分表{}_{}.csv'.format(ed, index_name, st, ed))
        #(all_active_rets+1).cumprod(axis=0).to_csv(save_path+'/指数相对于CCX1800收益拆分表{}_{}.csv'.format(ed, index_name, st, ed))



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-s", "--start_date", action="store", type="str", dest="start_date", help="examples: 20120105",default='20221125')
    parser.add_option("-e", "--end_date", action="store", type="str", dest="end_date", help="examples: 20210416", default='20221225')
    parser.add_option("-r", "--src", action="store", type="str", dest="src", help="examples: res")

    (options, args) = parser.parse_args()

    # start_date = '20220819'  # options.start_date
    # end_date = '20220819'  # options.end_date

    main(
        st = options.start_date,
        ed = options.end_date,
        src = 'ccx1800',
        index_name_list = ['ESG','Carbon']
        #src= 'tech_res_GrowthFactor+EarningQualityFactor+ResVolFactor+IlliqFactor+LeverageFactor+SizeNLFactor_QUARTERLY_ASSIGNED_WEIGHT_100_cap0.3_tvr0.1_SW'#options.src
    )

