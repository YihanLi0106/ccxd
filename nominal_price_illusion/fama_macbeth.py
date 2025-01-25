import pandas as pd
import numpy as np
from linearmodels import FamaMacBeth, PooledOLS
import statsmodels.api as sm

monthly_data = pd.read_csv('/home/liyihan/prepared_data.csv', sep='|')
monthly_data = monthly_data[['SECU_CODE', 'DT', 'exp_RETURN', 'PRICE', 'REVISION', 'beta', 'lnSIZE', 'lnBM', 'Ag', 'ROE', 'sum_RETURN', 'lag_RETURN', 'VOLATILITY', 'Age', 'INTANGIBLESIZE', 'INDUSTRY']]
monthly_data.sort_values(by=['DT'], inplace=True)
monthly_data = monthly_data.set_index(['SECU_CODE','DT'])

print(monthly_data)

def time_reg(x, x_cols):
    # input()
    if x.shape[0] < 3 or x.iloc[0].at['ROE'] == x.iloc[-1].at['ROE']:
        return np.nan
    # print(x[x_cols], x[x_cols].dtypes, sep='\n')
    add_exog= sm.add_constant(x[x_cols])
    ols = PooledOLS(dependent=x['exp_RETURN'], exog=x[x_cols], check_rank=False)
    model = ols.fit()
    params = model.params
    # print(model.summary)
    # print(params)
    if x_cols == ['PRICE']:
        return params['PRICE']
    return params.to_dict()

def calc_col_1():
    cols = ['PRICE']
    beta = monthly_data.groupby('SECU_CODE').apply(lambda x: time_reg(x, cols))
    beta.dropna(inplace=True)
    beta = beta.to_frame(name='beta')
    monthly_data.reset_index(inplace=True)
    betafactor_data = monthly_data.merge(beta,  how="left", on='SECU_CODE')
    betafactor_data.set_index(['SECU_CODE', 'DT'], inplace=True)

    fm = FamaMacBeth(dependent=betafactor_data['exp_RETURN'], exog=sm.add_constant(betafactor_data['beta']))

    # Newey-West adjust
    res_fm = fm.fit(cov_type='kernel', debiased=False, bandwidth=4)
    print(res_fm.summary)

def calc_col_2():
    cols = ['PRICE', 'REVISION', 'beta', 'lnSIZE', 'lnBM', 'Ag', 'ROE', 'sum_RETURN', 'lag_RETURN']
    # cols = ['PRICE', 'REVISION']
    res = monthly_data.groupby('SECU_CODE').apply(lambda x: time_reg(x, cols))
    
    res.dropna(inplace=True)
    res = pd.DataFrame({'SECU_CODE': res.index, 'dict': res.values})
    print(res)
    for col in cols:
        res[col] = res['dict'][col]
    print(res)
    monthly_data.reset_index(inplace=True)
    betafactor_data = monthly_data.merge(res,  how="left", on='SECU_CODE')
    betafactor_data.set_index(['SECU_CODE', 'DT'], inplace=True)
    input()
    fm = FamaMacBeth(dependent=betafactor_data['exp_RETURN'], exog=sm.add_constant(betafactor_data['beta']))

    # Newey-West adjust
    res_fm = fm.fit(cov_type='kernel', debiased=False, bandwidth=4)
    print(res_fm.summary)

if __name__ == '__main__':
    calc_col_1()
    calc_col_2()