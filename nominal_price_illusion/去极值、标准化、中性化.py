
# coding: utf-8

# In[ ]:

import numpy as np
#去极值
pe = DataAPI.MktStockFactorsOneDayGet(secID=set_universe('HS300'),tradeDate=u"20190815",field=u"secID,tradeDate,PE",pandas="1").set_index('secID')
pe['winsorized PE'] = winsorize(pe['PE'], win_type='NormDistDraw', n_draw=5)
pe.head()


# In[ ]:

#标准化
pe['standardize PE'] = standardize(pe['winsorized PE'])
pe.head()


# In[ ]:

import matplotlib.pyplot as plt
plt.hist(pe['standardize PE'])
plt.show()


# In[ ]:

#中性化
pe['neutralize PE'] = neutralize(pe['standardize PE'], '20190815', industry_type='SW1', exclude_style_list=[])
pe.head()

