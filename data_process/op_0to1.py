# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


import pandas as pd
import numpy as np
from collections import Counter

import natsort
datas_df = pd.read_csv('../Datas/Data_original.csv', encoding="GBK", header=0, low_memory=False)



data_np = np.array(datas_df)
data_np = data_np[1:,:]  #all shape
x_num = data_np.shape[0]
print('num of x:',x_num)
l=1
need_to_del = []
for i in range(0,datas_df.shape[1]):

    data_temp = data_np[:,i]
    dic = Counter(data_temp)
    if np.nan in dic.keys():
        dic.pop(np.nan)
    keys_number = len(dic.keys())

    keys = dic.keys()
    keys_value = dic.values()
    # print(keys)
    # print(keys_value)
    bfb = sum(keys_value)/x_num
    # print(bfb)

    if (bfb == 1 and len(keys) == 1) or len(keys) == 0:
        # if l % 5 != 0:
        #     print('del',i+1,end ='\t')
        # else:
        #     print('del', i + 1)
        # l= l+1
        need_to_del.append(i)
        # if '-99' in keys:
    #     print('yes',i+1)

print(need_to_del)

df2 = datas_df.drop(datas_df.iloc[:, need_to_del], axis = 1)
df2.to_csv('./Datas/Data_picked_1.csv',encoding='GBK',index = False)




