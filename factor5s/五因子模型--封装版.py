
# -*- coding: utf-8 -*-
#@author：筑基期摸鱼大师

# 导入包
import pandas as pd
import numpy as np
import statsmodels.api as sm


def main():

    # 一、预处理所有数据
    # 导入数据
    # 1.沪深300收盘价数据集及其他备用数据

    data_hs300 = pd.read_excel('沪深300成分股收盘价数据.xlsx')
    data_hs300_info = pd.read_excel('沪深300因子数据.xlsx')
    data_hs300index,data_hs300_price,data_hs300_info,june_last_trading_days = Data_Preprocessing(data_hs300,data_hs300_info)

    # 二、遍历每个调仓日计算每周的因子
    data_factor_all = pd.DataFrame()
    for k, june_last_trading_day in enumerate(june_last_trading_days[:-1]):    # 不要最后一个调仓日,因为后面没有数据了。
        data_week_factor = Getting_Factor_for_EachYear(data_hs300_info,june_last_trading_day,june_last_trading_days,data_hs300_price,k)
        data_factor_all = pd.concat([data_factor_all, data_week_factor], ignore_index=True)


    # 合并所有数据，计算市场收益率-无风险利率的市场因子
    # 无风险收益率：沪深300收益率数据，无风险利率：一年期国债收益率数据，原本想找国开债一年期到期收益率,但是下载不了，所以取个固定值1.8%

    data_hs300index['rf'] = float(0.018)/252 # 252个交易日
    data_hs300index['ret_rf'] = data_hs300index['ret'] - data_hs300index['rf']

    final_factor_data = pd.merge(left=data_factor_all,right=data_hs300index.loc[:,['time','ret_rf','rf']],on='time')

    # 将因子数合并到股票每周收盘价中
    final_factor_data = pd.merge(left=data_hs300_price.loc[:, ['time', 'thscode', 'ret']], right=final_factor_data,
                                 on='time')

    final_factor_data['stock_ret_rf'] = final_factor_data['ret'] - final_factor_data['rf']

    # 三、利用OLS拟合每个股票的参数,获取Aphla

    total_ols_df = final_factor_data.groupby('thscode').apply(lambda x: Get_OLS_regression_model(x))
    total_ols_df = total_ols_df.reset_index().drop(columns=['thscode', 'level_1'])

    total_ols_df = total_ols_df.sort_values(by=['ols截距'], ascending=False)

    # 结果写入excel
    total_ols_df.to_excel('五因子模型参数结果.xlsx',index=False)


# 一、预处理数据
def Data_Preprocessing(data_hs300,data_hs300_info):

    # 1.沪深300成分股收盘价数据的数据处理
    # 将 'close' 列转换为浮点数
    data_hs300['close'] = pd.to_numeric(data_hs300['close'], errors='coerce')

    # 按照股票名称、日期排序
    data_hs300 = data_hs300.sort_values(by=['thscode', 'time'])

    # 计算沪深300成分股每日收益率数据
    data_hs300['ret'] = data_hs300.groupby('thscode')['close'].transform(lambda x: x / x.shift(1) - 1)

    # 沪深300数据
    data_hs300index = data_hs300[data_hs300['thscode'] == 'sh300']
    data_hs300_price = data_hs300[~(data_hs300['thscode'] == 'sh300')]

    # 2.细化计算财务指标,沪深300因子数据的数据处理
    # 账面市值比
    data_hs300_info['BM'] = 1 / data_hs300_info['ths_pb_csi_release_stock']  # 账面市值比为市净率的倒数

    # 盈利利润率
    data_hs300_info['rate'] = data_hs300_info['ths_op_stock'] / data_hs300_info['ths_total_owner_equity_stock']

    # 投资净增长
    # 按照日期排序
    data_hs300_info = data_hs300_info.sort_values(by=['thscode', 'time'])
    data_hs300_info['asset'] = data_hs300_info.groupby('thscode')['ths_total_assets_stock'].pct_change()

    # 3.计算每年调仓日,获取所有年份的调仓日
    date_df = data_hs300_price[data_hs300_price['thscode'] == '000001.SZ']

    # 将时间戳转换为时间格式
    date_df['time'] = pd.to_datetime(date_df['time'])

    # 每年6月份的最后一个交易日
    date_df_6 = date_df[date_df['time'].dt.month == 6]
    june_last_trading_days = date_df_6.groupby(date_df_6['time'].dt.year).apply(lambda x: x['time'].max())
    june_last_trading_days = june_last_trading_days.tolist()
    june_last_trading_days = june_last_trading_days[1:]  # 不要第一个调仓日,因为财务指标数少了一年

    return data_hs300index,data_hs300_price,data_hs300_info,june_last_trading_days


# 二、计算每年的因子数
def Getting_Factor_for_EachYear(data_hs300_info,june_last_trading_day,june_last_trading_days,data_hs300_price,k):

    # 取数计算,取t-1年12月的年报作为财务指标计算期
    # 现在以2021年的换仓日：2021-06-25为准,利用2020年的年报数据进行分组
    data_year_info = data_hs300_info[(data_hs300_info['time'].dt.month == 12) &
                                     (data_hs300_info['time'].dt.year == june_last_trading_day.year - 1)]

    # 计算财务指标数据,按照分组规则进行每组
    def get_portfolio(data_info):

        # 按照日期排序
        data_info = data_info.sort_values(by=['thscode', 'time'])

        # 按照市值分成两组，大组B，小组S
        data_info['B_S'] = np.where(data_info['ths_market_value_stock'] >= data_info['ths_market_value_stock'].median(),
                                    'B', 'S')

        # 按照账面市值比分成三组，大组H，中组M，小组L
        data_info['H_M_L'] = np.where(data_info['BM'] <= data_info['BM'].quantile(0.3), 'L',
                                      np.where(data_info['BM'] <= data_info['BM'].quantile(0.7), 'M', 'H'))

        # 按照营业利润率分成三组，大组R，中组N，小组W
        data_info['R_Z_W'] = np.where(data_info['rate'] <= data_info['rate'].quantile(0.3), 'W',
                                      np.where(data_info['rate'] <= data_info['rate'].quantile(0.7), 'Z', 'R'))

        # 按照投资风格(t-1期总资产-t-2期总资产),大组C，中组N，小组A
        data_info['C_N_A'] = np.where(data_info['asset'] <= data_info['asset'].quantile(0.3), 'C',
                                      np.where(data_info['asset'] <= data_info['asset'].quantile(0.7), 'N', 'A'))

        return data_info

    data_year_info = get_portfolio(data_year_info)

    # 返回每年每个组合对应的持仓股票组合
    # 规模维度分别与其他三个维度进行交叉得到18种组合

    portfolio_name = ['SH', 'SM', 'SL', 'BH', 'BM', 'BL', 'SR', 'SZ', 'SW', 'BR', 'BZ', 'BW', 'SC', 'SN', 'SA', 'BC',
                      'BN', 'BA']
    portfolio_name_codelist = []
    for each in portfolio_name:

        # 根据组合名称选择组合对应的股票数据
        selected_columns = []
        for j in [char for char in each]:

            # 看选择那一列数据，如果有B或者S，则选B_S列,依次类推。
            if 'B' == j or 'S' == j:
                selected_column = 'B_S'
            elif 'H' == j or 'M' == j or 'L' == j:
                selected_column = 'H_M_L'
            elif 'R' == j or 'Z' == j or 'W' == j:
                selected_column = 'R_Z_W'
            else:
                selected_column = 'C_N_A'

            selected_columns.append(selected_column)

        # 根据上述规则选择数据,如SH,选择B_S列中值为S且H_M_L列中值为H的数据，然后再选择股票编码列
        one_portfolio_name = data_year_info[(data_year_info[selected_columns[0]] == each[0])
                                            & (data_year_info[selected_columns[1]] == each[1])]['thscode'].tolist()

        portfolio_name_codelist.append(one_portfolio_name)

    # 所有组合组成一个字典,键为组合名字，值为该组合对应的股票代码；如SH:[123.SZ,234.SH]
    portfolio_codelists_dict = dict(zip(portfolio_name, portfolio_name_codelist))

    # 根据组合股票计算每年每周的因子数
    # 选取对应的收益率数据，当年7月-次年6月的交易数据
    from datetime import datetime, timedelta

    start_day = june_last_trading_day + timedelta(days=1)  # 开始日期
    end_day = june_last_trading_days[k + 1]  # 第二年换仓日
    data_year_hs300_price = data_hs300_price[
        (data_hs300_price['time'] >= start_day) & (data_hs300_price['time'] <= end_day)]

    # 将空值和0填充为0
    data_year_hs300_price = data_year_hs300_price.fillna(0)

    # 先计算每周各组的市值加权收益率,再计算因子值【每周都有一个因子值】

    def get_group_ret(data_year_hs300_price, portfolio_codelists_dict, portfolio_name):

        # 计算每周所有组合的市值加权平均收益率
        def get_factor(companies):
            data = data_year_hs300_price[data_year_hs300_price['thscode'].isin(companies)]
            weighted_ret = np.average(data['ret'], weights=data['stock_market'])
            return float(weighted_ret)

        portfolio_weighted_rets = []
        for i in portfolio_name:
            companies = portfolio_codelists_dict[i]
            weighted_ret = get_factor(companies)
            portfolio_weighted_rets.append(weighted_ret)

        rets_dict = dict(zip(portfolio_name, portfolio_weighted_rets))

        # 开始计算因子
        # 规模因子：规模与估值两维度分组
        SMB1 = np.mean([rets_dict['SH'], rets_dict['SM'], rets_dict['SL']]) - np.mean(
            [rets_dict['BH'], rets_dict['BM'], rets_dict['BL']])

        # 规模因子：规模与盈利两维度分组
        SMB2 = np.mean([rets_dict['SR'], rets_dict['SZ'], rets_dict['SW']]) - np.mean(
            [rets_dict['BR'], rets_dict['BZ'], rets_dict['BW']])

        # 规模因子：规模与投资两维度分组
        SMB3 = np.mean([rets_dict['SC'], rets_dict['SN'], rets_dict['SA']]) - np.mean(
            [rets_dict['BC'], rets_dict['BN'], rets_dict['BA']])

        # 规模因子
        SMB = np.mean([SMB1, SMB2, SMB3])

        # 估值因子：规模与估值两维度分组
        HML = np.mean([rets_dict['SH'], rets_dict['BH']]) - np.mean([rets_dict['SL'], rets_dict['BL']])

        # 盈利因子：规模与估值两维度分组
        RMW = np.mean([rets_dict['SR'], rets_dict['BR']]) - np.mean([rets_dict['SW'], rets_dict['BW']])

        # 投资因子：规模与投资两维度分组
        CMA = np.mean([rets_dict['SC'], rets_dict['BC']]) - np.mean([rets_dict['SA'], rets_dict['BA']])

        factor_df = pd.DataFrame(dict({'SMB': [SMB], 'HML': [HML], 'RMW': [RMW], 'CMA': [CMA]}))

        return factor_df

    data_week_factor = data_year_hs300_price.groupby('time').apply(
        lambda x: get_group_ret(x, portfolio_codelists_dict, portfolio_name))
    data_week_factor = data_week_factor.reset_index(drop=False).drop(columns=['level_1'])

    return data_week_factor

# 三、拟合OLS参数
def Get_OLS_regression_model(group_df):

    Y = np.array(group_df.loc[:, ['stock_ret_rf']])
    X = np.array(group_df.loc[:, ['ret_rf', 'SMB', 'HML', 'RMW', 'CMA']])

    # 添加常数项
    X_with_intercept = sm.add_constant(X)

    # 创建模型
    model = sm.OLS(Y, X_with_intercept)

    # 拟合模型
    results = model.fit()

    # 获取模型参数
    slope = results.params[1]
    intercept = results.params[0]

    one_ols_df = pd.DataFrame(
        dict({'company': [group_df['thscode'].tolist()[0]], 'ols斜率': [slope], 'ols截距': [intercept]}))

    return one_ols_df


if __name__ == '__main__':
    main()
