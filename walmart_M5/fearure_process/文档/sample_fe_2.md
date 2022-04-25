对于价格信息的一些处理
    价格信息表sell_prices的key为store_id,item_id,wm_yr_wk

1、获取一些关于价格的特征数据
    在原dataframe下追加平均数，最大值，最小值，方差等

2、对价格本身做正则化处理

3、获取同一家商店中，同一商品的不同价格数和同一价格的不同商品数

4、为获取动量信息，通过wm_yr_wk把calender中的year和week信息连接起来
    获得价格每日增量（注意transform(lambda x: x.shift(1))操作）
    每日价格和周平均价格的比值
    每日价格和月平均价格的比值

5、把这些信息根据'store_id','item_id','wm_yr_wk'三个键连接到grid_df上
    操作细节：先保存grid_df原列表键，然后merge，之后取得新增的列，把grid_df保存为Main_index和新增列（此时该表不在保存销量信息）

