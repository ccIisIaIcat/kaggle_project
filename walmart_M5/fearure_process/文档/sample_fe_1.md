数据与处理相关——第一部分

1、自定义一些函数用于数据压缩和其他
    get_memory_usage() 获取当前进程的内存占用
    sizeof_fmt(num, suffix='B') 比特数据的结构化输出
    reduce_mem_usage(df, verbose=True) 根据数据的最大值选取更合适的数据类型进行储存，节约内存
    merge_by_concat(df1, df2, merge_on) 新的连接方式防止数据类型丢失

2、sales_train_validation表的细节
    用于标注的key为：['id','item_id','dept_id','cat_id','store_id','state_id']，分别代表产品id，商品id，部门id，种类id，商店id，和州的id

    针对key，d和values将dataframe进行melt

3、创建一个预测表格
    集成train_df的key列，然后去重，添加日期列（其值为训练集最后一天日期序列顺序加一）
    把预测表顺序添加到grid_df后面

4、把key值中的string类型改为category类型可以有效节约内存

5、把grid_df中wm_yr_wk不存在的列删除

6、把wm_yr_wk和最小值做差，保存为int16

