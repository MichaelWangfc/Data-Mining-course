####实验3：读取HIVE数据表，进行业务宽表合成

from pyspark import SparkContext

#引入spark的配置对象
from pyspark import SparkConf 

from pyspark.sql import HiveContext

#获取存储在Hive中数据的主入口
hiveCtx = HiveContext(sc)


#使用（或创建）Hive数据
database_name = 'test_db'
hiveCtx.sql("show databases").show()
# hiveCtx.sql("create database if not exists "+database_name)
hiveCtx.sql("use "+database_name)
hiveCtx.sql("show tables").show()
#使用HiveContext的tables()方法，也可以得到Hive中所有表的相关信息
#hiveCtx.tables().show()


#读取HIVE数据库中的表
table_name = 'personal_data'
personal_data= hiveCtx.sql("select * from "+database_name+' . '+table_name)
#hiveCtx.read.table(table_name).show(5)
personal_data.show(5)


#写入hive表
#1. 得到DataFrame后存入Hive表中，mode有append, overwrite, error, ignore 这4种模式
df.write.saveAsTable('testtable',mode='overwrite')

#2. 使用registerTempTable的方式建立一个临时表，然后使用HiveContext的sql方法生成新的Hive表
#写入hive分区中
hiveContext.sql("insert into table2 partition(date='2015-04-02') select name,col1,col2 from table1")


############Spark 2.x之后

#Spark 2.0 中的SparkSession 为 Hive 特性提供了内嵌的支持, 包括使用 HiveQL 编写查询的能力, 
#访问 Hive UDF,以及从 Hive 表中读取数据的能力.为了使用这些特性, 你不需要去有一个已存在的 Hive 设置.

#创建SparkSession对象，spark API的起始点
from pyspark.sql import SparkSession

#创建SparkSession
spark = SparkSession\
	.builder \
	.appName("Python Spark SQL basic example") \
	.enableHiveSupport()\
    .getOrCreate()
	
#显示spark版本
print(spark.version)


#显示hive的数据库
spark.sql("show databases").show()

#使用Hive数据
database_name = 'test_db'
spark.sql("use "+database_name)
#显示test_db中的表格
spark.sql("show tables").show()

#Spark SQL 的datasets API,
#spark.tables().show() #not work

#读取HIVE数据库中的表 personal_data
table_name = 'personal_data_ds'
personal_data = spark.read.table(table_name)

#查看personal_data表数据
#显示前5行
personal_data.show(5)
#显示表结构
personal_data.printSchema()
#显示表的行数
print(personal_data.count())


#读取HIVE数据库中的表 recharge_data
table_name = 'recharge_data_ds'
recharge_data = spark.read.table(table_name)
#查看recharge_data表数据
#显示前5行
recharge_data.show(5)
#显示表结构
recharge_data.printSchema()
#显示表的行数
print(recharge_data.count())

#读取HIVE数据库中的表 usage_data
table_name = 'usage_data'
usage_data = spark.read.table(table_name)
#查看 usage_data 表数据
#显示前5行
usage_data.show(5)
#显示表结构
usage_data.printSchema()
#显示表的行数
print(usage_data.count())

#首先进行两表join操作
cond = (personal_data.user_id == recharge_data.user_id)
per_rec = personal_data.join(recharge_data,cond,'inner').drop(recharge_data.user_id)


#显示join后的前5行
per_rec.show(5)
#显示表结构
per_rec.printSchema()

#对第3张表进行join操作
cond = (per_rec.user_id == usage_data.user_id)
usage_info_data = per_rec.join(usage_data,cond,'inner')\
.drop(usage_data.user_id).drop(usage_data.service_kind)
#显示join后的前5行
usage_info_data.show(5)
#显示表结构
usage_info_data.printSchema()

#基本统计计算
#每个城市的用户数
usage_info_data.groupBy('location').count().show()

#各市的呼叫时长平均值
usage_info_data.groupBy('location').agg({'call_duration':'mean'}).show()

#各市各个年龄段的用户数量
usage_info_data.groupBy(['location','age']).count()\
.orderBy(['location','count'],ascending =[1,1]).show(100)

#每种业务的使用情况
usage_info_data.groupBy('service').count().show()



#实验2：日志清洗
#引入SparkContext对象
from pyspark import SparkContext

#引入spark的配置对象
from pyspark import SparkConf 

from pyspark.sql import HiveContext


#创建 SparkContext 对象
#sc = SparkContext(appName="pyspark_sql_test")  #create a SparkContext object, which tells Spark how to access a cluster.

#创建 Spark-sql 对象
#Main entry point for Spark SQL functionality.
hiveCtx = HiveContext(sc)


#读取txt文件
rdd_log = sc.textFile('/data/access.log.txt')
rdd_log.take(1)


#创建提取函数
#提取访问的IP地址
def getIP(line):
    #input: line
    #output: 访问IP
    strs = line.split('- -')
    return strs[0].strip()


#提取访问时间
def getTime(line):
    #input: line
    #output: 访问时间,时区
    import re
    import time
	#使用正则表达式，匹配每行中的 [ ] 中的信息，从而提取时间信息
    pattern =re.compile(r'.*?\[(.*?)\].*?')      
    time_info = re.match(pattern,line).group(1).split() 
	# 得到时间信息,如 '18/Sep/2013:06:49:18'
    date_info = time_info[0].strip() 
	# 得到时区信息
    timezone = time_info[1].strip()
	# 从时间字符串转换为timestamp
    timestamp = time.mktime(time.strptime(date_info,"%d/%b/%Y:%H:%M:%S"))
	# 从timestamp转换为自己指定格式的时间字符串
    realtime_str = time.strftime("%Y/%m/%d %H:%M:%S",time.localtime(timestamp)) 
    return realtime_str,timezone
	

#提取请求Url和协议
def getRequestInfo(line):
    #inpout:line
    #output:url和协议，referer,user agent
    import re
    url_protocol,referer,user_agent =  re.findall(r'"(.*?)"',line)
    try:
        [method,url,protocol] = url_protocol.split(' ')
    except:
        [method,url,protocol] = [None,None,None]
    return method,url,protocol,referer,user_agent

#提取访问状态码和请求体的大小
def getStatus(line):
    #input: line
    #output: response code and size
    import re
    status_info  = re.findall(r'.*?(\d+\s+\d+).*?',line)[0].split(' ')
    [response_code,size] = [s.strip() for s in status_info]
    return response_code,size

#完整的提取函数
def extract_fun(line):
	IP =getIP(line)
	realtime_str,timezone = getTime(line)
	method,url,protocol,referer,user_agent = getRequestInfo(line)
	response_code,size = getStatus(line)
	return (IP,realtime_str,timezone,method,url,protocol,\
	referer,user_agent,response_code,size)

#对每一行做map操作，提取每一行的内容，生成新的rdd

line =rdd_log.take(1)[0]
line
rdd_log.map(getRequestInfo).take(3)

rdd_logdata = rdd_log.map(extract_fun)


#生成dataframe的schema
from pyspark.sql.types import StructType,StructField
from pyspark.sql.types import *

schema_log =  StructType([
		  StructField("IP",            	StringType(),True),
		  StructField("TIME",    		StringType(),True),  ##need to be string type
		  StructField("TIMEZONE",		StringType(),True),
		  StructField("METHOD",    		StringType(),True),
		  StructField("URL",      		StringType(),True),  #as datetype
		  StructField("PROTOCOL",       StringType(),True),  #no10
		  StructField("REFERER",        StringType(),True),
		  StructField("USER_AEENT",     StringType(),True),
		  StructField("RESPONSE_CODE", 	StringType(),True),
		  StructField("SIZE",   		StringType(),True)])#as string type
	  

#生成dataframe  
log_df = hiveCtx.createDataFrame(rdd_logdata,schema_log)


#6.数据分析
#IP地址分组统计
group_byIP = log_df.groupby(['IP']).agg({'Time':'count'})
group_byIP.show(5)


#IP地址统计排序
group_byIP.sort('count(Time)',ascending = False).show(5)



#根据 STATUS_CODE 字段进行过滤
useful_log = log_df.filter(log_df.RESPONSE_CODE<400)
useful_log.show(5)



#dataframe储存到hive-table
database_name = 'wang_test'
table_name = 'web_log'


hiveCtx.sql("create database if not exists "+database_name)
hiveCtx.sql("use "+database_name)
log_df.write.saveAsTable(table_name,mode = 'overwrite')




####实验3：读取结构化数据集


#引入SparkContext对象
from pyspark import SparkContext
#引入spark的配置对象
from pyspark import SparkConf 

from pyspark.sql import HiveContext


#创建 Spark-sql 对象
#Main entry point for Spark SQL functionality.
hiveCtx = HiveContext(sc)

#读取hdfs上的文件数据
rdd_csv= sc.textFile('/data/telecom.csv')

#对text文件做map和过滤
rdd_map= rdd_csv.map(lambda row:row.split(',')).filter(lambda row:len(row)>6)


#生成结构化的数据
筛选列式为7的数据为每一行
from string import strip
column_num = 7
rdd_data = rdd_map.map(lambda row:[strip(row[i]) for i in range(column_num)])


#生成表格的schema
from pyspark.sql.types import StructType,StructField
from pyspark.sql.types import *


schema_telecom= StructType([
		  StructField("NET_DRUATION",            	StringType(),True),
		  StructField("SERVICE_PRICE",    		StringType(),True),  ##need to be string type
		  StructField("MONTHLY_FLUX",			StringType(),True),
		  StructField("MONTHLY_FARE",    	StringType(),True),
		  StructField("MONTHLY_CALLING_DURATION",      	StringType(),True),  #as datetype
		  StructField("ARREARAGE",       StringType(),True),  #no10
		  StructField("ARREARAGE_MONTHS",StringType(),True)])#as string type

#生成dataframe  
df = hiveCtx.createDataFrame(rdd_data,schema_telecom)
#显示dataframe的前几行
df.show()



# 任务2： 数据探索
#显示表结构
df.printSchema()

#显示行数
df.count()


#转换数据类型
telecom_df= df.select(
df.NET_DRUATION.astype('int'),df.SERVICE_PRICE.astype('double'),
df.MONTHLY_FLUX.astype('double'),df.MONTHLY_FARE.astype('double'),
df.MONTHLY_CALLING_DURATION.astype('double'),df.ARREARAGE.astype('double'),
df.ARREARAGE_MONTHS.astype('int'))

telecom_df.printSchema()


#dataframe储存到hive-table
database_name = 'wang_test'
table_name = 'telecom'

hiveCtx.sql("create database if not exists "+database_name)
hiveCtx.sql("use "+ database_name)
log_df.write.saveAsTable(table_name,mode = 'overwrite')






