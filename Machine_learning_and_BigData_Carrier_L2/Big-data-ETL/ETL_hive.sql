#source
#beeline


create database if not exists wang_test;



create table  if not exists wang_test.carrier_svm
(user_id int,
service_kind string,
call_duration double,
called_duration double,
in_package_flux double,
out_package_flux double,
monthly_online_duration double,
net_duration double,
last_recharge_value double,
total_recharge_value double,
total_recharge_count int,
balanced double,
contractuser_flag int
)ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
stored as textfile;

#权限问题？
LOAD DATA LOCAL INPATH '/root/data_svm_english.csv' OVERWRITE INTO TABLE carrier_svm;

hadoop fs -put /root/data_svm_english.csv /data
LOAD DATA INPATH '/data/data_svm_english.csv' OVERWRITE INTO TABLE carrier_svm;



###########################
#业务宽表生成

#上传数据到HDFS
hadoop fs -mkdir /data/
hadoop fs -put /root/ /data


show databases;
create database test_db;
use test_db;

show tables;


#生成 personal_data(注意csv文件的不需要列名)
create table personal_data
(USER_ID int,SERVICE string,LOCATION string,AGE int,CAREER string,CREATEDATE string)
row format delimited fields terminated by ',' stored as textfile;
LOAD DATA  INPATH '/data/personal_data.csv' OVERWRITE INTO TABLE personal_data;
select * from personal_data limit 5;

#生成 recharge_data
create table recharge_data
(USER_ID int,LAST_RECHARGE_VALUE int,TOTAL_RECHARGE_VALUE int,
TOTAL_RECHARGE_COUNT int,BALANCED int)
row format delimited fields terminated by ',' stored as textfile;
LOAD DATA  INPATH '/data/recharge_data.csv' OVERWRITE INTO TABLE recharge_data;
select * from recharge_data limit 5;


#生成 usage_data
create table usage_data
(USER_ID int,SERVICE_KIND string,CALL_DURATION int,CALLED_DURATION int,IN_PACKAGE_FLUX int,OUT_PACKAGE_FLUX int,
MONTHLY_ONLINE_DURATION int,NET_DURATION int)
row format delimited fields terminated by ',' stored as textfile;

LOAD DATA  INPATH '/data/usage_data.csv' OVERWRITE INTO TABLE usage_data;
select * from usage_data limit 5;


#两表右连接
create table two_data
(USER_ID int,SERVICE string,LOCATION string,AGE int,CAREER string,CREATEDATE string,
LAST_RECHARGE_VALUE int,TOTAL_RECHARGE_VALUE int,
TOTAL_RECHARGE_COUNT int,BALANCED int)
row format delimited fields terminated by ',' stored as textfile;

create table usage_info_data as
select 
a.USER_ID,a.SERVICE,a.LOCATION,a.AGE,a.CAREER,a.CREATEDATE,
b.LAST_RECHARGE_VALUE,b.TOTAL_RECHARGE_VALUE,b.TOTAL_RECHARGE_COUNT,b.BALANCED,
c.CALL_DURATION,c.CALLED_DURATION,c.IN_PACKAGE_FLUX,c.OUT_PACKAGE_FLUX,
c.MONTHLY_ONLINE_DURATION,c.NET_DURATION
from personal_data a right join recharge_data b 
on a.USER_ID=b.USER_ID
left join usage_data c
on b.USER_ID=c.USER_ID;

select * from usage_info_data limit 5;

desc usage_info_data;

select LOCATION,count(*) from 
usage_info_data
group by LOCATION;

#各市的呼叫时长平均值
select LOCATION,avg(CALL_DURATION) from 
usage_info_data
group by LOCATION;