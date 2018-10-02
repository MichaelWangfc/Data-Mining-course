mysql -h 114.116.72.230 -P 3306 -u root -p



GRANT ALL PRIVILEGES ON *.* TO 'myuser'@'%' IDENTIFIED BY '123456';
WITH GRANT OPTION;
FLUSH PRIVILEGES;

show databases;
create database test_db;
use test_db;
create table personal_data
	(
		USER_ID int,
		SERVICE char(8),
		LOCATION char(16) ,
		AGE tinyint ,
		CAREER char(16) ,
		CREATEDATE datetime
	);


INSERT INTO test_db.personal_data 
(USER_ID,SERVICE,LOCATION,AGE,CAREER,CREATEDATE)
VALUES 
('38826','2G','576','48','IT_EMPLOEE','2015-10-16 09:45:33')


	
LOAD DATA INFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/personal_data.csv' 
INTO TABLE test_db.personal_data 
FIELDS TERMINATED BY ',';

select * from personal_data limit 5;



# 在某台DB上准备运行一个SQL语句，就是用SELECT INTO OUTFILE把查询结果写入到文件的时候提示以下信息：
# The MySQL server is running with the --secure-file-priv option so it cannot execute this statement
# 出现这个问题的原因是因为启动MySQL的时候使用了--secure-file-priv这个参数，这个参数的主要目的就是限制LOAD DATA INFILE或者SELECT INTO OUTFILE之类文件的目录位置，我们可以使用


#测试MySQL连接 
$sqoop list-databases 
--connect jdbc:mysql://192.168.0.116:3306  --username root --password Huawei@123456