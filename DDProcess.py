import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

# Test4-2: merge all the batches into 1 single partition
def merge_parquets(parameters):
	fs = pa.hdfs.connect('localhost',9000)
	batches, pids, hdfs_path = parameters
	for pid in pids:
		parquets = []
		for batch in range(batches):
			path = hdfs_path + str(batch) + '/partition_' + str(pid)+'.parquet'
			try:
				par = pq.read_table(path)
				parquets.append(par)
			except:
				continue
		merged_parquet = pa.concat_tables(parquets)
		merge_path = hdfs_path + 'merged/partition_' + str(pid)+'.parquet'
		fw = fs.open(merge_path, 'wb')
		pq.write_table(merged_parquet, fw)
		fw.close()
	print('exit merge process')


# Test4-1: write to different batches, finaly merge them
def dump_data(parameters):
	batch, pid_data_dict, column_names, hdfs_path = parameters
	fs = pa.hdfs.connect('localhost',9000)
	for pid in list(pid_data_dict.keys()):
		path = hdfs_path + str(batch) + '/partition_' + str(pid)+'.parquet'
		pdf = pd.DataFrame(pid_data_dict[pid], columns=column_names)
		adf = pa.Table.from_pandas(pdf)
		fw = fs.open(path, 'wb')
		pq.write_table(adf, fw)
		fw.close()
	print('exit dumping process')

# Test3: Using pyarrow to append new dataframe to parquet, which is not implemented in arrow
# def dump_data(parameters):
# 	pid_data_dict, column_names, hdfs_path = parameters
# 	fs = pa.hdfs.connect('localhost',9000)
# 	for pid in list(pid_data_dict.keys()):
# 		path = hdfs_path + 'partition_' + str(pid)+'.parquet'
# 		pdf = pd.DataFrame(pid_data_dict[pid], columns=column_names)
# 		adf = pa.Table.from_pandas(pdf)
# 		fw = None
# 		try:
# 			fw = fs.open(path, 'ab') # this is not implemented
# 		except:
# 			fw = fs.open(path, 'wb')
# 		pq.write_table(adf, fw)
# 		fw.close()
# 	print('exit dumping process')

# Test2: create spark context, Cannot run multiple spark context at once
# import findspark
# findspark.init() # this must be executed before the below import
# from pyspark import SparkContext, SparkConf
# from pyspark.sql import SQLContext
# from pyspark.sql import SparkSession
# from pyspark import SparkFiles

# Test1: pass Spark context, which is not allowed
# def dump_data(parameters):
# 	spark, pid_data_dict, column_names, hdfs_path = parameters
# 	#fs = pa.hdfs.connect('localhost',9000)
# 	for pid in list(pid_data_dict.keys()):
# 		path = hdfs_path + 'partition_' + str(pid)+'.parquet'
# 		pdf = pd.DataFrame(pid_data_dict[pid], columns=column_names)
# 		df = spark.createDataFrame(pdf)
# 		df.write.mode('append').parquet(path)
# 	print('exit dumping process')