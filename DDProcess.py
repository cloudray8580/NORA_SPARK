import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Cannot run multiple spark context at once
# import findspark
# findspark.init() # this must be executed before the below import
# from pyspark import SparkContext, SparkConf
# from pyspark.sql import SQLContext
# from pyspark.sql import SparkSession
# from pyspark import SparkFiles

def dump_data(parameters):
	pid_data_dict, column_names, hdfs_path = parameters
	for pid in list(pid_data_dict.keys()):
		path = hdfs_path + 'partition_' + str(pid)+'.parquet'
		pdf = pd.DataFrame(pid_data_dict[pid], columns=column_names)
		adf = pa.Table.from_pandas(pdf)
		fs = pa.hdfs.connect('localhost',9000)
		with fs.open(path, 'wb') as fw:
			pq.write_table(adf, fw)
		# df = sqlContext.createDataFrame(pdf)
		# df.write.mode('append').parquet(path)
		print('exit dumping process')