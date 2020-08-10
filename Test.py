# import findspark
# findspark.init() # this must be executed before the below import
# from pyspark.sql import SparkSession

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from DDProcess import *
from multiprocessing import Pool

# spark = SparkSession.builder.appName("Python Spark SQL basic example").getOrCreate()


# pdf1 = pd.DataFrame(np.random.randint(6, size=(10,5)))
# pdf2 = pd.DataFrame(np.random.randint(6, size=(10,5)))

# data1 = np.random.randint(6, size=(10,5)).tolist()
# data2 = np.random.randint(6, size=(10,5)).tolist()
# col_names = ['A', 'B', 'C', 'D', 'E']

# data = [{1:data1}, {2:data2}]
# path = 'hdfs://localhost:9000/user/cloudray/NORA/'

# paras = [[spark, data[k], col_names, path] for k in range(2)]
# pool.map(dump_data, [para for para in paras])

# pool = Pool(processes = 2)
# paras = [[3, [k], path] for k in range(2)]
# pool.map(merge_parquets, [para for para in paras])

# # ================================================
# fs = pa.hdfs.connect('localhost',9000)
# df = fs.read_parquet('hdfs://localhost:9000/user/cloudray/NORA/partition_1.parquet')
# pdf = df.to_pandas()


# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
# import numpy as np

# fs = pa.hdfs.connect('localhost',9000)
# fw = fs.open('hdfs://localhost:9000/user/cloudray/Test/test.parquet', 'wb')

# data = np.random.randint(6, size=(10,5)).tolist()
# col_names = ['A', 'B', 'C', 'D', 'E']

# pdf = pd.DataFrame(data, columns=col_names)
# adf = pa.Table.from_pandas(pdf)

# pq.write_table(adf, fw, version='2.0')
# fw.close()

# fw = fs.open('hdfs://localhost:9000/user/cloudray/Test/test.parquet', 'ab')
# pq.write_table(adf, fw, version='2.0')
# fw.close()

# fw = fs.open('hdfs://localhost:9000/user/cloudray/Test/test.parquet', 'ab')
# fw1 = fs.open('hdfs://localhost:9000/user/cloudray/NORA/test.parquet', 'wb')
# fw2 = fs.open('hdfs://localhost:9000/user/cloudray/NORA/test.parquet', 'wb')

# from pyarrow import csv
# fn = 'hdfs://localhost:9000/user/cloudray/NORA/grades.csv'
# table = csv.read_csv(fn)

# path = 'hdfs://localhost:9000/user/cloudray/NORA/0/partition_1.parquet'
# par = pq.read_table(path)


# now lets read the csv files and dump it into 2 parquet collections,
# the first has only 1 parquet, the second has 2 parquets

data = np.genfromtxt('/home/cloudray/Downloads/TPCH_12M_8Field.csv', delimiter=',')

len_data = len(data) // 2
pdf1 = pd.DataFrame(data[0:len_data])
pdf2 = pd.DataFrame(data[len_data:])

adf1 = pa.Table.from_pandas(pdf1)
adf2 = pa.Table.from_pandas(pdf2)

pdf = pd.DataFrame(data)
adf = pa.Table.from_pandas(pdf)

fs = pa.hdfs.connect('localhost',9000)


fw1 = fs.open('hdfs://localhost:9000/user/cloudray/NORA_Test/1.parquet', 'wb')
fw2 = fs.open('hdfs://localhost:9000/user/cloudray/NORA_Test/2.parquet', 'wb')
pq.write_table(adf1, fw1)
pq.write_table(adf2, fw2)

fw = fs.open('hdfs://localhost:9000/user/cloudray/QdTree_Test/1.parquet', 'wb')
pq.write_table(adf, fw)


# CSV
# -- One or more wildcard:
#        .../Downloads20*/*.csv
# --  braces and brackets   
#        .../Downloads201[1-5]/book.csv
#        .../Downloads201{11,15,19,99}/book.csv