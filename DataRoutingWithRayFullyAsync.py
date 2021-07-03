#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init() # this must be executed before the below import


# In[2]:


from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark import SparkFiles


# In[3]:


import ray
import time
import rtree
from rtree import index
import pandas as pd
import numpy as np
from numpy import genfromtxt
import threading
import pyarrow as pa
import pyarrow.parquet as pq


# In[4]:


conf = SparkConf().setAll([("spark.executor.memory", "8g"),("spark.driver.memory","8g"),
                           ("spark.memory.offHeap.enabled",True),("spark.memory.offHeap.size","8g")])

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


# In[5]:


def kdnode_2_border(kdnode):
    lower = [domain[0] for domain in kdnode[0]]
    upper = [domain[1] for domain in kdnode[0]]
    border = tuple(lower + upper) # non interleave
    return border

def load_partitions_from_file(path):
    stretched_kdnodes = genfromtxt(path, delimiter=',')
    num_dims = int(stretched_kdnodes[0,0])
    kdnodes = []
    for i in range(len(stretched_kdnodes)):
        domains = [ [stretched_kdnodes[i,k+1],stretched_kdnodes[i,1+num_dims+k]] for k in range(num_dims) ]
        row = [domains]
        row.append(stretched_kdnodes[i,2*num_dims+1])
        # to be compatible with qd-tree's partition, that do not have the last 4 attributes
        if len(stretched_kdnodes[i]) > 2*num_dims+2:
            row.append(stretched_kdnodes[i,-4])
            row.append(stretched_kdnodes[i,-3])
            row.append(stretched_kdnodes[i,-2])
            row.append(stretched_kdnodes[i,-1])
        kdnodes.append(row)
    return kdnodes

def process_chunk_row(row, used_dims, pidx, pid_data_dict, count, k):
    if count[0] % 100000 == 0:
        print('proces',k,'has routed',count[0],'rows')
    count[0] += 1
    row_numpy = row.to_numpy()
    row_used_dims_list = row_numpy[used_dims].tolist()
    row_border = tuple(row_used_dims_list+row_used_dims_list)
    try:
        pid = list(pidx.intersection(row_border))[0]
    except:
        print(row_border)
    if pid in pid_data_dict:
        pid_data_dict[pid]+=[row_numpy.tolist()]
        #print('update dict..')
    else:
        pid_data_dict[pid]=[row_numpy.tolist()]
        #print('initialize dict..')

@ray.remote
def process_chunk(chunk, used_dims, partition_path, k):
    print("enter data routing process", k, '..')    
    pid_data_dict = {}
    partitions = load_partitions_from_file(partition_path)
    p = index.Property()
    p.leaf_capacity = 32
    p.index_capacity = 32
    p.NearMinimumOverlaoFactor = 16
    p.fill_factor = 0.8
    p.overwrite = True
    pidx = index.Index(properties = p)
    for i in range(len(partitions)):
        pidx.insert(i, kdnode_2_border(partitions[i]))
    count = [0]
    chunk.apply(lambda row: process_chunk_row(row, used_dims, pidx, pid_data_dict, count, k), axis=1)
    dict_id = ray.put(pid_data_dict)
    print("exit data routing process", k, ".")
    return dict_id


# In[6]:


def merge_dict(base_dict, new_dict):
    for key, val in new_dict.items():
        if key in base_dict:
            base_dict[key] += val
        else:
            base_dict[key] = val
    new_dict.clear()

def dump_dict_2_hdfs_simple(merged_dict, pq_writers, column_names, hdfs_path, fs):
    #print('= = = start dumping in main thread = = =')
    for pid, val in merged_dict.items():
        #print("writing to pid:",pid)
        path = hdfs_path + 'partition_' + str(pid)+'.parquet'
        pdf = pd.DataFrame(val, columns=column_names)
        adf = pa.Table.from_pandas(pdf)
        if pid in pq_writers:
            pq_writers[pid].write_table(table=adf)
        else:
            writer = pq.ParquetWriter(path, adf.schema, fs)
            pq_writers[pid] = writer
            writer.write_table(table=adf)
    #print('= = = exit dumping = = =')


# In[7]:


def batch_data_parallel(table_path, partition_path, chunk_size, used_dims, hdfs_path, num_dims, num_process, hdfs_private_ip):
    
    begin_time = time.time()
    
    ray.init(num_cpus=num_process)
    
    # column names for pandas dataframe
    cols = [i for i in range(num_dims)]
    col_names = ['_c'+str(i) for i in range(num_dims)]
    
    # pyarrow parquent append
    pq_writers = {}
    fs = pa.fs.HadoopFileSystem(hdfs_private_ip, port=9000, user='hdfs', replication=1)
    
    # chunks
    chunk_count = 0
    
    # collect object refs
    result_ids = []
    last_batch_ids = [] 
    first_loop = True
    
    for chunk in pd.read_table(table_path, delimiter='|', usecols=cols, names=col_names, chunksize=chunk_size):
        print('reading chunk: ', chunk_count)
        
        chunk_id = ray.put(chunk)
        result_id = process_chunk.remote(chunk_id, used_dims, partition_path, chunk_count)
        
        del chunk_id
        result_ids.append(result_id)
        del result_id
        
        # after all process allocated a chunk, process and dump the data
        if chunk_count % num_process == num_process - 1:
            
            if first_loop:
                first_loop = False
                last_batch_ids = result_ids.copy()
                result_ids.clear()
                chunk_count += 1
                continue
            else:
                print("= = = Process Dump For Chunk", chunk_count-2*num_process+1, "to", chunk_count-num_process, "= = =")
                base_dict = {}
                while len(last_batch_ids):
                    done_id, last_batch_ids = ray.wait(last_batch_ids)
                    dict_id = ray.get(done_id[0])
                    result_dict = ray.get(dict_id)
                    merge_dict(base_dict, result_dict)
                dump_dict_2_hdfs_simple(base_dict, pq_writers, col_names, hdfs_path, fs)
                base_dict.clear()
                print("= = = Finish Dump For Chunk", chunk_count-2*num_process+1, "to", chunk_count-num_process, "= = =")
                last_batch_ids = result_ids.copy()
                result_ids.clear()
                
            current_time = time.time()
            time_elapsed = current_time - begin_time
            print("= = = TOTAL PROCESSED SO FAR:", (chunk_count-num_process+1) * chunk_size,"ROWS. TIME SPENT:", time_elapsed, "SECONDS = = =")
                
        chunk_count += 1
        
    # process the last few batches
    print("= = = Process Dump For Last Few Chunks = = =")
    base_dict = {}
    while len(last_batch_ids):
        done_id, last_batch_ids = ray.wait(last_batch_ids)
        dict_id = ray.get(done_id[0])
        result_dict = ray.get(dict_id)
        merge_dict(base_dict, result_dict)
    dump_dict_2_hdfs_simple(base_dict, pq_writers, col_names, hdfs_path, fs)
    base_dict.clear()
    last_batch_ids.clear()

    base_dict = {}
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        dict_id = ray.get(done_id[0])
        result_dict = ray.get(dict_id)
        merge_dict(base_dict, result_dict)
    result_ids.clear() # clear up the references
    dump_dict_2_hdfs_simple(base_dict, pq_writers, col_names, hdfs_path, fs) 
    base_dict.clear()
    result_ids.clear()

    for pid, writer in pq_writers.items():
        writer.close()
    
    ray.shutdown()
    
    finish_time = time.time()
    print('= = = = = TOTAL DATA ROUTING AND PERISITING TIME:', finish_time - begin_time, "= = = = =")


# In[8]:


# = = = Configuration (UBDA Cloud Centos) = = =
scale_factor = 100

table_base_path = '/media/datadrive1/TPCH/dbgen/'
table_path = table_base_path + 'lineitem_' + str(scale_factor) + '.tbl'

num_process = 6
chunk_size = 2000000 
# 6M rows = about 1GB raw data

num_dims = 16
used_dims = [1,2]

# base path of HDFS
hdfs_private_ip = '192.168.6.62'
hdfs_base_path = 'hdfs://192.168.6.62:9000/user/cloudray/'

nora_hdfs = hdfs_base_path + 'NORA/scale' + str(scale_factor) + '/'
qdtree_hdfs = hdfs_base_path + 'QdTree/scale' + str(scale_factor) + '/'
kdtree_hdfs = hdfs_base_path + 'KDTree/scale' + str(scale_factor) + '/'

# base path of Partition
partition_base_path = '/home/centos/PartitionLayout/'

nora_partition = partition_base_path + 'nora_partitions_' + str(scale_factor)
qdtree_partition = partition_base_path + 'qdtree_partitions_' + str(scale_factor)
kdtree_partition = partition_base_path + 'kdtree_partitions_' + str(scale_factor)


# In[ ]:


# = = = Execution = = =
if __name__ == '__main__':
    #batch_data_parallel(table_path, nora_partition, chunk_size, used_dims, nora_hdfs, num_dims, num_process, hdfs_private_ip)
    #print('finish nora data routing..')
    #batch_data_parallel(table_path, qdtree_partition, chunk_size, used_dims, qdtree_hdfs, num_dims, num_process, hdfs_private_ip)
    #print('finish qdtree data routing..')
    batch_data_parallel(table_path, kdtree_partition, chunk_size, used_dims, kdtree_hdfs, num_dims, num_process, hdfs_private_ip)
    print('finish kdtree data routing..')


# In[10]:


# ray.shutdown()


# In[16]:


# read every parquet and dump it, see if it has a difference in query response time
# use pyarrow !!!


# In[ ]:




