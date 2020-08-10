from rtree import index
from numpy import genfromtxt

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

def process_chunk_row(row, used_dims, pidx, pid_data_dict):
	row_numpy = row.to_numpy()
	row_used_dims_list = row_numpy[used_dims].tolist()
	row_border = tuple(row_used_dims_list+row_used_dims_list)
	try:
		pid = list(pidx.intersection(row_border))[0]
	except:
		print(row_border)
	if pid in pid_data_dict:
		pid_data_dict[pid].append(row_numpy.tolist())
	else:
		pid_data_dict.update({pid:[row_numpy.tolist()]})

def process_chunk(parameters):
	chunk, used_dims, partition_path, pid_data_dict = parameters
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
	chunk.apply(lambda row: process_chunk_row(row, used_dims, pidx, pid_data_dict), axis=1)
	return pid_data_dict