import copy
import time
import random
import numpy as np
import pandas as pd # for batch data loading, in generating sampled dataset
from rtree import index # this package is only used for constructing Rtree filter
from numpy import genfromtxt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

class QueryMBR:
    '''
    the MBR that bound overlapped queries
    '''
    def __init__(self, boundary, added_as_fist_query = True):
        self.num_dims = len(boundary) / 2
        self.boundary = boundary
        self.num_query = 1
        self.queries = []
        self.bound_size = None # number of records this MBR overlaps
        self.total_query_result_size = None # total query results size of all the queries inside this MBR
        self.query_result_size = [] # record each query's result size
        self.is_extended = False
        self.ill_extended = False
        if added_as_fist_query:
            self.queries = [boundary]
        
    def check_condition3(self, data_threshold):
        '''
        check whether this MBR satisfy the new bounding split condition 3:
        1. every query size > BP - b
        2. total_query_result_size + b > bound_size * num_query
        '''
        for size in self.query_result_size:
            if size <= self.bound_size - data_threshold:
                return False
        
        if self.total_query_result_size + data_threshold <= self.bound_size * self.num_query:
            return False
        
        return True

class PartitionNode:
    '''
    A partition node, including both the internal and leaf nodes in the partition tree
    '''
    def __init__(self, num_dims = 0, boundary = [], nid = None, pid = None, is_irregular_shape_parent = False,
                 is_irregular_shape = False, num_children = 0, children_ids = [], is_leaf = True, node_size = 0):
        
        # print("Initialize PartitionTree Root: num_dims",num_dims,"boundary:",boundary,"children_ids:",children_ids)
        self.num_dims = num_dims # number of dimensions
        # the domain, [l1,l2,..,ln, u1,u2,..,un,], for irregular shape partition, one need to exempt its siblings
        self.boundary = boundary # I think the lower side should be inclusive and the upper side should be exclusive?
        self.nid = nid # node id
        self.pid = pid # parent id
        self.is_irregular_shape_parent = is_irregular_shape_parent # whether the [last] child is an irregular shape partition
        self.is_irregular_shape = is_irregular_shape # an irregular shape partition cannot be further split, and it must be a leaf node
        self.num_children = num_children # number of children, should be 0, 2, or 3
        self.children_ids = children_ids # if it's the irregular shape parent, then the last child should be the irregular partition
        self.is_leaf = is_leaf
        self.node_size = node_size # number of records in this partition
        
        # the following attributes will not be serialized
        self.dataset = None # only used in partition algorithms, temporary, should consist records that within this partition
        self.queryset = None # only used in partition algorithms, temporary, should consist queries that overlap this partition
        self.partitionable = True # only used in partition algorithms
        self.query_MBRs = None # only used in partition algorithms, temporary
        self.split_type = None # only used in partition algorithms
        
        # Rtree filters
        self.rtree_filters = None # a collection of MBRs, in the shape of boundary, used to indicate the data distribution
        
        # beam search
        self.depth = 0 # only used in beam search, root node depth is 0
        
    def is_overlap(self, query):
        '''
        query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
        !query dimension should match the partition dimensions! i.e., all projected or all not projected
        return 0 if no overlap
        return 1 if overlap
        return 2 if inside
        '''
        if len(query) != 2 * self.num_dims:
            return -1 # error
        
        overlap_flag = True
        inside_flag = True
        
        for i in range(self.num_dims):
            if query[i] >= self.boundary[self.num_dims + i] or query[self.num_dims + i] <= self.boundary[i]:
                overlap_flag = False
                inside_flag = False
                return 0
            elif query[i] < self.boundary[i] or query[self.num_dims + i] > self.boundary[self.num_dims + i]:
                inside_flag = False
                
        if inside_flag:
            return 2
        elif overlap_flag:
            return 1
        else:
            return 0
    
    def is_overlap_np(self, query):
        '''
        the numpy version of the is_overlap function
        the query here and boundary class attribute should in the form of numpy array
        '''
        if all((boundary[0:self.num_dims] > query[self.num_dims:]) | (boundary[self.num_dims:] <= query[0:self.num_dims])):
            return 0 # no overlap
        elif all((boundary[0:self.num_dims] >= query[0:self.num_dims]) & (boundary[self.num_dims:] <= query[self.num_dims:])):
            return 2 # inside
        else:
            return 1 # overlap
    
    def is_contain(self, point):
        '''
        used to determine wheter a data point is contained in this node
        point: [dim1_value, dim2_value,...], should has the same dimensions as this node
        '''
        for i in range(self.num_dims):
            if point[i] > self.boundary[self.num_dims + i] or point[i] < self.boundary[i]:
                return False
        return True
    
    def get_candidate_cuts(self, extended = False):
        '''
        get the candidate cut positions
        if extended is set to True, also add medians from all dimensions
        '''
        candidate_cut_pos = []
        for query in self.queryset:
            for dim in range(self.num_dims):
                # check if the cut position is inside the partition, as the queryset are queries overlap this partition
                if query[dim] >= self.boundary[dim] and query[dim] <= self.boundary[self.num_dims+dim]:
                    candidate_cut_pos.append((dim, query[dim]))
                if query[self.num_dims+dim] >= self.boundary[dim] and query[self.num_dims+dim] <= self.boundary[self.num_dims+dim]:
                    candidate_cut_pos.append((dim, query[self.num_dims+dim]))
        
        if extended:
            for dim in range(self.num_dims):
                split_value = np.median(self.dataset[:,dim])
                candidate_cut_pos.append((dim, split_value))
        
        return candidate_cut_pos
    
    def if_split(self, split_dim, split_value, data_threshold, test = False): # rename: if_split_get_gain
        '''
        return the skip gain and children partition size if split a node from a given split dimension and split value
        '''
        #print("current_node.nid:", current_node.nid)
        #print("current_node.is_leaf:", current_node.is_leaf)
        #print("current_node.dataset is None:", current_node.dataset is None)
        sub_dataset1_size = np.count_nonzero(self.dataset[:,split_dim] < split_value) # process time: 0.007
        sub_dataset2_size = self.node_size - sub_dataset1_size

        if sub_dataset1_size < data_threshold or sub_dataset2_size < data_threshold:
            return False, 0, sub_dataset1_size, sub_dataset2_size
        
        left_part, right_part, mid_part = self.split_queryset(split_dim, split_value)
        num_overlap_child1 = len(left_part) + len(mid_part)
        num_overlap_child2 = len(right_part) + len(mid_part)
        
        if test:
            print("num left part:",len(left_part), "num right part:",len(right_part), "num mid part:",len(mid_part))
            print("left part:", left_part, "right part:", right_part, "mid part:",mid_part)
        
        # temp_child_node1, temp_child_node2 = self.__if_split_get_child(split_dim, split_value)
        skip_gain = len(self.queryset)*self.node_size - num_overlap_child1*sub_dataset1_size - num_overlap_child2*sub_dataset2_size
        return True, skip_gain, sub_dataset1_size, sub_dataset2_size
    
    def if_bounding_split(self, data_threshold, approximate = False, force_extend = False):
        '''
        # the split node is assumed to be >= 2b
        approximate: whether use approximation (even distribution) to find the number of records within a partition
        force_extend: whether extend the bounding partition to make its size greater than data_threshold, if possible
        return availability, skip gain, and the (possible extended) bound
        '''
        max_bound = self.__max_bound(self.queryset)
        bound_size = self.query_result_size(max_bound, approximate)
        if bound_size is None:
            return False, None, None
        
        extended_bound = copy.deepcopy(max_bound)
        if bound_size < data_threshold: # assume the partition is >= 2b, then we must be able to find the valid extension
            if force_extend:
                side = 0
                for dim in range(self.num_dims):
                    valid, extended_bound, bound_size = self.__try_extend(extended_bound, dim, 0, data_threshold) # lower side
                    if valid:
                        break
                    valid, extended_bound, bound_size = self.__try_extend(extended_bound, dim, 1, data_threshold) # upper side
                    if valid:
                        break
            else:
                return False, None, None   
        
        remaining_size = self.node_size - bound_size
        if remaining_size < data_threshold:
            return False, None, None
        cost_before_split = len(self.queryset) * self.node_size
        cost_bound_split = len(self.queryset) * bound_size
        skip_gain = cost_before_split - cost_bound_split
        
        if force_extend:
            return True, skip_gain, extended_bound
        else:
            return True, skip_gain, max_bound # TODO: should we also return the extended bound? 

    def if_new_bounding_split(self, data_threshold, approximate = False, force_extend = True):
        '''
        In this version, we try to generate a collection of MBR partitions if every MBR satisfy:
        1. its size <= b; or
        2. it contains only 1 query; or
        3. |Q|*Core + b > its size * |Q|
        
        OR (if the above failed) a single bounding partition and an irregular shape partition as the old version
        '''
        if self.query_MBRs is None or len(self.query_MBRs) == 0:
            return False
        
        check_valid = True
        extended_flag = False
        
        # simple pruning
        if len(self.query_MBRs) * data_threshold > self.node_size:
            check_valid = False
        else:
            for MBR in self.query_MBRs:
                if MBR.bound_size <= data_threshold or MBR.num_query == 1 or MBR.check_condition3(data_threshold):
                    pass
                else:
                    check_valid = False
                    break
        
        if check_valid:
            # try extend the MBRs to satisfy b, and check whether the extended MBRs overlap with others
            for MBR in self.query_MBRs:
                if MBR.bound_size < data_threshold:
                    MBR.boundary, MBR.bound_size = self.extend_bound(MBR.boundary, data_threshold)
                    MBR.is_extended = True
                    if MBR.bound_size > 2 * data_threshold:
                        MBR.ill_extended = True # if there are too many same key records
                if MBR.is_extended:
                    extended_flag = True # also for historical extended MBRs !!!
                    
            
        # check if the extended MBRs overlaps each other
        if extended_flag and len(self.query_MBRs) > 1:
            for i in range(len(self.query_MBRs) - 1):
                for j in range(i+1, len(self.query_MBRs)):
                    if self.query_MBRs[i].ill_extended or self.query_MBRs[j].ill_extended or self.__is_overlap(self.query_MBRs[i].boundary, self.query_MBRs[j].boundary):
                        #print("partition",self.nid,"found overlap of extended MBRs:", self.query_MBRs[i].boundary, self.query_MBRs[j].boundary)
                        check_valid = False
                        break
                if not check_valid:
                    break
        
        if len(self.query_MBRs) == 1 and self.query_MBRs[0].ill_extended: # in case there is only 1 MBR
            check_valid = False
        
        # check the remaining partition size, if it's not greater than b, return false
        remaining_size = self.node_size
        for MBR in self.query_MBRs:
            remaining_size -= MBR.bound_size
        if remaining_size < data_threshold:
            check_valid = False
        
        # if the above failed
        if check_valid:
            return True # since this is the optimal, we don't need to return skip
        else:
            # do we need to restore the MBRs?
            # NO, when split cross a MBR, it will be rebuilt on both side
            # In other cases, the extended MBR doesn't matter
            return False  
    
    def if_dual_bounding_split(self, split_dim, split_value, data_threshold, approximate = False):
        '''
        check whether it's available to perform dual bounding split
        return availability and skip gain
        '''
        # split queriese first
        left_part, right_part, mid_part = self.split_queryset(split_dim, split_value)
        max_bound_left = self.__max_bound(left_part)
        max_bound_right = self.__max_bound(right_part)
        
        # Should we only consider the case when left and right cannot be further split? i.e., [b,2b)
        # this check logic is given in the PartitionAlgorithm, not here, as the split action should be general
        naive_left_size = np.count_nonzero(self.dataset[:,split_dim] < split_value)
        naive_right_size = self.node_size - naive_left_size
        
        # get (irregular-shape) sub-partition size
        left_size = self.query_result_size(max_bound_left, approximate)
        if left_size is None: # there is no query within the left 
            left_size = naive_left_size # use the whole left part as its size
        if left_size < data_threshold:
            return False, None
        right_size = self.query_result_size(max_bound_right, approximate)
        if right_size is None: # there is no query within the right
            right_size = naive_right_size # use the whole right part as its size
        if right_size < data_threshold:
            return False, None
        remaining_size = self.node_size - left_size - right_size
        if remaining_size < data_threshold:
            return False, None
        
        # check cost
        cost_before_split = len(self.queryset) * self.node_size
        cost_dual_split = len(left_part) * left_size + len(right_part) * right_size + len(mid_part) * remaining_size
        for query in mid_part:
            # if it overlap left bounding box
            if max_bound_left is None or self.__is_overlap(max_bound_left, query) > 0:
                cost_dual_split += left_size
            # if it overlap right bounding box
            if max_bound_right is None or self.__is_overlap(max_bound_right, query) > 0:
                cost_dual_split += right_size
        skip_gain = cost_before_split - cost_dual_split
        return True, skip_gain
        
    def num_query_crossed(self, split_dim, split_value):
        '''
        similar to the split_queryset function, but just return how many queries the intended split will cross
        '''
        count = 0
        if self.queryset is not None:
            for query in self.queryset:
                if query[split_dim] < split_value and query[self.num_dims + split_dim] > split_value:
                    count += 1
            return count
        return None
    
    def split_queryset(self, split_dim, split_value):
        '''
        split the queryset into 3 parts:
        the left part, the right part, and those cross the split value
        '''
        if self.queryset is not None:
            left_part = []
            right_part = []
            mid_part = []
            for query in self.queryset:
                if query[split_dim] >= split_value:
                    right_part.append(query)
                elif query[self.num_dims + split_dim] <= split_value:
                    left_part.append(query)
                elif query[split_dim] < split_value and query[self.num_dims + split_dim] > split_value:
                    mid_part.append(query)
            return left_part, right_part, mid_part
    
    def query_result_size(self, query, approximate = False):
        '''
        get the query result's size on this node
        the approximate parameter is set to True, the use even distribution to approximate
        '''
        if query is None:
            return None
        
        result_size = 0
        if approximate:
            query_volume = 1
            volume = 1
            for d in range(self.num_dims):
                query_volume *= query[self.num_dims + d] - query[d]
                volume *= self.boundary[self.num_dims + d] - self.boundary[d]

            result_size = int(query_volume / volume * self.node_size)
        else:
            constraints = []
            for d in range(self.num_dims):
                constraint_L = dataset[:,d] >= query[d]
                constraint_U = dataset[:,d] <= query[self.num_dims + d]
                constraints.append(constraint_L)
                constraints.append(constraint_U)
            constraint = np.all(constraints, axis=0)
            result_size = np.count_nonzero(constraint)
        return result_size
    
    def split_query_MBRs(self, split_dim, split_value):
        if self.query_MBRs is not None:
            left_part = [] # totally in left
            right_part = [] # totally in right
            mid_part = []
            for MBR in self.query_MBRs:
                if MBR.boundary[split_dim] >= split_value:
                    right_part.append(MBR)
                elif MBR.boundary[self.num_dims + split_dim] <= split_value:
                    left_part.append(MBR)
                elif MBR.boundary[split_dim] < split_value and MBR.boundary[self.num_dims + split_dim] > split_value:
                    mid_part.append(MBR)
                    
            # process each mid_part MBR
            overlap_left_part_queries = []
            overlap_right_part_queries = []
            for MBR in mid_part:
                for query in MBR.queries:
                    if query[split_dim] < split_value:
                        overlap_left_part_queries.append(query)
                    if query[self.num_dims + split_dim] > split_value:
                        overlap_right_part_queries.append(query)
                
            # generate MBRs for both part. Notice we cannot simply adjust the shape using original MBRs
            mid_part_left_MBRs = self.generate_query_MBRs(overlap_left_part_queries)
            mid_part_right_MBRs = self.generate_query_MBRs(overlap_right_part_queries)
            
            left_part += mid_part_left_MBRs
            right_part += mid_part_right_MBRs
            
            return left_part, right_part
    
    def generate_query_MBRs(self, queryset = None):
        '''
        bound the overlapped queries in this partition into MBRs
        the MBRs will only contains the part inside this partition
        '''
        if queryset is None:
            queryset = self.queryset
        
        if len(queryset) == 0:
            return []
        
        query_MBRs = []
        for query in queryset:
            query_MBRs.append(QueryMBR(query, True))
            
        #print("before merged, number of query MBRs:", len(query_MBRs))
        
        while len(query_MBRs) >= 2:
            
            new_query_MBRs = []
            merged_qids = []

            for i in range(len(query_MBRs)-1):
                new_MBR = copy.deepcopy(query_MBRs[i])
                
                if i in merged_qids:
                    continue
                
                for j in range(i+1, len(query_MBRs)):
                    if j in merged_qids:
                        continue
                    if self.__is_overlap(query_MBRs[i].boundary, query_MBRs[j].boundary):
                        #print("merge:",i,j,query_MBRs[i].boundary,query_MBRs[j].boundary)
                        new_MBR = self.__merge_2MBRs(new_MBR, query_MBRs[j])
                        merged_qids.append(j)
                
                new_query_MBRs.append(new_MBR)
                #print("for iteration",i, "current new_query_MBRs size:",len(new_query_MBRs))
            
            if len(query_MBRs)-1 not in merged_qids:
                new_query_MBRs.append(query_MBRs[-1])
            
            if len(query_MBRs) == len(new_query_MBRs):
                break
            else:
                query_MBRs = copy.deepcopy(new_query_MBRs)
        
        #print("after merged, number of query MBRs:", len(query_MBRs))
        
        # bound each query MBRs by its partition boundary, and calculate the result size
        for MBR in query_MBRs:
            MBR.boundary = self.__max_bound_single(MBR.boundary)
            MBR.bound_size = self.query_result_size(MBR.boundary)
            for query in MBR.queries:
                MBR.query_result_size.append(self.query_result_size(query))
            MBR.total_query_result_size = sum(MBR.query_result_size)
        
        self.query_MBRs = query_MBRs
        
        return query_MBRs
    
    def extend_bound(self, bound, data_threshold, print_info = False):
        '''
        extend a bound to be at least b, assume the bound is within the partition boundary
        '''
        side = 0
        for dim in [2,0,1,4,3,5,6]: #[0,1,4,3,5,6,2]: #range(self.num_dims): # reranged by distinct values
            if dim+1 > self.num_dims:
                continue
            
            valid, bound, bound_size = self.__try_extend(bound, dim, 0, data_threshold, print_info) # lower side
            if print_info:
                print("dim:",dim,"current bound:",bound,valid,bound_size)
            if valid:
                break
            valid, bound, bound_size = self.__try_extend(bound, dim, 1, data_threshold, print_info) # upper side
            if print_info:
                print("dim:",dim,"current bound:",bound,valid,bound_size)
            if valid:
                break
        return bound, bound_size
    
    # = = = = = internal functions = = = = =
    
    def __try_extend(self, current_bound, try_dim, side, data_threshold, print_info = False):
        '''
        side = 0: lower side
        side = 1: upper side
        return whether this extend has made bound greater than b, current extended bound, and the size
        '''
        # first try the extreme case
        dim = try_dim
        if side == 1:
            dim += self.num_dims
            
        extended_bound = copy.deepcopy(current_bound)
        extended_bound[dim] = self.boundary[dim]
        
        bound_size = self.query_result_size(extended_bound, approximate = False)
        if bound_size < data_threshold:
            return False, extended_bound, bound_size
        
        # binary search in this extend direction
        L, U = None, None
        if side == 0:
            L, U = self.boundary[dim], current_bound[dim]
        else:
            L, U = current_bound[dim], self.boundary[dim]
        
        if print_info:
            print("L,U:",L,U)
        
        loop_count = 0
        while L < U and loop_count < 30:
            mid = (L+U)/2
            extended_bound[dim] = mid
            bound_size = self.query_result_size(extended_bound, approximate = False)
            if bound_size < data_threshold:
                    L = mid
            elif bound_size > data_threshold:
                    U = mid
                    if U - L < 0.00001:
                        break
            else:
                break
            if print_info:
                print("loop,L:",L,"U:",U,"mid:",mid,"extended_bound:",extended_bound,"size:",bound_size)
            loop_count += 1
            
        return bound_size >= data_threshold, extended_bound, bound_size
        
    
    def __is_overlap(self, boundary, query):
        '''
        the difference between this function and the public is_overlap function lies in the boundary parameter
        '''
        if len(query) != 2 * self.num_dims:
            return -1 # error
        
        overlap_flag = True
        inside_flag = True
        
        for i in range(self.num_dims):
            if query[i] >= boundary[self.num_dims + i] or query[self.num_dims + i] <= boundary[i]:
                overlap_flag = False
                inside_flag = False
                return 0
            elif query[i] < boundary[i] or query[self.num_dims + i] > boundary[self.num_dims + i]:
                inside_flag = False
                
        if inside_flag:
            return 2
        elif overlap_flag:
            return 1
        else:
            return 0
        
    def __merge_2MBRs(self, MBR1, MBR2):
        '''
        merge 2 MBRs into 1 (the first one)
        in this step we do not consider whether the merged MBR exceeds the current partition
        '''
        for i in range(self.num_dims):
            MBR1.boundary[i] = min(MBR1.boundary[i], MBR2.boundary[i])
            MBR1.boundary[self.num_dims + i] = max(MBR1.boundary[self.num_dims + i], MBR2.boundary[self.num_dims + i])
        
        MBR1.queries += MBR2.queries
        MBR1.num_query += MBR2.num_query
        return MBR1
    
    def __max_bound(self, queryset):
        '''
        bound the queries by their maximum bounding rectangle !NOTE it is for a collection of queries!!!
        then constraint the MBR by the node's boundary!
        
        the return bound is in the same form as boundary
        '''
        if len(queryset) == 0:
            return None
        #if len(queryset) == 1:
        #    pass, I don't think there will be shape issue here
        
        max_bound_L = np.amin(np.array(queryset)[:,0:self.num_dims],axis=0).tolist()
        # bound the lower side with the boundary's lower side
        max_bound_L = np.amax(np.array([max_bound_L, self.boundary[0:self.num_dims]]),axis=0).tolist()
        
        max_bound_U = np.amax(np.array(queryset)[:,self.num_dims:],axis=0).tolist()
        # bound the upper side with the boundary's upper side
        max_bound_U = np.amin(np.array([max_bound_U, self.boundary[self.num_dims:]]),axis=0).tolist()
        
        max_bound = max_bound_L + max_bound_U # concat
        return max_bound
    
    def __max_bound_single(self, query):
        '''
        bound anything in the shape of query by the current partition boundary
        '''
        for i in range(self.num_dims):
            query[i] = max(query[i], self.boundary[i])
            query[self.num_dims + i] = min(query[self.num_dims + i], self.boundary[self.num_dims + i])
        return query
        
    
    def __if_split_get_child(self, split_dim, split_value): # should I rename this to if_split_get_child
        '''
        return 2 child nodes if a split take place on given dimension with given value
        This function is only used to simplify the skip calculation process, it does not really split the node
        '''
        boundary1 = self.boundary.copy()
        boundary1[split_dim + self.num_dims] = split_value
        boundary2 = self.boundary.copy()
        boundary2[split_dim] = split_value
        child_node1 = PartitionNode(self.num_dims, boundary1)
        child_node2 = PartitionNode(self.num_dims, boundary2)
        return child_node1, child_node2

class PartitionTree:
    '''
    The data structure that represent the partition layout, which also maintain the parent, children relation info
    Designed to provide efficient online query and serialized ability
    
    The node data structure could be checked from the PartitionNode class
    
    '''   
    def __init__(self, num_dims = 0, boundary = []):
        
        # the node id of root should be 0, its pid should be -1
        # note this initialization does not need dataset and does not set node size!

        self.pt_root = PartitionNode(num_dims, boundary, nid = 0, pid = -1, is_irregular_shape_parent = False, 
                                     is_irregular_shape = False, num_children = 0, children_ids = [], is_leaf = True, node_size = 0)
        self.nid_node_dict = {0: self.pt_root} # node id to node dictionary
        self.node_count = 1 # the root node
    
    # = = = = = public functions (API) = = = = =
    
    def save_tree(self, path):
        node_list = self.__generate_node_list(self.pt_root) # do we really need this step?
        serialized_node_list = self.__serialize(node_list)
        #print(serialized_node_list)
        np.savetxt(path, serialized_node_list, delimiter=',')
        return serialized_node_list
        
    def load_tree(self, path):
        serialized_node_list = genfromtxt(path, delimiter=',')
        self.__build_tree_from_serialized_node_list(serialized_node_list)
    
    def query_single(self, query, using_rtree_filter = False):
        '''
        query is in plain form, i.e., [l1,l2,...,ln, u1,u2,...,un]
        return the overlapped leaf partitions ids!
        '''
        partition_ids = self.__find_overlapped_partition(self.pt_root, query, using_rtree_filter)
        return partition_ids
    
    def query_batch(self, queries):
        '''
        to be implemented
        '''
        pass
    
    def get_queryset_cost(self, queries):
        '''
        return the cost array directly
        '''
        costs = []
        for query in queries:
            overlapped_leaf_ids = self.query_single(query)
            cost = 0
            for nid in overlapped_leaf_ids:
                cost += self.nid_node_dict[nid].node_size
            costs.append(cost)
        return costs
    
    def evaluate_query_cost(self, queries, print_result = False, using_rtree_filter = False):
        '''
        get the logical IOs of the queris
        return the average query cost
        '''
        total_cost = 0
        case = 0
        total_overlap_ids = {}
        case_cost = {}
        
        for query in queries:
            cost = 0
            overlapped_leaf_ids = self.query_single(query, using_rtree_filter)
            total_overlap_ids[case] = overlapped_leaf_ids
            for nid in overlapped_leaf_ids:
                cost += self.nid_node_dict[nid].node_size
            total_cost += cost
            case_cost[case] = cost
            case += 1
        
        if print_result:
            print("Total logical IOs:", total_cost)
            print("Average logical IOs:", total_cost // len(queries))
            for case, ids in total_overlap_ids.items():
                print("query",case, ids, "cost:", case_cost[case])
        
        return total_cost // len(queries)
    
    def get_pid_for_data_point(self, point):
        '''
        get the corresponding leaf partition nid for a data point
        point: [dim1_value, dim2_value...], contains the same dimenions as the partition tree
        '''
        return self.__find_resided_partition(self.pt_root, point)
    
    def add_node(self, parent_id, child_node):
        child_node.nid = self.node_count
        self.node_count += 1
        
        child_node.pid = parent_id
        self.nid_node_dict[child_node.nid] = child_node
        
        child_node.depth = self.nid_node_dict[parent_id].depth + 1
        
        self.nid_node_dict[parent_id].children_ids.append(child_node.nid)
        self.nid_node_dict[parent_id].num_children += 1
        self.nid_node_dict[parent_id].is_leaf = False
    
    
    def apply_split(self, parent_nid, split_dim, split_value, split_type = 0, extended_bound = None, approximate = False,
                    pretend = False):
        '''
        split_type = 0: split a node into 2 sub-nodes by a given dimension and value
        split_type = 1: split a node by bounding split (will create an irregular shape partition)
        split_type = 2: split a node by daul-bounding split (will create an irregular shape partition)
        
        extended_bound is only used in split type 1
        approximate: used for measure query result size
        pretend: if pretend is True, return the split result, but do not apply this split
        '''
        parent_node = self.nid_node_dict[parent_nid]
        if pretend:
            parent_node = copy.deepcopy(self.nid_node_dict[parent_nid])
        
        child_node1, child_node2 = None, None
        
        if split_type == 0:
        
            # create sub nodes
            child_node1 = copy.deepcopy(parent_node)
            child_node1.boundary[split_dim + child_node1.num_dims] = split_value
            child_node1.children_ids = []

            child_node2 = copy.deepcopy(parent_node)
            child_node2.boundary[split_dim] = split_value
            child_node2.children_ids = []
            
            if parent_node.query_MBRs is not None:
                MBRs1, MBRs2 = parent_node.split_query_MBRs(split_dim, split_value)
                child_node1.query_MBRs = MBRs1
                child_node2.query_MBRs = MBRs2
                
            # if parent_node.dataset != None: # The truth value of an array with more than one element is ambiguous.
            # https://stackoverflow.com/questions/36783921/valueerror-when-checking-if-variable-is-none-or-numpy-array
            if parent_node.dataset is not None:
                child_node1.dataset = parent_node.dataset[parent_node.dataset[:,split_dim] < split_value]
                child_node1.node_size = len(child_node1.dataset)
                child_node2.dataset = parent_node.dataset[parent_node.dataset[:,split_dim] >= split_value]
                child_node2.node_size = len(child_node2.dataset)

            if parent_node.queryset is not None:
                left_part, right_part, mid_part = parent_node.split_queryset(split_dim, split_value)
                child_node1.queryset = left_part + mid_part
                child_node2.queryset = right_part + mid_part

            # update current node
            if not pretend:
                self.add_node(parent_nid, child_node1)
                self.add_node(parent_nid, child_node2)
                self.nid_node_dict[parent_nid].split_type = "candidate cut"
        
        elif split_type == 1: # must reach leaf node, hence no need to maintain dataset and queryset any more
            
            child_node1 = copy.deepcopy(parent_node) # the bounding partition
            child_node2 = copy.deepcopy(parent_node) # the remaining partition, i.e., irregular shape
            
            child_node1.is_leaf = True
            child_node2.is_leaf = True
            
            child_node1.children_ids = []
            child_node2.children_ids = []
            
            max_bound = None
            if extended_bound is not None:
                max_bound = extended_bound
            else:
                max_bound = parent_node._PartitionNode__max_bound(parent_node.queryset)
            child_node1.boundary = max_bound
            child_node2.is_irregular_shape = True
            
            bound_size = parent_node.query_result_size(max_bound, approximate = False)
            remaining_size = parent_node.node_size - bound_size           
            child_node1.node_size = bound_size
            child_node2.node_size = remaining_size
            
            child_node1.partitionable = False
            child_node2.partitionable = False
            
            if not pretend:
                self.add_node(parent_nid, child_node1)
                self.add_node(parent_nid, child_node2)
                self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                self.nid_node_dict[parent_nid].split_type = "sole-bounding split"
        
        elif split_type == 2: # must reach leaf node, hence no need to maintain dataset and queryset any more
            
            child_node1 = copy.deepcopy(parent_node) # the bounding partition 1
            child_node2 = copy.deepcopy(parent_node) # the bounding partition 2
            child_node3 = copy.deepcopy(parent_node) # the remaining partition, i.e., irregular shape
            
            child_node1.is_leaf = True
            child_node2.is_leaf = True
            child_node3.is_leaf = True
            
            child_node1.children_ids = []
            child_node2.children_ids = []
            child_node3.children_ids = []
            
            left_part, right_part, mid_part = parent_node.split_queryset(split_dim, split_value)
            max_bound_1 = parent_node._PartitionNode__max_bound(left_part)
            max_bound_2 = parent_node._PartitionNode__max_bound(right_part)
            
            child_node1.boundary = max_bound_1
            child_node2.boundary = max_bound_2
            child_node3.is_irregular_shape = True          
            
            # Should we only consider the case when left and right cannot be further split? i.e., [b,2b)
            # this check logic is given in the PartitionAlgorithm, not here, as the split action should be general
            naive_left_size = np.count_nonzero(parent_node.dataset[:,split_dim] < split_value)
            naive_right_size = parent_node.node_size - naive_left_size

            # get (irregular-shape) sub-partition size
            bound_size_1 = parent_node.query_result_size(max_bound_1, approximate)
            if bound_size_1 is None: # there is no query within the left 
                bound_size_1 = naive_left_size # use the whole left part as its size
           
            bound_size_2 = parent_node.query_result_size(max_bound_2, approximate)
            if bound_size_2 is None: # there is no query within the right
                bound_size_2 = naive_right_size # use the whole right part as its size
           
            remaining_size = parent_node.node_size - bound_size_1 - bound_size_2
            
            child_node1.node_size = bound_size_1
            child_node2.node_size = bound_size_2
            child_node3.node_size = remaining_size
            
            child_node1.partitionable = False
            child_node2.partitionable = False
            child_node3.partitionable = False
            
            if not pretend:
                self.add_node(parent_nid, child_node1)
                self.add_node(parent_nid, child_node2)
                self.add_node(parent_nid, child_node3)
                self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                self.nid_node_dict[parent_nid].split_type = "dual-bounding split"
        
        elif split_type == 3: # new bounding split, create a collection of MBR partitions
            
            remaining_size = parent_node.node_size
            for MBR in parent_node.query_MBRs:
                child_node = copy.deepcopy(parent_node)
                child_node.is_leaf = True
                child_node.children_ids = []
                child_node.boundary = MBR.boundary
                child_node.node_size = MBR.bound_size
                child_node.partitionable = False
                remaining_size -= child_node.node_size
                if not pretend:
                    self.add_node(parent_nid, child_node)
            
            # the last irregular shape partition
            child_node = copy.deepcopy(parent_node)
            child_node.is_leaf = True
            child_node.children_ids = []
            child_node.is_irregular_shape = True
            child_node.node_size = remaining_size
            child_node.partitionable = False
            
            if not pretend:
                self.add_node(parent_nid, child_node)
                self.nid_node_dict[parent_nid].is_irregular_shape_parent = True
                self.nid_node_dict[parent_nid].split_type = "var-bounding split"
        
        else:
            print("Invalid Split Type!")
        
        if not pretend:
            del self.nid_node_dict[parent_nid].dataset
            del self.nid_node_dict[parent_nid].queryset
            #del self.nid_node_dict[parent_nid].query_MBRs
            #self.nid_node_dict[parent_nid] = parent_node
            
        return child_node1, child_node2
    
    def get_leaves(self, use_partitionable = False):
        nodes = []
        if use_partitionable:
            for nid, node in self.nid_node_dict.items():
                if node.is_leaf and node.partitionable:
                    nodes.append(node)
        else:
            for nid, node in self.nid_node_dict.items():
                if node.is_leaf:
                    nodes.append(node)
        return nodes
    
    def visualize(self, dims = [0, 1], queries = [], path = None):
        '''
        visualize the partition tree's leaf nodes
        '''
        if len(dims) == 2:
            self.__visualize_2d(dims, queries, path)
        else:
            self.__visualize_3d(dims[0:3], queries, path)
        
    
    # = = = = = internal functions = = = = =
    
    def __generate_node_list(self, node):
        '''
        recursively add childrens into the list
        '''
        node_list = [node]
        for nid in node.children_ids:
            node_list += self.__generate_node_list(self.nid_node_dict[nid])
        return node_list
    
    def __serialize(self, node_list):
        '''
        convert object to attributes to save
        '''
        serialized_node_list = []
        for node in node_list:
            # follow the same order of attributes in partition class
            attributes = [node.num_dims]
            #attributes += node.boundary
            if isinstance(node.boundary, list):
                attributes += node.boundary
            else:
                attributes += node.boundary.tolist()
            attributes.append(node.nid) # node id = its ow id
            attributes.append(node.pid) # parent id
            attributes.append(1 if node.is_irregular_shape_parent else 0)
            attributes.append(1 if node.is_irregular_shape else 0)
            attributes.append(node.num_children) # number of children
            #attributes += node.children_ids
            attributes.append(1 if node.is_leaf else 0)
            attributes.append(node.node_size)
            
            serialized_node_list.append(attributes)
        return serialized_node_list
    
    def __build_tree_from_serialized_node_list(self, serialized_node_list):
        
        self.pt_root = None
        self.nid_node_dict.clear()
        pid_children_ids_dict = {}
        
        for serialized_node in serialized_node_list:
            num_dims = int(serialized_node[0])
            boundary = serialized_node[1: 1+2*num_dims]
            nid = int(serialized_node[1+2*num_dims]) # node id
            pid = int(serialized_node[2+2*num_dims]) # parent id
            is_irregular_shape_parent = False if serialized_node[3+2*num_dims] == 0 else True
            is_irregular_shape = False if serialized_node[4+2*num_dims] == 0 else True
            num_children = int(serialized_node[5+2*num_dims])
#                 children_ids = []
#                 if num_children != 0:
#                     children_ids = serialized_node[1+5+2*num_dims: 1+num_children+1+5+2*num_dims] # +1 for the end exclusive
#                 is_leaf = False if serialized_node[1+num_children+5+2*num_dims] == 0 else True
#                 node_size = serialized_node[2+num_children+5+2*num_dims] # don't use -1 in case of match error
            is_leaf = False if serialized_node[6+2*num_dims] == 0 else True
            node_size = int(serialized_node[7+2*num_dims])
            
            node = PartitionNode(num_dims, boundary, nid, pid, is_irregular_shape_parent, 
                                 is_irregular_shape, num_children, [], is_leaf, node_size) # let the children_ids empty
            self.nid_node_dict[nid] = node # update dict
            
            if node.pid in pid_children_ids_dict:
                pid_children_ids_dict[node.pid].append(node.nid)
            else:
                pid_children_ids_dict[node.pid] = [node.nid]
        
        # make sure the irregular shape partition is placed at the end of the child list
        for pid, children_ids in pid_children_ids_dict.items():
            if pid == -1:
                continue
            if self.nid_node_dict[pid].is_irregular_shape_parent and not self.nid_node_dict[children_ids[-1]].is_irregular_shape:
                # search for the irregular shape partition
                new_children_ids = []
                irregular_shape_id = None
                for nid in children_ids:
                    if self.nid_node_dict[nid].is_irregular_shape:
                        irregular_shape_id = nid
                    else:
                        new_children_ids.append(nid)
                new_children_ids.append(irregular_shape_id)
                self.nid_node_dict[pid].children_ids = new_children_ids
            else:
                self.nid_node_dict[pid].children_ids = children_ids
        
        self.pt_root = self.nid_node_dict[0]
    
    def __bound_query_by_boundary(self, query, boundary):
        '''
        bound the query by a node's boundary
        '''
        bounded_query = query.copy()
        num_dims = self.pt_root.num_dims
        for dim in range(num_dims):
            bounded_query[dim] = max(query[dim], boundary[dim])
            bounded_query[num_dims+dim] = min(query[num_dims+dim], boundary[num_dims+dim])
        return bounded_query
    
    def __find_resided_partition(self, node, point):
        '''
        for data point only
        '''
        #print("enter function!")
        if node.is_leaf:
            #print("within leaf",node.nid)
            if node.is_contain(point):
                return node.nid
        
        for nid in node.children_ids:
            if self.nid_node_dict[nid].is_contain(point):
                #print("within child", nid, "of parent",node.nid)
                return self.__find_resided_partition(self.nid_node_dict[nid], point)
        
        #print("no children of node",node.nid,"contains point")
        return -1
    
    def __find_overlapped_partition(self, node, query, using_rtree_filter = False):
        
        if node.is_leaf:
            if using_rtree_filter and node.rtree_filters is not None:
                for mbr in node.rtree_filters:
                    if node._PartitionNode__is_overlap(mbr, query) > 0:
                        return [node.nid]
                return []
            else:
                return [node.nid] if node.is_overlap(query) > 0 else []
        
        node_id_list = []
        if node.is_overlap(query) <= 0:
            pass
        elif node.is_irregular_shape_parent: # special process for irregular shape partitions!
            # bound the query with parent partition's boundary, that's for the inside case determination
            bounded_query = self.__bound_query_by_boundary(query, node.boundary)
            
            overlap_irregular_shape_node_flag = False
            for nid in node.children_ids[0: -1]: # except the last one, should be the irregular shape partition
                overlap_case = self.nid_node_dict[nid].is_overlap(bounded_query)
                if overlap_case == 2:
                    node_id_list = [nid]
                    overlap_irregular_shape_node_flag = False
                    break
                if overlap_case == 1:
                    node_id_list.append(nid)
                    overlap_irregular_shape_node_flag = True
                #if overlap_case > 0:
                 #   node_id_list.append(nid)      
            if overlap_irregular_shape_node_flag:
                node_id_list.append(node.children_ids[-1])
        else:  
            for nid in node.children_ids:
                node_id_list += self.__find_overlapped_partition(self.nid_node_dict[nid], query, using_rtree_filter)
        return node_id_list
    
    def __visualize_2d(self, dims, queries = [], path = None):
        fig, ax = plt.subplots(1)
        
        num_dims = self.pt_root.num_dims
        plt.xlim(self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0]+num_dims])
        plt.ylim(self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1]+num_dims])
        
        leaves = self.get_leaves()
        for leaf in leaves:
            
            lower1 = leaf.boundary[dims[0]]
            lower2 = leaf.boundary[dims[1]]             
            upper1 = leaf.boundary[dims[0]+num_dims]
            upper2 = leaf.boundary[dims[1]+num_dims]
            
            rect = Rectangle((lower1,lower2),upper1-lower1,upper2-lower2,fill=False,edgecolor='g',linewidth=1)
            ax.text(lower1, lower2, leaf.nid, fontsize=7)
            ax.add_patch(rect)
        
        case = 0
        for query in queries:

            lower1 = query[dims[0]]
            lower2 = query[dims[1]]  
            upper1 = query[dims[0]+num_dims]
            upper2 = query[dims[1]+num_dims]    
            
            rect = Rectangle((lower1,lower2),upper1-lower1,upper2-lower2,fill=False,edgecolor='r',linewidth=1)
            ax.text(upper1, upper2, case, color='b',fontsize=7)
            case += 1
            ax.add_patch(rect)

        ax.set_xlabel('dim 1', fontsize=15)
        ax.set_ylabel('dim 2', fontsize=15)
        #plt.xticks(np.arange(0, 400001, 100000), fontsize=10)
        #plt.yticks(np.arange(0, 20001, 5000), fontsize=10)

        plt.tight_layout() # preventing clipping the labels when save to pdf

        if path is not None:
            fig.savefig(path)

        plt.show()
    
    def __visualize_3d(self, dims, queries = [], path = None):
        fig = plt.figure()
        ax = Axes3D(fig)
        
        num_dims = self.pt_root.num_dims
        plt.xlim(self.pt_root.boundary[dims[0]], self.pt_root.boundary[dims[0]+num_dims])
        plt.ylim(self.pt_root.boundary[dims[1]], self.pt_root.boundary[dims[1]+num_dims])
        ax.set_zlim(self.pt_root.boundary[dims[2]], self.pt_root.boundary[dims[2]+num_dims])
        
        leaves = self.get_leaves()
        for leaf in leaves:
            
            L1 = leaf.boundary[dims[0]]
            L2 = leaf.boundary[dims[1]]
            L3 = leaf.boundary[dims[2]]      
            U1 = leaf.boundary[dims[0]+num_dims]
            U2 = leaf.boundary[dims[1]+num_dims]
            U3 = leaf.boundary[dims[2]+num_dims]
            
            # the 12 lines to form a rectangle
            x = [L1, U1]
            y = [L2, L2]
            z = [L3, L3]
            ax.plot3D(x,y,z,color="g")
            y = [U2, U2]
            ax.plot3D(x,y,z,color="g")
            z = [U3, U3]
            ax.plot3D(x,y,z,color="g")
            y = [L2, L2]
            ax.plot3D(x,y,z,color="g")

            x = [L1, L1]
            y = [L2, U2]
            z = [L3, L3]
            ax.plot3D(x,y,z,color="g")
            x = [U1, U1]
            ax.plot3D(x,y,z,color="g")
            z = [U3, U3]
            ax.plot3D(x,y,z,color="g")
            x = [L1, L1]
            ax.plot3D(x,y,z,color="g")

            x = [L1, L1]
            y = [L2, L2]
            z = [L3, U3]
            ax.plot3D(x,y,z,color="g")
            x = [U1, U1]
            ax.plot3D(x,y,z,color="g")
            y = [U2, U2]
            ax.plot3D(x,y,z,color="g")
            x = [L1, L1]
            ax.plot3D(x,y,z,color="g")
        
        for query in queries:

            L1 = query[dims[0]]
            L2 = query[dims[1]]
            L3 = query[dims[2]]
            U1 = query[dims[0]+num_dims]
            U2 = query[dims[1]+num_dims]
            U3 = query[dims[2]+num_dims]

            # the 12 lines to form a rectangle
            x = [L1, U1]
            y = [L2, L2]
            z = [L3, L3]
            ax.plot3D(x,y,z,color="r")
            y = [U2, U2]
            ax.plot3D(x,y,z,color="r")
            z = [U3, U3]
            ax.plot3D(x,y,z,color="r")
            y = [L2, L2]
            ax.plot3D(x,y,z,color="r")

            x = [L1, L1]
            y = [L2, U2]
            z = [L3, L3]
            ax.plot3D(x,y,z,color="r")
            x = [U1, U1]
            ax.plot3D(x,y,z,color="r")
            z = [U3, U3]
            ax.plot3D(x,y,z,color="r")
            x = [L1, L1]
            ax.plot3D(x,y,z,color="r")

            x = [L1, L1]
            y = [L2, L2]
            z = [L3, U3]
            ax.plot3D(x,y,z,color="r")
            x = [U1, U1]
            ax.plot3D(x,y,z,color="r")
            y = [U2, U2]
            ax.plot3D(x,y,z,color="r")
            x = [L1, L1]
            ax.plot3D(x,y,z,color="r")

        if path is not None:
            fig.savefig(path)

        plt.show()