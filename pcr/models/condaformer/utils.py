import numbers
import torch
import numpy as np
from torch_scatter import segment_csr, gather_csr
from torch_geometric.nn import voxel_grid
from .functional import precompute_all, dot_prod_with_idx_all_tri_plane, attention_step2_with_rel_pos_value_tri_plane
from .functional import dot_prod_with_idx_all, attention_step2_with_rel_pos_value

def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().cuda(non_blocking=True)

def to_3d_numpy(size):
    if isinstance(size, numbers.Number):
        size = np.array([size, size, size]).astype(np.float32)
    elif isinstance(size, list):
        size = np.array(size)
    elif isinstance(size, np.ndarray):
        size = size
    else:
        raise ValueError("size is either a number, or a list, or a np.ndarray")
    return size

def grid_sample(pos, batch, size, start, return_p2v=True, return_counts=True, return_unique=False):
    # pos: float [N, 3]
    # batch: long [N]
    # size: float [3, ]
    # start: float [3, ] / None

    cluster = voxel_grid(pos, batch, size, start=start) #[N, ]

    if return_p2v == False and return_counts == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)

    if return_p2v == False and return_counts == True:
        return cluster, counts.max().item(), counts

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)

    if return_unique:
        return cluster, p2v_map, counts, unique

    return cluster, p2v_map, counts

def get_indices_params(xyz, batch, window_size, shift_win: bool):
    
    if isinstance(window_size, list) or isinstance(window_size, np.ndarray):
        window_size = torch.from_numpy(window_size).type_as(xyz).to(xyz.device)
    else:
        window_size = torch.tensor([window_size]*3).type_as(xyz).to(xyz.device)
    
    if shift_win:
        v2p_map, k, counts = grid_sample(xyz+1/2*window_size, batch, window_size, start=xyz.min(0)[0], return_p2v=False, return_counts=True)
    else:
        v2p_map, k, counts = grid_sample(xyz, batch, window_size, start=xyz.min(0)[0], return_p2v=False, return_counts=True)
    v2p_map, sort_idx = v2p_map.sort()

    n = counts.shape[0]
    N = v2p_map.shape[0]

    n_max = k
    index_0_offsets, index_1_offsets, index_0, index_1 = precompute_all(N, n, n_max, counts)
    index_0 = index_0.long()
    index_1 = index_1.long()

    return index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx

def scatter_softmax_csr(src: torch.Tensor, indptr: torch.Tensor, dim: int = -1):
    ''' src: (N, C),
        index: (Ni+1, ), [0, n0^2, n0^2+n1^2, ...]
    '''
    max_value_per_index = segment_csr(src, indptr, reduce='max')
    max_per_src_element = gather_csr(max_value_per_index, indptr)
    
    recentered_scores = src - max_per_src_element
    recentered_scores_exp = recentered_scores.exp_()

    sum_per_index = segment_csr(
        recentered_scores_exp, indptr, reduce='sum')
    
    normalizing_constants = gather_csr(sum_per_index, indptr)

    return recentered_scores_exp.div(normalizing_constants)

def tri_plane_self_attention(
    query, 
    key, 
    value, 
    index_0,
    index_0_offsets,
    n_max,
    index_1,
    index_1_offsets,
    relative_position_index,
    m_offsets,
    relative_pos_query_table=None, 
    relative_pos_key_table=None, 
    relative_pos_value_table=None,
):

    attn_flat = dot_prod_with_idx_all_tri_plane(query, index_0, index_0_offsets, key, index_1, index_1_offsets, relative_pos_query_table, relative_pos_key_table, relative_position_index, n_max, m_offsets)
    index_0_offset_shape = query.shape[0] + 1
    softmax_attn_flat = torch.zeros_like(attn_flat)
    softmax_attn_flat[0:m_offsets[0]] = scatter_softmax_csr(src=attn_flat[0:m_offsets[0]], indptr=index_0_offsets[0:index_0_offset_shape].long(), dim=0)
    softmax_attn_flat[m_offsets[0]:m_offsets[1]] = scatter_softmax_csr(src=attn_flat[m_offsets[0]:m_offsets[1]], indptr=index_0_offsets[index_0_offset_shape:2*index_0_offset_shape].long(), dim=0)
    softmax_attn_flat[m_offsets[1]:m_offsets[2]] = scatter_softmax_csr(src=attn_flat[m_offsets[1]:m_offsets[2]], indptr=index_0_offsets[2*index_0_offset_shape:].long(), dim=0)
    x = attention_step2_with_rel_pos_value_tri_plane(softmax_attn_flat, value, index_0, index_0_offsets, n_max, index_1, index_1_offsets, relative_pos_value_table, relative_position_index, m_offsets)
    return x

def sparse_self_attention(query, 
    key, 
    value, 
    xyz,
    index_0,
    index_0_offsets,
    n_max,
    index_1,
    index_1_offsets,
    sort_idx,
    window_size, 
    shift_win, 
    quant_size=None, 
    quant_grid_length=None, 
    relative_pos_query_table=None, 
    relative_pos_key_table=None, 
    relative_pos_value_table=None,
    pe_type='contextual', 
    rel_query=True, 
    rel_key=True, 
    rel_value=True, 
):
    query = query[sort_idx]
    key = key[sort_idx]
    value = value[sort_idx]
    xyz_ctg = xyz[sort_idx]
    
    window_size = torch.from_numpy(window_size).float().cuda()
    shift_size = 1/2 * window_size if shift_win else 0.0
    xyz_quant = (xyz_ctg - xyz_ctg.min(0)[0] + shift_size) % window_size
    xyz_quant = torch.div(xyz_quant, torch.from_numpy(quant_size).float().cuda(), rounding_mode='floor') #[N, 3]
    relative_position = xyz_quant[index_0.long()] - xyz_quant[index_1.long()] #[M, 3]
    relative_position_index = relative_position + quant_grid_length - 1       
    relative_position_index = relative_position_index.int()
    attn_flat = dot_prod_with_idx_all(query, index_0, index_0_offsets, key, index_1, index_1_offsets, relative_pos_query_table, relative_pos_key_table, relative_position_index, n_max)

    softmax_attn_flat = scatter_softmax_csr(src=attn_flat, indptr=index_0_offsets.long(), dim=0) #[M, num_heads]
    x = attention_step2_with_rel_pos_value(softmax_attn_flat, value, index_0, index_0_offsets, n_max, index_1, index_1_offsets, relative_pos_value_table, relative_position_index)
    
    out = torch.empty_like(x)
    out[sort_idx] = x
    x = out
    return x