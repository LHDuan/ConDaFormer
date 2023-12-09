import torch
import torch.nn as nn
import spconv.pytorch as spconv
from timm.models.layers import DropPath, trunc_normal_
from torch_geometric.nn import voxel_grid
from torch_scatter import scatter_mean, scatter_sum, segment_csr, gather_csr
from .utils import to_3d_numpy, get_indices_params, tri_plane_self_attention, sparse_self_attention

class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self,
                 in_channels,
                 embed_channels,
                 stride=1,
                 norm_fn=None,
                 indice_key=None,
                 bias=False,
                 ):
        super().__init__()

        assert norm_fn is not None

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels)
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = spconv.SubMConv3d(
            embed_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Window_Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, quant_size, shift_win=False, dropout=0., 
                 qk_scale=None, qkv_bias=True, rel_query=True, rel_key=True, rel_value=True, indice_key=""
    ):   
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.shift_win = shift_win
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = to_3d_numpy(window_size)
        self.rel_query, self.rel_key, self.rel_value = rel_query, rel_key, rel_value

        self.quant_size = to_3d_numpy(quant_size)
        self.quant_grid_length = int((window_size[0] + 1e-4)/ quant_size[0])
        assert int((window_size[0] + 1e-4)/ quant_size[0]) == int((window_size[1] + 1e-4)/ quant_size[1])
        assert self.rel_query and self.rel_key and self.rel_value

        if self.rel_query:
            self.q_table = nn.Parameter(torch.zeros(2*self.quant_grid_length-1, 3, self.num_heads//3, self.head_dim))
            trunc_normal_(self.q_table, std=.02)
        if self.rel_key:
            self.k_table = nn.Parameter(torch.zeros(2*self.quant_grid_length-1, 3, self.num_heads//3, self.head_dim))
            trunc_normal_(self.k_table, std=.02)
        if self.rel_value:
            self.v_table = nn.Parameter(torch.zeros(2*self.quant_grid_length-1, 3, self.num_heads//3, self.head_dim))
            trunc_normal_(self.v_table, std=.02)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout, inplace=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout, inplace=True)

        self.conv_1 = spconv.SparseSequential(
                spconv.SubMConv3d(embed_dim, embed_dim//2, kernel_size=1, padding=1, groups=1, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(embed_dim//2, eps=1e-5, momentum=0.1),
            )
        self.conv_3 = spconv.SparseSequential(
                spconv.SubMConv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.1),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim//2),
                nn.BatchNorm1d(embed_dim//2, eps=1e-5, momentum=0.1),
            )
        self.conv_1_post = spconv.SparseSequential(
                spconv.SubMConv3d(embed_dim, embed_dim//2, kernel_size=1, padding=1, groups=1, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(embed_dim//2, eps=1e-5, momentum=0.1),
            )
        self.conv_3_post = spconv.SparseSequential(
                spconv.SubMConv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(embed_dim, eps=1e-5, momentum=0.1),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim//2),
                nn.BatchNorm1d(embed_dim//2, eps=1e-5, momentum=0.1),
            )

    def get_relative_position_index(self, window_size, xyz_ctg, index_0, index_1):
        window_size = torch.from_numpy(window_size).float().cuda()
        shift_size = 1/2 * window_size if self.shift_win else 0.0
        xyz_quant = (xyz_ctg - xyz_ctg.min(0)[0] + shift_size) % window_size
        xyz_quant = torch.div(xyz_quant, torch.from_numpy(self.quant_size).float().cuda(), rounding_mode='floor') #[N, 3]
        relative_position = xyz_quant[index_0.long()] - xyz_quant[index_1.long()] #[M, 3]
        relative_position_index = relative_position + self.quant_grid_length - 1       
        relative_position_index = relative_position_index.int()
        return relative_position_index

    def strip_attention(self, xyz, batch, window_size, query, key, value, q_table, k_table, v_table):
        index_0, index_0_offsets, n_max, index_1, index_1_offsets, sort_idx = get_indices_params(
            xyz, 
            batch, 
            window_size, 
            self.shift_win
        )
        kwargs = {"query": query.contiguous().float(),
            "key": key.contiguous().float(), 
            "value": value.contiguous().float(),
            "xyz": xyz.float(),
            "index_0": index_0.int(),
            "index_0_offsets": index_0_offsets.int(),
            "n_max": n_max,
            "index_1": index_1.int(), 
            "index_1_offsets": index_1_offsets.int(),
            "sort_idx": sort_idx,
            "window_size": window_size,
            "shift_win": self.shift_win,
            "quant_size": self.quant_size,
            "quant_grid_length": self.quant_grid_length,
            "relative_pos_query_table": q_table.contiguous().float(),
            "relative_pos_key_table": k_table.contiguous().float(),
            "relative_pos_value_table": v_table.contiguous().float()
        }
        out = sparse_self_attention(**kwargs)
        return out

    def forward_qkv(self, query, key, value, xyz, batch):
        q_table, k_table, v_table = self.q_table, self.k_table, self.v_table

        window_size_xy = self.window_size.copy() 
        window_size_xy[2] = 1
        window_size_xz = self.window_size.copy() 
        window_size_xz[1] = 1
        window_size_yz = self.window_size.copy() 
        window_size_yz[0] = 1
        s_h_m = self.num_heads // 3 # split_head_num
        out_xy = self.strip_attention(xyz, batch, window_size_xy, 
                                        query[:, :s_h_m], key[:, :s_h_m], value[:, :s_h_m], 
                                        q_table, k_table, v_table)
        out_xz = self.strip_attention(xyz, batch, window_size_xz, 
                                        query[:, s_h_m:2*s_h_m], key[:, s_h_m:2*s_h_m], value[:, s_h_m:2*s_h_m], 
                                        q_table, k_table, v_table)
        out_yz = self.strip_attention(xyz, batch, window_size_yz, 
                                        query[:, 2*s_h_m:], key[:, 2*s_h_m:], value[:, 2*s_h_m:], 
                                        q_table, k_table, v_table)
        return [out_xy, out_xz, out_yz]

    def forward_qkv_one_pass(self, query, key, value, xyz, batch):
        q_table, k_table, v_table = self.q_table, self.k_table, self.v_table
        window_size_xy = self.window_size.copy() 
        window_size_xy[2] = 1
        window_size_xz = self.window_size.copy() 
        window_size_xz[1] = 1
        window_size_yz = self.window_size.copy() 
        window_size_yz[0] = 1
        index_0_xy, index_0_offsets_xy, n_max_xy, index_1_xy, index_1_offsets_xy, sort_idx_xy = get_indices_params(
            xyz, 
            batch, 
            window_size_xy, 
            self.shift_win
        )

        index_0_xz, index_0_offsets_xz, n_max_xz, index_1_xz, index_1_offsets_xz, sort_idx_xz = get_indices_params(
            xyz, 
            batch, 
            window_size_xz, 
            self.shift_win
        )
        index_0_yz, index_0_offsets_yz, n_max_yz, index_1_yz, index_1_offsets_yz, sort_idx_yz = get_indices_params(
            xyz, 
            batch, 
            window_size_yz, 
            self.shift_win
        )

        s_h_m = self.num_heads // 3
        query_xy, key_xy, value_xy = query[:, :s_h_m][sort_idx_xy], key[:, :s_h_m][sort_idx_xy], value[:, :s_h_m][sort_idx_xy]
        query_xz, key_xz, value_xz = query[:, s_h_m:2*s_h_m][sort_idx_xz], key[:, s_h_m:2*s_h_m][sort_idx_xz], value[:, s_h_m:2*s_h_m][sort_idx_xz]
        query_yz, key_yz, value_yz = query[:, 2*s_h_m:][sort_idx_yz], key[:, 2*s_h_m:][sort_idx_yz], value[:, 2*s_h_m:][sort_idx_yz]

        xyz_xy, xyz_xz, xyz_yz = xyz[sort_idx_xy], xyz[sort_idx_xz], xyz[sort_idx_yz]
        
        relative_position_index_xy = self.get_relative_position_index(window_size_xy, xyz_xy, index_0_xy, index_1_xy)
        relative_position_index_xz = self.get_relative_position_index(window_size_xz, xyz_xz, index_0_xz, index_1_xz)
        relative_position_index_yz = self.get_relative_position_index(window_size_yz, xyz_yz, index_0_yz, index_1_yz)

        query_xyz = torch.cat([query_xy, query_xz, query_yz], dim=1)
        key_xyz = torch.cat([key_xy, key_xz, key_yz], dim=1)
        value_xyz = torch.cat([value_xy, value_xz, value_yz], dim=1)

        index_0_xyz = torch.cat([index_0_xy, index_0_xz, index_0_yz], dim=0)
        index_1_xyz = torch.cat([index_1_xy, index_1_xz, index_1_yz], dim=0)

        index_0_offsets_xyz = torch.cat([index_0_offsets_xy, index_0_offsets_xz, index_0_offsets_yz], dim=0)
        index_1_offsets_xyz = torch.cat([index_1_offsets_xy, index_1_offsets_xz, index_1_offsets_yz], dim=0)

        relative_position_index_xyz = torch.cat([relative_position_index_xy, relative_position_index_xz, relative_position_index_yz], dim=0)
        m_offsets = torch.cuda.IntTensor([index_0_xy.shape[0], index_0_xz.shape[0], index_0_yz.shape[0]]).cumsum(-1).int().cuda()

        n_max = max(max(n_max_xy, n_max_xz), n_max_yz)
        out = tri_plane_self_attention(
                query_xyz.contiguous().float(), key_xyz.contiguous().float(), value_xyz.contiguous().float(), 
                index_0_xyz.int(), index_0_offsets_xyz.int(),
                n_max,
                index_1_xyz.int(), index_1_offsets_xyz.int(),
                relative_position_index_xyz,
                m_offsets,
                q_table.contiguous().float(), k_table.contiguous().float(), v_table.contiguous().float()
        )
        out_xy, out_xz, out_yz = torch.empty_like(out[:, :s_h_m]), torch.empty_like(out[:, s_h_m:2*s_h_m]), torch.empty_like(out[:, 2*s_h_m:])
        out_xy[sort_idx_xy] = out[:, :s_h_m]
        out_xz[sort_idx_xz] = out[:, s_h_m:2*s_h_m]
        out_yz[sort_idx_yz] = out[:, 2*s_h_m:]
        return [out_xy, out_xz, out_yz]

    def get_qkv_from_inp(self, inp):
        N, C = inp.features.shape
        inp = inp.replace_feature(torch.cat([self.conv_1(inp).features, self.conv_3(inp).features], dim=1))
        qkv = self.qkv(inp.features).reshape(N, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3).contiguous()
        query, key, value = qkv[0], qkv[1], qkv[2] #[N, num_heads, C//num_heads]
        query = query * self.scale
        return query, key, value

    def get_feat_from_attn(self, inp, x):
        N, C = inp.features.shape
        x = torch.cat(x, dim=1)
        x = x.view(N, C).contiguous()
        inp_x = inp.replace_feature(x)
        x = torch.cat([self.conv_1_post(inp_x).features, self.conv_3_post(inp_x).features], dim=1).contiguous()
        return x

    def forward(self, inp, xyz, batch):
        assert xyz.shape[1] == 3
        query, key, value = self.get_qkv_from_inp(inp) 
        x = self.forward_qkv_one_pass(query, key, value, xyz, batch)
        x = self.get_feat_from_attn(inp, x)
        x = self.proj(x)
        x = self.proj_drop(x) #[N, C]
        return inp.replace_feature(x)


class BasicBlock_Attention(nn.Module):
    def __init__(self, dim, window_size, quant_size, head_dim=16, shift=False, drop_path_rate=0.0, 
                 mlp_ratio=4.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, indice_key=""
    ):
        super().__init__()
        self.window_size = window_size
        num_heads = dim // head_dim
        self.norm1 = norm_layer(dim)
        self.attn = Window_Attention(
            dim, 
            shift_win=shift,
            num_heads=num_heads, 
            window_size=window_size, 
            quant_size=quant_size, 
            indice_key=indice_key,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.indice_key = indice_key

    def forward(self, x, xyz, batch):
        # feats: [N, c]
        # xyz: [N, 3]
        # batch: [N]
        assert (x.indices[:, 0] == batch).all()
        residual = x

        out = x.replace_feature(self.norm1(x.features))
        out = self.attn(out, xyz, batch)
        out = out.replace_feature(residual.features + self.drop_path(out.features))
        out = out.replace_feature(out.features + self.drop_path(self.mlp(self.norm2(out.features))))
        return out


class Grid_Pool(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, kernel_size=2, stride=2, indice_id=1):
        super(Grid_Pool, self).__init__()
        self.indice_key = 'spconv{}'.format(indice_id)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6) #if indice_id != 1 else nn.Identity()
        self.stride = 2**indice_id
        self.window = torch.tensor([self.stride]*3)

    def forward(self, inp, xyz, xyz_count, batch):
        inp = inp.replace_feature(self.linear(self.norm(inp.features)))
        
        feat = inp.features

        cluster = voxel_grid(pos=xyz, size=self.window.type_as(xyz).to(xyz.device), batch=batch, start=0)
        unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)
        _, sorted_cluster_indices = torch.sort(cluster)
        
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])

        xyz_multi_count = xyz[sorted_cluster_indices] * xyz_count[sorted_cluster_indices]
        xyz_sum = segment_csr(xyz_multi_count, idx_ptr, reduce="sum")
        xyz_count_next = segment_csr(xyz_count[sorted_cluster_indices], idx_ptr, reduce="sum")
        xyz_count_next = xyz_count_next.clamp_(min=1)
        xyz_next = torch.true_divide(xyz_sum, xyz_count_next)
        batch_next = segment_csr(batch.float()[sorted_cluster_indices], idx_ptr, reduce="mean")
        batch_next = batch_next.int()
        
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")

        discrete_coord = torch.div(xyz_next, self.stride, rounding_mode='floor')
        coords_batch = torch.cat([batch_next.unsqueeze(-1).int(), discrete_coord.int()], dim=1)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist()
        out = spconv.SparseConvTensor(
            features=feat,
            indices=coords_batch.contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch_next[-1].tolist() + 1
        )
        return out, xyz_next, xyz_count_next, batch_next


class Grid_Unpool(nn.Module):
    INTERPOLATION = True
    def __init__(self, in_channels, skip_channels, out_channels, bias=True, kernel_size=2, stride=2, indice_id=1):
        super(Grid_Unpool, self).__init__()
        self.indice_key = 'spconv{}'.format(indice_id)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Sequential(nn.LayerNorm(in_channels, eps=1e-6),
                                  nn.Linear(in_channels, out_channels, bias=bias))
        self.proj_skip = nn.Sequential(nn.LayerNorm(skip_channels, eps=1e-6), # if indice_id != 1 else nn.Identity(),
                                       nn.Linear(skip_channels, out_channels, bias=bias))

    def forward(self, inp, skip_inp, xyz, batch, skip_xyz, skip_batch):
        inp = inp.replace_feature(self.proj(inp.features))
        skip_inp = skip_inp.replace_feature(self.proj_skip(skip_inp.features))

        import pointops2.pointops as pointops
        inp_offset = torch.cumsum(batch.long().bincount(), dim=0).long()
        skip_offset = torch.cumsum(skip_batch.long().bincount(), dim=0).long()
        inter_feat = pointops.interpolation(
                        xyz.contiguous(), skip_xyz.contiguous(), inp.features.contiguous(), 
                        inp_offset.int(), skip_offset.int())
        skip_inp = skip_inp.replace_feature(skip_inp.features + inter_feat)
        return skip_inp