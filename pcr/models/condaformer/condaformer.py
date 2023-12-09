import functools
import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
import math
from .modules import BasicBlock, BasicBlock_Attention, Grid_Pool, Grid_Unpool

from ..builder import MODELS

def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().cuda(non_blocking=True)

@MODELS.register_module("ConDaFormer-v1m1")
class ConDaFormer(nn.Module):
    DOWNSAMPLE = Grid_Pool
    UPSAMPLE = Grid_Unpool
    BLOCK = BasicBlock_Attention
    def __init__(self,
                 in_channels,
                 out_channels,
                 voxel_size=0.04,
                 window_size=4,
                 quant_size=0.25,
                 base_channels=48,
                 head_dim=16,
                 drop_path_rate=0.3,
                 channels=(48, 96, 192, 384, 384, 192, 96, 48),
                 layers=(3, 9, 3, 3, 0, 0, 0, 0),
        ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.voxel_size = voxel_size
        self.num_stages = len(layers) // 2
        
        window_size = np.array([window_size, window_size, window_size]).astype(np.float32)
        quant_size = np.array([quant_size, quant_size, quant_size]).astype(np.float32)

        enc_dprs = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.layers[:4]))]
        dec_dprs = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.layers[4:]))]
        cur_dps = 0

        norm_fn_bn = functools.partial(nn.BatchNorm1d, eps=1e-5, momentum=0.1)
        norm_fn_ln = functools.partial(nn.LayerNorm, eps=1e-6)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key='stem1'),
            norm_fn_bn(base_channels),
            nn.LeakyReLU(negative_slope=0.2),
            BasicBlock(base_channels, base_channels, norm_fn=norm_fn_bn, indice_key='stem2')
        )   

        self.pool1 = self.DOWNSAMPLE(base_channels, self.channels[0], indice_id=1)
        self.block1 = nn.ModuleList([self.BLOCK(self.channels[0], drop_path_rate=enc_dprs[cur_dps+i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm1") for i in range(self.layers[0])])
        cur_dps += self.layers[0]

        self.pool2 = self.DOWNSAMPLE(self.channels[0], self.channels[1], indice_id=2)
        self.block2 = nn.ModuleList([self.BLOCK(self.channels[1], drop_path_rate=enc_dprs[cur_dps+i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm2") for i in range(self.layers[1])])   
        cur_dps += self.layers[1]

        self.pool3 = self.DOWNSAMPLE(self.channels[1], self.channels[2], indice_id=3)
        self.block3 = nn.ModuleList([self.BLOCK(self.channels[2], drop_path_rate=enc_dprs[cur_dps+i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm3") for i in range(self.layers[2])])
        cur_dps += self.layers[2]

        self.pool4 = self.DOWNSAMPLE(self.channels[2], self.channels[3], indice_id=4)
        self.block4 = nn.ModuleList([self.BLOCK(self.channels[3], drop_path_rate=enc_dprs[cur_dps+i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm4") for i in range(self.layers[3])])
        cur_dps += self.layers[3]

        cur_dps = sum(self.layers[4:]) - 1
        self.unpool1 = self.UPSAMPLE(self.channels[3], self.channels[2], self.channels[4], indice_id=4)
        self.block5 = nn.ModuleList([self.BLOCK(self.channels[4], drop_path_rate=dec_dprs[cur_dps-i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm3") for i in range(self.layers[4])])
        cur_dps -= self.layers[4]
        
        self.unpool2 = self.UPSAMPLE(self.channels[4], self.channels[1], self.channels[5], indice_id=3)
        self.block6 = nn.ModuleList([self.BLOCK(self.channels[5], drop_path_rate=dec_dprs[cur_dps-i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm2") for i in range(self.layers[5])])
        cur_dps -= self.layers[5]

        self.unpool3 = self.UPSAMPLE(self.channels[5], self.channels[0], self.channels[6], indice_id=2)
        self.block7 = nn.ModuleList([self.BLOCK(self.channels[6], drop_path_rate=dec_dprs[cur_dps-i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm1") for i in range(self.layers[6])])
        cur_dps -= self.layers[6]
        
        self.unpool4 = self.UPSAMPLE(self.channels[6], base_channels, self.channels[7], indice_id=1)
        self.block8 = nn.ModuleList([self.BLOCK(self.channels[7], drop_path_rate=dec_dprs[cur_dps-i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm0") for i in range(self.layers[7])])
        cur_dps -= self.layers[7]

        self.final = nn.Sequential(
            nn.Linear(self.channels[7], self.channels[7]),
            nn.BatchNorm1d(self.channels[7]),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels[7], out_channels)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, spconv.SubMConv3d) and m.groups != 1:
            with torch.no_grad():
                n = m.out_channels * int(np.prod(m.kernel_size)) 
                stdv = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        discrete_coord = input_dict["discrete_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        coord = input_dict["coord"]
        batch = offset2batch(offset)

        for i in range(len(offset)):
            batch_mask = batch == i
            coord[batch_mask] = coord[batch_mask] - coord[batch_mask].min(0)[0]
            coord[batch_mask] = coord[batch_mask] / self.voxel_size
            discrete_coord[batch_mask] = discrete_coord[batch_mask] - discrete_coord[batch_mask].min(0)[0]
            coord[batch_mask] = coord[batch_mask] - torch.min(coord[batch_mask]-discrete_coord[batch_mask], dim=0)[0]

        coords_batch = torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist()
        
        inp = spconv.SparseConvTensor(
            features=feat,
            indices=coords_batch.contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1
        )        

        out_p1 = self.stem(inp)
        points_p1, batch_p1 = coord, batch
        counts_p1 = torch.ones(len(points_p1), dtype=points_p1.dtype, device=points_p1.device).unsqueeze_(1)

        out_p2, points_p2, counts_p2, batch_p2 = self.pool1(out_p1, points_p1, counts_p1, batch_p1)
        for module in self.block1:
            out_p2 = module(out_p2, points_p2 / 2.0, batch_p2)

        out_p4, points_p4, counts_p4, batch_p4 = self.pool2(out_p2, points_p2, counts_p2, batch_p2)
        for module in self.block2:
            out_p4 = module(out_p4, points_p4 / 4.0, batch_p4)

        out_p8, points_p8, counts_p8, batch_p8 = self.pool3(out_p4, points_p4, counts_p4, batch_p4)
        for module in self.block3:
            out_p8 = module(out_p8, points_p8 / 8.0, batch_p8)

        out_p16, points_p16, counts_p16, batch_p16 = self.pool4(out_p8, points_p8, counts_p8, batch_p8)
        for module in self.block4:
            out_p16 = module(out_p16, points_p16 / 16.0, batch_p16)

        out = self.unpool1(out_p16, out_p8, points_p16, batch_p16, points_p8, batch_p8)
        for module in self.block5:
            out = module(out, points_p8 / 8.0, batch_p8)

        out = self.unpool2(out, out_p4, points_p8, batch_p8, points_p4, batch_p4)
        for module in self.block6:
            out = module(out, points_p4 / 4.0, batch_p4)

        out = self.unpool3(out, out_p2, points_p4, batch_p4, points_p2, batch_p2)
        for module in self.block7:
            out = module(out, points_p2 / 2.0, batch_p2)

        out = self.unpool4(out, out_p1, points_p2, batch_p2, points_p1, batch_p1)
        for module in self.block8:
            out = module(out, points_p1 , batch_p1)

        out = self.final(out.features)
        if "inverse" in input_dict:
            inverse_idx = input_dict["inverse"]
            inverse_length = input_dict["length"]
            st = inverse_length[0]
            for i in range(1, len(inverse_length)):
                inverse_idx[st:st+inverse_length[i]] += offset[i-1]
                st += inverse_length[i]
            inverse_idx = inverse_idx.long()
            out = out[inverse_idx]
        return out


@MODELS.register_module("ConDaFormer-small-v1m1")
class ConDaFormer_Small(nn.Module):
    DOWNSAMPLE = Grid_Pool
    UPSAMPLE = Grid_Unpool
    BLOCK = BasicBlock_Attention
    def __init__(self,
                 in_channels,
                 out_channels,
                 voxel_size=0.04,
                 window_size=4,
                 quant_size=0.25,
                 base_channels=48,
                 head_dim=16,
                 drop_path_rate=0.3,
                 channels=(96, 192, 384, 192, 96, 48),
                 layers=(2, 6, 2, 0, 0, 0),
        ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.voxel_size = voxel_size
        self.num_stages = len(layers) // 2
        
        window_size = np.array([window_size, window_size, window_size]).astype(np.float32)
        quant_size = np.array([quant_size, quant_size, quant_size]).astype(np.float32)

        enc_dprs = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.layers[:3]))]
        dec_dprs = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.layers[3:]))]
        cur_dps = 0

        norm_fn_bn = functools.partial(nn.BatchNorm1d, eps=1e-5, momentum=0.1)
        norm_fn_ln = functools.partial(nn.LayerNorm, eps=1e-6)

        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False, indice_key='stem1'),
            norm_fn_bn(base_channels),
            nn.LeakyReLU(negative_slope=0.2),
            BasicBlock(base_channels, base_channels, norm_fn=norm_fn_bn, indice_key='stem2')
        )   

        self.pool1 = self.DOWNSAMPLE(base_channels, self.channels[0], indice_id=1)
        self.block1 = nn.ModuleList([self.BLOCK(self.channels[0], drop_path_rate=enc_dprs[cur_dps+i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm1") for i in range(self.layers[0])])
        cur_dps += self.layers[0]

        self.pool2 = self.DOWNSAMPLE(self.channels[0], self.channels[1], indice_id=2)
        self.block2 = nn.ModuleList([self.BLOCK(self.channels[1], drop_path_rate=enc_dprs[cur_dps+i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm2") for i in range(self.layers[1])])   
        cur_dps += self.layers[1]

        self.pool3 = self.DOWNSAMPLE(self.channels[1], self.channels[2], indice_id=3)
        self.block3 = nn.ModuleList([self.BLOCK(self.channels[2], drop_path_rate=enc_dprs[cur_dps+i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm3") for i in range(self.layers[2])])
        cur_dps += self.layers[2]

        cur_dps = sum(self.layers[3:]) - 1
        self.unpool1 = self.UPSAMPLE(self.channels[2], self.channels[1], self.channels[3], indice_id=3)
        self.block4 = nn.ModuleList([self.BLOCK(self.channels[3], drop_path_rate=dec_dprs[cur_dps-i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm2") for i in range(self.layers[3])])
        cur_dps -= self.layers[3]
        
        self.unpool2 = self.UPSAMPLE(self.channels[3], self.channels[0], self.channels[4], indice_id=2)
        self.block5 = nn.ModuleList([self.BLOCK(self.channels[4], drop_path_rate=dec_dprs[cur_dps-i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm1") for i in range(self.layers[4])])
        cur_dps -= self.layers[4]

        self.unpool3 = self.UPSAMPLE(self.channels[4], base_channels, self.channels[5], indice_id=1)
        self.block6 = nn.ModuleList([self.BLOCK(self.channels[5], drop_path_rate=dec_dprs[cur_dps-i], head_dim=head_dim, 
                                     window_size=window_size, quant_size=quant_size, norm_layer=norm_fn_ln,
                                     shift=True if i % 2 == 1 else False, indice_key="subm0") for i in range(self.layers[5])])
        self.final = nn.Sequential(
            nn.Linear(self.channels[5], self.channels[5]),
            nn.BatchNorm1d(self.channels[5]),
            nn.ReLU(inplace=True),
            nn.Linear(self.channels[5], out_channels)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, spconv.SubMConv3d) and m.groups != 1:
            with torch.no_grad():
                n = m.out_channels * int(np.prod(m.kernel_size)) 
                stdv = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-stdv, stdv)
                if m.bias is not None:
                    m.bias.data.uniform_(-stdv, stdv)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        discrete_coord = input_dict["discrete_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        coord = input_dict["coord"]
        batch = offset2batch(offset)

        for i in range(len(offset)):
            batch_mask = batch == i
            coord[batch_mask] = coord[batch_mask] - coord[batch_mask].min(0)[0]
            coord[batch_mask] = coord[batch_mask] / self.voxel_size
            discrete_coord[batch_mask] = discrete_coord[batch_mask] - discrete_coord[batch_mask].min(0)[0]
            coord[batch_mask] = coord[batch_mask] - torch.min(coord[batch_mask]-discrete_coord[batch_mask], dim=0)[0]

        coords_batch = torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist()
        
        inp = spconv.SparseConvTensor(
            features=feat,
            indices=coords_batch.contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1
        )        

        out_p1 = self.stem(inp)
        points_p1, batch_p1 = coord, batch
        counts_p1 = torch.ones(len(points_p1), dtype=points_p1.dtype, device=points_p1.device).unsqueeze_(1)

        out_p2, points_p2, counts_p2, batch_p2 = self.pool1(out_p1, points_p1, counts_p1, batch_p1)
        for module in self.block1:
            out_p2 = module(out_p2, points_p2 / 2.0, batch_p2)

        out_p4, points_p4, counts_p4, batch_p4 = self.pool2(out_p2, points_p2, counts_p2, batch_p2)
        for module in self.block2:
            out_p4 = module(out_p4, points_p4 / 4.0, batch_p4)

        out_p8, points_p8, counts_p8, batch_p8 = self.pool3(out_p4, points_p4, counts_p4, batch_p4)
        for module in self.block3:
            out_p8 = module(out_p8, points_p8 / 8.0, batch_p8)

        out = self.unpool1(out_p8, out_p4, points_p8, batch_p8, points_p4, batch_p4)
        for module in self.block4:
            out = module(out, points_p4 / 4.0, batch_p4)

        out = self.unpool2(out, out_p2, points_p4, batch_p4, points_p2, batch_p2)
        for module in self.block5:
            out = module(out, points_p2 / 2.0, batch_p2)

        out = self.unpool3(out, out_p1, points_p2, batch_p2, points_p1, batch_p1)
        for module in self.block6:
            out = module(out, points_p1 / 1.0, batch_p1) 

        out = self.final(out.features)
        if "inverse" in input_dict:
            inverse_idx = input_dict["inverse"]
            inverse_length = input_dict["length"]
            st = inverse_length[0]
            for i in range(1, len(inverse_length)):
                inverse_idx[st:st+inverse_length[i]] += offset[i-1]
                st += inverse_length[i]
            inverse_idx = inverse_idx.long()
            out = out[inverse_idx]
        return out