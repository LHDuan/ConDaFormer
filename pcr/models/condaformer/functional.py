import time
import torch
from torch.autograd import Function
import sptr_cuda

DEBUG = False

class PrecomputeAll(Function):
    @staticmethod
    def forward(ctx, N, n, n_max, counts):
        """
        input: p2v_map: (n, k), counts: (n), index_0_offsets: (N+1)
        output: index_0, index_1: [M]
        """
        assert counts.is_contiguous()
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(-1)], 0)
        sq_offsets = torch.cat([counts.new_zeros(1), (counts**2).cumsum(-1)], 0)
        M = sq_offsets[-1].item()

        index_0_offsets = torch.cuda.IntTensor(N).zero_()
        index_1_offsets = torch.cuda.IntTensor(N).zero_()
        index_0 = torch.cuda.IntTensor(M).zero_()
        index_1 = torch.cuda.IntTensor(M).zero_()

        sptr_cuda.precompute_all_cuda(N, n, n_max, counts.int(), offsets.int(), sq_offsets.int(), index_0_offsets, index_1_offsets, index_0, index_1)
        index_0_offsets = torch.cat([index_0_offsets, torch.tensor([M]).cuda()], 0)
        return index_0_offsets, index_1_offsets, index_0, index_1

precompute_all = PrecomputeAll.apply

class DotProdWithIdxAll(Function):
    @staticmethod
    def forward(ctx, q, index_q, index_q_offsets, k, index_k, index_k_offsets, table_q, table_k, rel_idx, n_max):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, 3, h, hdim), table_k: (L, 3, h, hdim), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L
        assert q.shape[0] == k.shape[0]
        assert L > rel_idx.max(), "L = {}, while rel_idx.max() = {}".format(L, rel_idx.max())
        assert L <= 128

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        output = torch.cuda.FloatTensor(h, M).zero_()
        
        q_transpose = q.permute(1,2,0).contiguous() #[h, hdim, N]
        k_transpose = k.permute(1,2,0).contiguous() #[h, hdim, N]
        table_q_transpose = table_q.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        table_k_transpose = table_k.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        rel_idx_transpose = rel_idx.permute(1,0).contiguous() #[3, M]

        sptr_cuda.dot_prod_with_idx_all_forward_cuda(N, M, h, hdim, n_max, L, q_transpose, index_q, index_q_offsets, k_transpose, index_k, table_q_transpose, table_k_transpose, rel_idx_transpose, output)
        
        output = output.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_all: forward time: {}s, M: {}".format(time.time() - t, M))
        
        ctx.n_max = n_max
        ctx.save_for_backward(q, index_q_offsets, index_q, k, index_k_offsets, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, index_q, k, index_k_offsets, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and index_q.is_contiguous() and k.is_contiguous() and index_k_offsets.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        assert L <= 128
        assert 512 % hdim == 0

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        sptr_cuda.dot_prod_with_idx_backward_cuda(N, M, h, hdim, n_max, L, grad_output, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k)
        
        grad_q_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_k_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()

        sptr_cuda.attention_step1_backward_cuda(N, M, h, hdim, n_max, grad_output, index_q, index_q_offsets, index_k, index_k_offsets, q, k, grad_q_2, grad_k_2)
        
        grad_q += grad_q_2
        grad_k += grad_k_2

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_all: backward time: {}s, M: {}".format(time.time() - t, M))
        
        return grad_q, None, None, grad_k, None, None, grad_table_q, grad_table_k, None, None

dot_prod_with_idx_all = DotProdWithIdxAll.apply

class AttentionStep2WithRelPosValue(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, 3, h, hdim), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and index1_offsets.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, h = attn.shape
        _, h, hdim = v.shape
        N = index0_offsets.shape[0] - 1
        L = table.shape[0]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
            
        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        # assert hdim == 16
        assert L <= 128

        sptr_cuda.attention_step2_with_rel_pos_value_forward_cuda(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, 3, h, hdim).zero_()
        
        table = table.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        v = v.permute(1,2,0).contiguous()
        rel_idx = rel_idx.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        sptr_cuda.attention_step2_with_rel_pos_value_backward_cuda(N, M, h, hdim, L, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        return grad_attn, grad_v, None, None, None, None, None, grad_table, None

attention_step2_with_rel_pos_value = AttentionStep2WithRelPosValue.apply

class DotProdWithIdxAllTriPlane(Function):
    @staticmethod
    def forward(ctx, q, index_q, index_q_offsets, k, index_k, index_k_offsets, table_q, table_k, rel_idx, n_max, m_offsets):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, 3, h, hdim), table_k: (L, 3, h, hdim), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert q.is_contiguous() and index_q.is_contiguous() and index_q_offsets.is_contiguous() and k.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous()

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L
        assert q.shape[0] == k.shape[0]
        assert L > rel_idx.max(), "L = {}, while rel_idx.max() = {}".format(L, rel_idx.max())
        assert L <= 128

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()

        output = torch.cuda.FloatTensor(h//3, M).zero_()
        
        q_transpose = q.permute(1,2,0).contiguous() #[h, hdim, L]
        k_transpose = k.permute(1,2,0).contiguous() #[h, hdim, L]
        table_q_transpose = table_q.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        table_k_transpose = table_k.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        rel_idx_transpose = rel_idx.permute(1,0).contiguous() #[3, M]

        sptr_cuda.dot_prod_with_idx_all_tri_plane_forward_cuda(N, M, h//3, hdim, n_max, L, q_transpose, index_q, index_q_offsets, k_transpose, index_k, table_q_transpose, table_k_transpose, rel_idx_transpose, m_offsets, output)
        
        output = output.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_all: forward time: {}s, M: {}".format(time.time() - t, M))
        
        ctx.n_max = n_max
        ctx.save_for_backward(q, index_q_offsets, index_q, k, index_k_offsets, index_k, table_q, table_k, rel_idx, m_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, index_q, k, index_k_offsets, index_k, table_q, table_k, rel_idx, m_offsets = ctx.saved_tensors
        M, _ = grad_output.shape
        N, h, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max
        N_k = k.shape[0]
        
        grad_output = grad_output.contiguous()
        assert q.is_contiguous() and index_q_offsets.is_contiguous() and index_q.is_contiguous() and k.is_contiguous() and index_k_offsets.is_contiguous() and index_k.is_contiguous() and table_q.is_contiguous() and table_k.is_contiguous() and rel_idx.is_contiguous() and grad_output.is_contiguous()

        assert L <= 128
        assert 512 % hdim == 0

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, 3, h//3, hdim).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, 3, h//3, hdim).zero_()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        sptr_cuda.dot_prod_with_idx_tri_plane_backward_cuda(N, M, h//3, hdim, n_max, L, grad_output, q, index_q_offsets, k, index_k_offsets, index_k, table_q, table_k, rel_idx, m_offsets, grad_q, grad_k, grad_table_q, grad_table_k)
        
        grad_q_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_k_2 = torch.cuda.FloatTensor(N, h, hdim).zero_()

        sptr_cuda.attention_step1_tri_plane_backward_cuda(N, M, h, hdim, n_max, grad_output, index_q, index_q_offsets, index_k, index_k_offsets, m_offsets, q, k, grad_q_2, grad_k_2)
        
        grad_q += grad_q_2
        grad_k += grad_k_2

        if DEBUG:
            torch.cuda.synchronize()
            print("dot_prod_with_idx_all: backward time: {}s, M: {}".format(time.time() - t, M))
        
        return grad_q, None, None, grad_k, None, None, grad_table_q, grad_table_k, None, None, None

dot_prod_with_idx_all_tri_plane = DotProdWithIdxAllTriPlane.apply

class AttentionStep2WithRelPosValueTriPlane(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index0_offsets, n_max, index1, index1_offsets, table, rel_idx, m_offsets):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, 3, h, hdim), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert attn.is_contiguous() and v.is_contiguous() and index0.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and index1_offsets.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        M, _ = attn.shape
        N, h, hdim = v.shape
        # N = index0_offsets.shape[0] - 1
        L = table.shape[0]

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
            
        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        
        assert hdim == 16
        assert L <= 128

        sptr_cuda.attention_step2_with_rel_pos_value_tri_plane_forward_cuda(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, m_offsets, output)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value: forward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx, m_offsets)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0, index0_offsets, index1, index1_offsets, table, rel_idx, m_offsets = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        N_k = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()

        assert attn.is_contiguous() and v.is_contiguous() and index0_offsets.is_contiguous() and index1.is_contiguous() and grad_output.is_contiguous() and table.is_contiguous() and rel_idx.is_contiguous()

        grad_attn = torch.cuda.FloatTensor(M, h//3).zero_()
        grad_v = torch.cuda.FloatTensor(N_k, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, 3, h//3, hdim).zero_()
        
        table = table.permute(2,3,1,0).contiguous() #[h, hdim, 3, L]
        v = v.permute(1,2,0).contiguous()
        rel_idx = rel_idx.permute(1,0).contiguous()

        if DEBUG:
            torch.cuda.synchronize()
            t = time.time()
        
        sptr_cuda.attention_step2_with_rel_pos_value_tri_plane_backward_cuda(N, M, h//3, hdim, L, n_max, grad_output, index0, index0_offsets, index1, index1_offsets, attn, v, table, rel_idx, m_offsets, grad_attn, grad_v, grad_table)
        
        if DEBUG:
            torch.cuda.synchronize()
            print("attn_step2_with_rel_pos_value: backward time: {}s, M: {}, h: {}".format(time.time() - t, M, h))

        return grad_attn, grad_v, None, None, None, None, None, grad_table, None, None

attention_step2_with_rel_pos_value_tri_plane = AttentionStep2WithRelPosValueTriPlane.apply

