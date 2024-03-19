'''
taken form code of spikformer 
abhi ise snntorch ke saath use karne ke liye modify karna padega
original code used spiking jelley
'''
import torch
import torch.nn as nn
import snntorch as snn

# class SSA(nn.Module):
#     def __init__(self, dim, num_heads=8):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#         self.dim = dim
#         self.num_heads = num_heads
#         self.scale = 0.125
#         self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
#         self.q_lif  = snn.Leaky(beta = 2)

#         self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
#         self.k_lif  = snn.Leaky(beta = 2)

#         self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1,bias=False)
#         self.v_lif  = snn.Leaky(beta = 2)
#         self.attn_lif = snn.Leaky(beta= 2, threshold=0.5)

#         self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
#         self.proj_lif = snn.Leaky(beta= 2)

#     def forward(self, x, 
#                 #res_attn
#                 ):
        
#         mem_q = self.q_lif.init_leaky()
#         mem_k = self.k_lif.init_leaky()
#         mem_v = self.v_lif.init_leaky()
#         mem_attn = self.attn_lif.init_leaky()
#         mem_proj = self.proj_lif.init_leaky()
        
#         T,B,C,H,W = x.shape
#         x = x.flatten(3)
#         T, B, C, N = x.shape
#         x_for_qkv = x.flatten(0, 1)
        
#         q_conv_out = self.q_conv(x_for_qkv).reshape(T,B,C,N).contiguous()
#         q_conv_out, mem_q = self.q_lif(q_conv_out, mem_q)
#         q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

#         k_conv_out = self.k_conv(x_for_qkv).reshape(T,B,C,N).contiguous()
#         k_conv_out, mem_k = self.k_lif(k_conv_out, mem_k)
#         k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

#         v_conv_out = self.v_conv(x_for_qkv).reshape(T,B,C,N).contiguous()
#         v_conv_out, mem_v = self.v_lif(v_conv_out, mem_v)
#         v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

#         x = k.transpose(-2,-1) @ v
#         x = (q @ x) * self.scale

#         x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
#         x, mem_attn = self.attn_lif(x, mem_attn)
#         x = x.flatten(0,1)
#         x, mem_proj = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T,B,C,H,W), mem_proj)
#         return x #v
    

import torch
import torch.nn.functional as F
from config import *

class SSA(nn.Module):
    def __init__(self, d_model, num_heads): #dmodel will be embed dimensions
        super(SSA, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 0.125

        # Linear transformations for query, key, and value
        self.W_q = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1,bias=False)
        self.q_lif  = snn.Leaky(beta = 2, spike_grad=surrogate_grad)

        self.W_k = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1,bias=False)
        self.k_lif  = snn.Leaky(beta = 2, spike_grad=surrogate_grad)

        self.W_v = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1,bias=False)
        self.v_lif  = snn.Leaky(beta = 2, spike_grad=surrogate_grad)

        # Linear transformation for output
        self.W_o = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1)
        self.o_lif = snn.Leaky(beta= 2, threshold=0.5, spike_grad=surrogate_grad)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate the dot product of query and key matrices
        ktv = torch.matmul(K.transpose(-2, -1),V)
        out= torch.matmul(Q,ktv)*self.scale
        
        if mask is not None:
            scores += mask * -1e9
        
        return out
    
    def forward(self, Q, K, V, mask=None):

        mem_q = self.q_lif.init_leaky()
        mem_k = self.k_lif.init_leaky()
        mem_v = self.v_lif.init_leaky()
        mem_o = self.o_lif.init_leaky()


        # Linear transformations
        Q = self.W_q(Q)
        Q, mem_q = self.q_lif(Q, mem_q)
        K = self.W_k(K)
        K, mem_k = self.k_lif(K, mem_k)
        V = self.W_v(V)
        V, mem_v = self.v_lif(V, mem_v)
        
        # Splitting into multiple heads
        Q = Q.view(Q.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        output= self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate attention outputs of all heads and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(output.size(0),self.d_model,-1)
        output = self.W_o(output)
        output, mem_o = self.o_lif(output, mem_o)
        
        return output, mem_o
