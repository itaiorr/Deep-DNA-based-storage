import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any
import numpy as np
import copy
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, repeat

########################################################################
class linear_block(nn.Module):
    def __init__(self, config, input_len, output_len):
        super(linear_block, self).__init__()
        self.fc_1   = nn.Linear(input_len,output_len)
        self.norm_1 = nn.LayerNorm(output_len,elementwise_affine=True)
        self.act_1  = nn.GELU()
        self.dout_1 = nn.Dropout(config.p_dropout)
        self.fc_2   = nn.Linear(output_len,output_len)
        self.norm_2 = nn.LayerNorm(output_len,elementwise_affine=True)
        self.act_2  = nn.GELU()
        self.dout_2 = nn.Dropout(config.p_dropout)
        self.fc_3   = nn.Linear(output_len,output_len)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.norm_1(x)
        x = self.act_1(x)
        x = self.dout_1(x)
        x = self.act_2(self.norm_2(self.fc_2(x)))
        x = self.dout_2(x)
        x = self.fc_3(x)
        return x

########################################################################
class output_module(nn.Module):
    def __init__(self, config, in_ch, out_ch):
        super(output_module, self).__init__()
        
        self.conv_1 = nn.Conv1d(in_ch, in_ch, 1)
        self.conv_2 = nn.Conv1d(in_ch, in_ch, 1)
        self.conv_3 = nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        
        return x

########################################################################
class fusion_module(nn.Module):
    def __init__(self, config):
        super(fusion_module, self).__init__()
        
        label_length = config.label_length
        if config.filter_index:
            label_length -= config.index_length
        
        self.pred_fusion_left  = nn.Parameter(torch.ones(label_length).to(config.device))
        self.pred_fusion_right = nn.Parameter(torch.ones(label_length).to(config.device))
        
        self.conv_1 = nn.Conv1d(config.output_ch, config.output_ch, 1)
        self.conv_2 = nn.Conv1d(config.output_ch, config.output_ch, 1)
        self.conv_3 = nn.Conv1d(config.output_ch, config.output_ch, 1)


    def forward(self, x):
        
        x_left  = x[:x.shape[0]//2,:,:]
        x_right = torch.flip(x[x.shape[0]//2:,:,:],dims=[-1])
        x = (x_left * self.pred_fusion_left + x_right * self.pred_fusion_right) / 2
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x) 
        
        return x, x_left, x_right

########################################################################
class alignement_module(nn.Module):
    def __init__(self, config, in_ch, out_ch):
        super(alignement_module, self).__init__()
        
        self.config = config
        
        self.conv_block_1 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=1)
        self.conv_block_2 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=3, padding=1)
        self.conv_block_3 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=5, padding=2)
        self.conv_block_4 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=7, padding=3)
    
        self.linear_block = linear_block(config, input_len=config.noisy_copies_length, output_len=config.noisy_copies_length)
   
    def forward(self, x):      
        batch, cluster, emb, seq = x.shape
        
        # Get features for each copy
        x = rearrange(x, 'b cluster emb seq -> (b cluster) emb seq')
        x = torch.cat([self.conv_block_1(x), self.conv_block_2(x) ,self.conv_block_3(x), self.conv_block_4(x)],dim=1)
        x = self.linear_block(x)
                        
        # Rearrange back to input ordering
        x = rearrange(x, '(b cluster) emb seq -> b cluster emb seq', b=batch, cluster=cluster)
        
        return x

########################################################################
class embedding_module(nn.Module):
    def __init__(self, config, in_ch, out_ch):
        super(embedding_module, self).__init__()
        
        self.config = config
        
        self.label_length = self.config.label_length
        if self.config.filter_index:
            self.label_length -= self.config.index_length
        
        self.conv_block_1 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=1)
        self.conv_block_2 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=3, padding=1)
        self.conv_block_3 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=5, padding=2)
        self.conv_block_4 = double_conv1D(config, in_ch, out_ch//4, seq_len=config.noisy_copies_length, kernel_size=7, padding=3)
          
        self.linear_block = linear_block(config, input_len=config.noisy_copies_length, output_len=self.label_length)
        
        if config.use_input_scaling:
            self.input_scaling_1 = nn.Parameter(torch.ones(1).to(config.device))
            self.input_scaling_2 = nn.Parameter(torch.ones(1).to(config.device))
                   
    def forward(self, x):
              
        # Sum over cluster dimension (non coherent integration)
        x = torch.sum(x,dim=1)
                
        # Feature extraction
        x = torch.cat([self.conv_block_1(x), self.conv_block_2(x) ,self.conv_block_3(x), self.conv_block_4(x)],dim=1)
        x = self.linear_block(x)
        
        return x

########################################################################
class double_conv1D(nn.Module):
    '''(conv => norm => act) * 2'''
    def __init__(self, config, in_ch, out_ch, seq_len, padding=0,kernel_size=3,stride=1,output_padding=0):
        super(double_conv1D, self).__init__()
        self.conv = nn.Sequential(
            depthwise_separable_conv_1d(in_ch, out_ch, kernels_per_layer=1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LayerNorm(seq_len,elementwise_affine=True),
            nn.GELU(),
            nn.Dropout(config.p_dropout),
            depthwise_separable_conv_1d(out_ch, out_ch, kernels_per_layer=1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LayerNorm(seq_len,elementwise_affine=True),
            nn.GELU(), 
            nn.Dropout(config.p_dropout))

    def forward(self, x):
        x = self.conv(x)
        return x

########################################################################
class depthwise_separable_conv_1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernels_per_layer=1, kernel_size=3, stride=1, padding=0):
        super(depthwise_separable_conv_1d, self).__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch * kernels_per_layer, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch * kernels_per_layer, out_ch, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
                
        return out
        
########################################################################
class net(nn.Module):
    def __init__(self, config):
        super(net, self).__init__()
        
        self.config = config
        
        # Alignment
        self.alignement = alignement_module(config, config.enc_filters, config.alignment_filters)
        
        # Embedding
        self.embedding = embedding_module(config, config.alignment_filters, config.d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, dim_feedforward=config.dim_feedforward,nhead=config.n_head, activation=config.activation, dropout=config.p_dropout)
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers) 
        
        # Output layer
        self.output_module = output_module(config, config.d_model, config.output_ch)
        
        # Fuse left and right branches
        self.fusion = fusion_module(config)

    def forward(self,x): 
        
        ########################################################################
        # Shared weights between left and right branches
        
        data_shape = x.shape
            
        # Alignement module
        x = self.alignement(x)
            
        # Embedding module
        x = self.embedding(x)
            
        # Transformer module
        x = self.encoder(rearrange(x, 'b emb seq -> seq b emb'))
    
        # Output module
        x = self.output_module(rearrange(x, 'seq b emb -> b emb seq'))
            
        # Fusion module
        x, x_left, x_right = self.fusion(x)
            
        ########################################################################
        model_output = {}
        model_output['pred']      = x
        model_output['pred_left']  = x_left
        model_output['pred_right'] = x_right
        
        return model_output
