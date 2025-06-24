from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, PatchEmbed, trunc_normal_, Mlp
from torchdiffeq import odeint, odeint_adjoint



class ODEFuncTransformer(nn.Module):
    def __init__(self, dim):
        super(ODEFuncTransformer, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.activation = nn.ReLU()

    def forward(self, t, x):
        residual = x  
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.norm(x + residual)  
        return x


class ODEBlock(nn.Module):
    def __init__(self, ode_func, tol=1e-3, solver='rk4'):
        super(ODEBlock, self).__init__()
        self.ode_func = ode_func
        self.tol = tol
        self.solver = solver

    def forward(self, x, lead_times):
        init_time = 0
        final_time = lead_times.max().item()
        resolution = 0.01
        t = torch.arange(init_time, final_time+resolution, step=resolution).to(x.device)   
        trajectory = odeint(self.ode_func, x, t, rtol=self.tol, atol=self.tol, method=self.solver)

        final_outputs = []

        for i in range(len(lead_times)):
            time_idx = int(lead_times[i].item()/resolution)  
            final_outputs.append(trajectory[time_idx, i])

        final_outputs = torch.stack(final_outputs)  
        return final_outputs

class ContinuousAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_vars, dropout=0.1):
        super(ContinuousAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_vars = num_vars
        self.dropout = nn.Dropout(dropout)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  

    def forward(self, x):
        B, N, C = x.shape  
        H = self.num_heads
        head_dim = C // H

        qkv = self.qkv_proj(x).reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        patch_attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5) 
        patch_attn_scores = F.softmax(patch_attn_scores, dim=-1)
        patch_attn_scores = self.dropout(patch_attn_scores)
        patch_attn_output = torch.matmul(patch_attn_scores, v) 
        
        # ---- Cross-Image Attention (Between Images) ---- #
        q_flat = q.permute(2, 0, 1, 3).reshape(N, B * H, head_dim)  
        k_flat = k.permute(2, 0, 1, 3).reshape(N, B * H, head_dim)  
        v_flat = v.permute(2, 0, 1, 3).reshape(N, B * H, head_dim)  

        cross_image_attn_scores = torch.matmul(q_flat, k_flat.transpose(-2, -1)) / (head_dim ** 0.5)  
        cross_image_attn_scores = F.softmax(cross_image_attn_scores, dim=-1)
        cross_image_attn_scores = self.dropout(cross_image_attn_scores)
        cross_image_attn_output = torch.matmul(cross_image_attn_scores, v_flat) 
        cross_image_attn_output = cross_image_attn_output.reshape(N, B, H, head_dim).permute(1, 2, 0, 3)  

        attn_output = patch_attn_output + cross_image_attn_output 
        attn_output = patch_attn_output.permute(0, 2, 1, 3).reshape(B, N, C) 
        attn_output = self.out_proj(attn_output)

        return attn_output


class ODETransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_vars, drop_rate=0.1, drop_path=0.0, solver='rk4'):
        super(ODETransformerBlock, self).__init__()
        
        self.attn = ContinuousAttention(embed_dim, num_heads, num_vars, dropout=drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ode_func = ODEFuncTransformer(embed_dim)
        self.ode_block = ODEBlock(self.ode_func, solver=solver)
        
    def forward(self, x, lead_times):
        attn_out = self.attn(x)
        x = x + attn_out
        x = self.norm1(x)

        x = x + self.ode_block(x, lead_times)
        x = self.norm2(x)
        
        return x

class ConViTCast(nn.Module):
    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        depth=8,
        decoder_depth=2,
        num_heads=16,
        drop_path=0.1,
        drop_rate=0.1,
        solver='rk4'
    ):
        super().__init__()
        
        if img_size[0] % patch_size != 0:
            pad_size = patch_size - img_size[0] % patch_size
            img_size = (img_size[0] + pad_size, img_size[1])

        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = default_vars

        self.token_embeds = nn.ModuleList(
            [PatchEmbed(img_size, patch_size, 1, embed_dim) for _ in range(len(default_vars))]
        )
        self.num_patches = self.token_embeds[0].num_patches

        self.var_embed, self.var_map = self.create_var_embedding(embed_dim)
        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        self.lead_time_embed = nn.Linear(1, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                ODETransformerBlock(
                    embed_dim,
                    num_heads,
                    num_vars=len(default_vars),
                    drop_rate=drop_rate,
                    drop_path=dpr[i],
                    solver=solver
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.default_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)

        self.initialize_weights()
    
    
    def get_2d_sincos_pos_embed(self, embed_dim, grid_size_h, grid_size_w, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size_h, dtype=np.float32)
        grid_w = np.arange(grid_size_w, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed


    def get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb


    def get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.0
        omega = 1.0 / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb


    def initialize_weights(self):
        pos_embed = self.get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        var_embed = self.get_1d_sincos_pos_embed_from_grid(self.var_embed.shape[-1], np.arange(len(self.default_vars)))
        self.var_embed.data.copy_(torch.from_numpy(var_embed).float().unsqueeze(0))

      
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    @lru_cache(maxsize=None)
    def get_var_ids(self, vars, device):
        ids = np.array([self.var_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        p = self.patch_size
        c = len(self.default_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_variables(self, x: torch.Tensor):
        b, _, l, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)

        var_query = self.var_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.var_agg(var_query, x, x)
        x = x.squeeze()
        x = x.unflatten(dim=0, sizes=(b, l))
        return x
    
    def pad(self, x: torch.Tensor):
        h = x.shape[-2]
        if h % self.patch_size != 0:
            pad_size = self.patch_size - h % self.patch_size
            padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
        else:
            padded_x = x
            pad_size = 0
        return padded_x, pad_size

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
    
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)

        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)

        x = self.aggregate_variables(x)
        x = x + self.pos_embed

        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, lead_times)
        x = self.norm(x)

        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat):
        padded_x, pad_size = self.pad(x)
        out_transformers = self.forward_encoder(padded_x, lead_times, variables)
        preds = self.head(out_transformers)
        preds = self.unpatchify(preds)[:, :, pad_size:]
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix):
        _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat)
        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
