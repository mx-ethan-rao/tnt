# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F


from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

def random_sim_prune(importance_map, token_embeddings, r, use_tokens):
    batch_size, num_tokens, embed_size = token_embeddings.shape
    
    # Function to perform one pruning iteration
    def prune_step(token_embeddings, importance_map, r):
        # Step 1: Random partition into Group A and Group B
        rand_perm = torch.randperm(num_tokens, device=token_embeddings.device)
        group_a_indices = rand_perm[:num_tokens // 2]  # Randomly select half for Group A
        group_b_indices = rand_perm[num_tokens // 2:]  # The other half for Group B
        
        group_a = torch.gather(token_embeddings, 1, group_a_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, embed_size))
        group_b = torch.gather(token_embeddings, 1, group_b_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, embed_size))
        
        remaining_indices_a = group_a_indices.unsqueeze(0).expand(batch_size, -1)
        # Step 3: Compute cosine similarity between Group A and Group B
        # Normalize embeddings to unit vectors
        group_b_norm = F.normalize(group_b, p=2, dim=-1)  # [batch_size, half, embed_size]
        group_a_norm = F.normalize(group_a, p=2, dim=-1)  # [batch_size, num_tokens - half, embed_size]

        # Compute cosine similarity: [batch_size, num_tokens - half, half]
        similarities = torch.bmm(group_a_norm, group_b_norm.transpose(1, 2))  

        # Step 4: Identify the top `r` most similar tokens in Group A to prune
        # For each token in Group A, find the maximum similarity with any token in Group B
        max_similarities, _ = similarities.max(dim=2)  # [batch_size, num_tokens - half]

        # Get the indices of the top `r` similar tokens to prune
        topk = torch.topk(max_similarities, k=r, dim=1, largest=True, sorted=False)
        prune_indices = topk.indices  # [batch_size, r]

        # Create a mask to keep tokens not selected for pruning
        mask = torch.ones_like(group_a_norm[:, :, 0], dtype=torch.bool)  # [batch_size, num_tokens - half]
        batch_indices = torch.arange(batch_size).unsqueeze(1).to(token_embeddings.device)
        mask[batch_indices, prune_indices] = False  # Set pruned tokens to False

        # Apply the mask across the entire batch to get pruned indices
        pruned_indices_a = remaining_indices_a[mask].view(batch_size, -1)  # Reshape back to batch size after pruning

        # Combine remaining tokens from Group A and all tokens from Group B
        group_b_indices_expanded = group_b_indices.unsqueeze(0).repeat(batch_size, 1)  # Repeat group_b_indices for all batches
        combined_group_indices = torch.cat((pruned_indices_a, group_b_indices_expanded), dim=1)
        pruned_token_embeddings = torch.gather(token_embeddings, 1, combined_group_indices.unsqueeze(-1).expand(-1, -1, embed_size))
        pruned_importance_map = torch.gather(importance_map, 1, combined_group_indices)  # Keep importance map in sync
        
        return pruned_token_embeddings, pruned_importance_map
    
    # Perform pruning twice
    pruned_token_embeddings, pruned_importance_map = prune_step(token_embeddings, importance_map, r)
    
    # Step 5: Sort pruned tokens by their importance and return top `use_tokens`
    final_sorted_indices = torch.argsort(pruned_importance_map, dim=1, descending=True)
    final_pruned_embeddings = torch.gather(pruned_token_embeddings, 1, final_sorted_indices.unsqueeze(-1).expand(-1, -1, embed_size))
    final_output = final_pruned_embeddings[:, :use_tokens, :]
    
    return final_output

def sequential_sim_prune(importance_map, token_embeddings, r, use_tokens):
    """
    Prunes tokens based on similarity and importance.

    Args:
        importance_map (torch.Tensor): [batch_size, num_tokens]
        token_embeddings (torch.Tensor): [batch_size, num_tokens, embed_size]
        r (int): Number of tokens to drop in each iteration.
        use_tokens (int): Number of tokens to retain after pruning.

    Returns:
        torch.Tensor: Pruned token embeddings of shape [batch_size, use_tokens, embed_size]
    """
    batch_size, num_tokens, embed_size = token_embeddings.shape

    # Step 1: Sort tokens by importance in descending order
    sorted_indices = torch.argsort(importance_map, dim=1, descending=True)  # [batch_size, num_tokens]
    sorted_importance_map = torch.gather(importance_map, 1, sorted_indices)  # [batch_size, num_tokens]
    sorted_token_embeddings = torch.gather(
        token_embeddings, 
        1, 
        sorted_indices.unsqueeze(-1).expand(-1, -1, embed_size)
    )  # [batch_size, num_tokens, embed_size]

    def prune_step(sorted_tokens, sorted_importance, r):
        """
        Performs one pruning iteration.

        Args:
            sorted_tokens (torch.Tensor): [batch_size, num_tokens, embed_size]
            sorted_importance (torch.Tensor): [batch_size, num_tokens]
            r (int): Number of tokens to drop.

        Returns:
            torch.Tensor: Pruned tokens after one iteration.
        """
        # Step 2: Partition tokens into Group B (more important) and Group A (less important)
        half = num_tokens // 2
        group_b = sorted_tokens[:, :half, :]  # [batch_size, half, embed_size]
        group_a = sorted_tokens[:, half:, :]  # [batch_size, num_tokens - half, embed_size]

        # Step 3: Compute cosine similarity between Group A and Group B
        # Normalize embeddings to unit vectors
        group_b_norm = F.normalize(group_b, p=2, dim=-1)  # [batch_size, half, embed_size]
        group_a_norm = F.normalize(group_a, p=2, dim=-1)  # [batch_size, num_tokens - half, embed_size]

        # Compute cosine similarity: [batch_size, num_tokens - half, half]
        similarities = torch.bmm(group_a_norm, group_b_norm.transpose(1, 2))  

        # Step 4: Identify the top `r` most similar tokens in Group A to prune
        # For each token in Group A, find the maximum similarity with any token in Group B
        max_similarities, _ = similarities.max(dim=2)  # [batch_size, num_tokens - half]

        # Get the indices of the top `r` similar tokens to prune
        topk = torch.topk(max_similarities, k=r, dim=1, largest=True, sorted=False)
        prune_indices = topk.indices  # [batch_size, r]

        # Create a mask to keep tokens not selected for pruning
        mask = torch.ones_like(group_a_norm[:, :, 0], dtype=torch.bool)  # [batch_size, num_tokens - half]
        batch_indices = torch.arange(batch_size).unsqueeze(1).to(token_embeddings.device)
        mask[batch_indices, prune_indices] = False  # Set pruned tokens to False

        # Apply mask to Group A
        pruned_group_a = group_a[mask].view(batch_size, -1, embed_size)  # [batch_size, num_tokens - half - r, embed_size]

        # Step 5: Combine Group B and pruned Group A
        pruned_tokens = torch.cat((group_b, pruned_group_a), dim=1)  # [batch_size, num_tokens - r, embed_size]

        # Update the sorted_indices accordingly for further pruning if needed
        return pruned_tokens

    # Perform pruning twice
    pruned_token_embeddings = prune_step(sorted_token_embeddings, sorted_importance_map, r)

    # After pruning, select the top `use_tokens` tokens
    final_pruned_embeddings = pruned_token_embeddings[:, :use_tokens, :]  # [batch_size, use_tokens, embed_size]

    return final_pruned_embeddings

class DistilledVisionTransformerWithTNT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        #################################
        # Parameters introduced: Add alpha heads to produce noise signal term
        self.alpha_norm = kwargs['norm_layer'](self.embed_dim)
        self.alpha_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, 1) for _ in range(kwargs['depth'])
        ])
        self.pru_stat = None
        self.single_l = None
        #################################

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        #################################
        if self.training:
            # Noise allocator: To add noise to token embeddings at 1-5 layers while fine-tuning
            for i, (blk, alpha_head) in enumerate(zip(self.blocks, self.alpha_heads)):
                x = blk(x)
                if i < 5:
                    alpha = alpha_head(x[:, 2:])
                    alpha = torch.softmax(alpha.squeeze(-1), dim=-1)
                    alpha = 1 - alpha
                    noise = torch.randn_like(x[:, 2:]) * alpha.unsqueeze(-1).repeat(1,1,x.size(-1))
                    zero_noise = torch.zeros_like(cls_tokens).repeat(1,2,1)
                    noise = torch.cat((zero_noise, noise), dim=1)
                    x = self.alpha_norm(x)
                    x = x + 0.02 * noise
        else:
            if not self.single_l:
                alpha = torch.rand((x.size(0), x.size(1) - 2)).to(x.device)
                x_keep = random_sim_prune(alpha, x[:, 2:], r=self.pru_stat['sim_pru'], use_tokens=self.pru_stat['after_sim_pru'])
                x = torch.cat((x[:, :2], x_keep), dim=1)

            for i, (blk, alpha_head, rate) in enumerate(zip(self.blocks, self.alpha_heads, self.pru_stat['info_pru'])):
                x = blk(x)
                if rate != 1.:
                    num_tokens = int((x.size(1)-2) * rate)
                    alpha = alpha_head(x[:, 2:])
                    alpha = torch.softmax(alpha.squeeze(-1), dim=-1)
                    if self.single_l:
                        top_k_indices = torch.topk(alpha, k=num_tokens+self.pru_stat['sim_pru'], dim=1, largest=True).indices  # [B, top_k]
                        # Gather the top k tokens
                        x_keep = torch.gather(x[:, 2:], 1, top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [B, top_k, hidden_dim]
                        alpha = torch.gather(alpha, 1, top_k_indices)
                        x_keep = random_sim_prune(alpha, x_keep, r=self.pru_stat['sim_pru'], use_tokens=num_tokens)
                    else:
                        top_k_indices = torch.topk(alpha, k=num_tokens, dim=1, largest=True).indices  # [B, top_k]
                        # Gather the top k tokens
                        x_keep = torch.gather(x[:, 2:], 1, top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [B, top_k, hidden_dim]
                    x = torch.cat((x[:, :2], x_keep), dim=1)
        #################################


        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2
        

class VisionTransformerWithTNT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #################################
        # Parameters introduced: Add alpha heads to produce noise signal term
        self.alpha_norm = kwargs['norm_layer'](self.embed_dim)
        self.alpha_heads = nn.ModuleList([
            nn.Linear(self.embed_dim, 1) for _ in range(kwargs['depth'])
        ])
        self.pru_stat = None
        self.single_l = None
        #################################

        self.alpha_heads.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        #################################
        if self.training:
            # Noise allocator: To add noise to token embeddings at 1-5 layers while fine-tuning
            for i, (blk, alpha_head) in enumerate(zip(self.blocks, self.alpha_heads)):
                x = blk(x)
                if i < 5:
                    alpha = alpha_head(x[:, 1:])
                    alpha = torch.softmax(alpha.squeeze(-1), dim=-1)
                    alpha = 1 - alpha
                    noise = torch.randn_like(x[:, 1:]) * alpha.unsqueeze(-1).repeat(1,1,x.size(-1))
                    zero_noise = torch.zeros_like(cls_tokens)
                    noise = torch.cat((zero_noise, noise), dim=1)
                    x = self.alpha_norm(x)
                    x = x + 0.02 * noise
        else:
            if not self.single_l:
                alpha = torch.rand((x.size(0), x.size(1) - 1)).to(x.device)
                x_keep = random_sim_prune(alpha, x[:, 1:], r=self.pru_stat['sim_pru'], use_tokens=self.pru_stat['after_sim_pru'])
                x = torch.cat((x[:, :1], x_keep), dim=1)

            for i, (blk, alpha_head, rate) in enumerate(zip(self.blocks, self.alpha_heads, self.pru_stat['info_pru'])):
                x = blk(x)
                if rate != 1.:
                    num_tokens = int((x.size(1)-1) * rate)
                    alpha = alpha_head(x[:, 1:])
                    alpha = torch.softmax(alpha.squeeze(-1), dim=-1)
                    if self.single_l:
                        top_k_indices = torch.topk(alpha, k=num_tokens+self.pru_stat['sim_pru'], dim=1, largest=True).indices  # [B, top_k]
                        # Gather the top k tokens
                        x_keep = torch.gather(x[:, 1:], 1, top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [B, top_k, hidden_dim]
                        alpha = torch.gather(alpha, 1, top_k_indices)
                        x_keep = random_sim_prune(alpha, x_keep, r=self.pru_stat['sim_pru'], use_tokens=num_tokens)
                    else:
                        top_k_indices = torch.topk(alpha, k=num_tokens, dim=1, largest=True).indices  # [B, top_k]
                        # Gather the top k tokens
                        x_keep = torch.gather(x[:, 1:], 1, top_k_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))  # [B, top_k, hidden_dim]
                    x = torch.cat((x[:, :1], x_keep), dim=1)
        #################################

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformerWithTNT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformerWithTNT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformerWithTNT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformerWithTNT(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformerWithTNT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformerWithTNT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformerWithTNT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformerWithTNT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
