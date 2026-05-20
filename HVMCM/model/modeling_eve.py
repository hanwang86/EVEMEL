import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class BiGIM(nn.Module):
    """Bidirectional Gated Intra-modal Matching."""

    def __init__(self, cls_dim, tok_dim, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_cls = nn.Linear(cls_dim, hidden_dim)

        self.fc_q_e2m = nn.Linear(tok_dim, hidden_dim)
        self.fc_k_e2m = nn.Linear(tok_dim, hidden_dim)
        self.fc_v_e2m = nn.Linear(tok_dim, hidden_dim)
        self.ln_e2m = nn.LayerNorm(hidden_dim)

        self.fc_q_m2e = nn.Linear(tok_dim, hidden_dim)
        self.fc_k_m2e = nn.Linear(tok_dim, hidden_dim)
        self.fc_v_m2e = nn.Linear(tok_dim, hidden_dim)
        self.ln_m2e = nn.LayerNorm(hidden_dim)

        self.fusion_gate = nn.Parameter(torch.tensor([0.5, 0.0, 0.0]))

        self.gate_temp = nn.Parameter(torch.ones(1))

        self.ln_mention_tokens = nn.LayerNorm(tok_dim)
        self.ln_entity_tokens = nn.LayerNorm(tok_dim)

        self.residual_logit_e2m = nn.Parameter(torch.tensor(-2.0))
        self.residual_logit_m2e = nn.Parameter(torch.tensor(-2.0))


    def forward(
        self,
        mention_cls,
        mention_tokens,
        entity_cls,
        entity_tokens
    ):

        mention_tokens = self.ln_mention_tokens(mention_tokens)
        entity_tokens = self.ln_entity_tokens(entity_tokens)

        m_global = F.normalize(self.fc_cls(mention_cls), dim=-1)   
        e_global = F.normalize(self.fc_cls(entity_cls), dim=-1)    
        g2g_score = torch.matmul(m_global, e_global.t()) 


        alpha_e2m = torch.sigmoid(self.residual_logit_e2m)
        alpha_m2e = torch.sigmoid(self.residual_logit_m2e)

        Q_e = self.fc_q_e2m(entity_tokens)   
        K_m = self.fc_k_e2m(mention_tokens)  
        V_m = self.fc_v_e2m(mention_tokens)  

        attn_e2m = torch.einsum('ejh,mih->emji', Q_e, K_m) / math.sqrt(self.hidden_dim)


        attn_probs_e2m = F.softmax(attn_e2m, dim=-1)  
        saliency_e2m = attn_probs_e2m.mean(dim=2)    

        ctx_e2m = torch.einsum('emi,mih->emh', saliency_e2m, V_m)
        ctx_e2m = ctx_e2m + alpha_e2m * m_global.unsqueeze(0)
        ctx_e2m = F.normalize(self.ln_e2m(ctx_e2m), dim=-1)

        score_e2m = torch.einsum('emh,eh->em', ctx_e2m, e_global).t()  

        Q_m = self.fc_q_m2e(mention_tokens)   
        K_e = self.fc_k_m2e(entity_tokens)    
        V_e = self.fc_v_m2e(entity_tokens)    

        attn_m2e = torch.einsum('mih,ejh->meij', Q_m, K_e) / math.sqrt(self.hidden_dim)

        attn_probs_m2e = F.softmax(attn_m2e, dim=-1)  
        saliency_m2e = attn_probs_m2e.mean(dim=2)     

        ctx_m2e = torch.einsum('mej,ejh->meh', saliency_m2e, V_e)
        ctx_m2e = ctx_m2e + alpha_m2e * e_global.unsqueeze(0)
        ctx_m2e = F.normalize(self.ln_m2e(ctx_m2e), dim=-1)

        score_m2e = torch.einsum('meh,mh->me', ctx_m2e, m_global)     



        fusion_weights = F.softmax(
            self.fusion_gate / self.gate_temp.clamp_min(0.1),
            dim=0
        )

        final_score = (
            fusion_weights[0] * g2g_score +
            fusion_weights[1] * score_e2m +
            fusion_weights[2] * score_m2e
        )

        return final_score


class IntraLevelCrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=query_dim, kdim=kv_dim, vdim=kv_dim,
                                         num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(query_dim)
        self.mha.out_proj.weight.data.zero_()
        self.mha.out_proj.bias.data.zero_()


    def forward(self, text_query, image_kv):

        attn_output, _ = self.mha(query=text_query, key=image_kv, value=image_kv)

        return self.layer_norm(attn_output)



class ExpertRouter(nn.Module):
    def __init__(self, hidden_dim, num_experts=5):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear((num_experts + 1) * hidden_dim, num_experts + 1)

        nn.init.constant_(self.router.weight, 0.0)
        nn.init.constant_(self.router.bias, 0.0)
        self.router.bias.data[0] = 3.0 * math.log(5)

    def forward(self, text_feat, visual_feats):
        router_input = torch.cat([text_feat] + visual_feats, dim=-1)
        router_logits = self.router(router_input)
        return torch.softmax(router_logits / 3.0, dim=-1)


class TokenMoEFusion(nn.Module):

    def __init__(self, hidden_dim, num_experts=5):
        super().__init__()
        self.router = ExpertRouter(hidden_dim, num_experts)

        self.experts_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(num_experts)
        ])

        self.text_expert_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        for expert in self.experts_mlp:
            expert[-1].weight.data.zero_()
            expert[-1].bias.data.zero_()

        self.text_expert_mlp[-1].weight.data.zero_()
        self.text_expert_mlp[-1].bias.data.zero_()

    def forward(self, text_feat, visual_feats):
        router_weights = self.router(text_feat, visual_feats)
        expert_outputs = []

        text_out = self.text_expert_mlp(text_feat)
        expert_outputs.append(text_out.unsqueeze(2))

        for i, expert_module in enumerate(self.experts_mlp):
            expert_in = torch.cat([text_feat, visual_feats[i]], dim=-1)
            out = expert_module(expert_in)
            expert_outputs.append(out.unsqueeze(2))

        expert_outputs = torch.cat(expert_outputs, dim=2)
        final_output = torch.sum(router_weights.unsqueeze(-1) * expert_outputs, dim=2)

        return final_output, router_weights


class HVMCM(nn.Module):
    def __init__(self, text_dim, vision_dim):
        super().__init__()
        self.target_layers = [0, 3, 6, 9, 12]



        self.intra_aggs = nn.ModuleList([
            IntraLevelCrossAttention(query_dim=text_dim, kv_dim=vision_dim)
            for _ in self.target_layers
        ])

        self.token_moe_fusion = TokenMoEFusion(hidden_dim=text_dim, num_experts=len(self.target_layers))
        self.final_norm = nn.LayerNorm(text_dim)
        self.text_layer_norm = nn.LayerNorm(text_dim)
        self.vision_layer_norms = nn.ModuleList([
            nn.LayerNorm(vision_dim) for _ in self.target_layers
        ])

    def forward(self, text_last_hidden, vision_all_hidden_states):
        target_visual_feats = []

        text_last_hidden = self.text_layer_norm(text_last_hidden)

        for idx, layer_idx in enumerate(self.target_layers):
            feat = vision_all_hidden_states[layer_idx]
            feat = self.vision_layer_norms[idx](feat) 
            target_visual_feats.append(feat)

        tva_outputs = []
        for i, layer_feat in enumerate(target_visual_feats):
            out = self.intra_aggs[i](text_last_hidden, layer_feat)
            tva_outputs.append(out)

        fused_output , router_weights= self.token_moe_fusion(text_last_hidden, tva_outputs)

        return self.final_norm(fused_output), router_weights, target_visual_feats


class HVMCMEncoder(nn.Module):
    def __init__(self, args):
        super(HVMCMEncoder, self).__init__()
        self.args = args
        self.clip = CLIPModel.from_pretrained(self.args.pretrained_model)

        text_config = self.clip.config.text_config
        vision_config = self.clip.config.vision_config
        self.text_dim = text_config.hidden_size
        self.vision_dim = vision_config.hidden_size
        self.projection_dim = self.clip.projection_dim

        self.hv_fusion = HVMCM(self.text_dim, self.vision_dim)


        self.embed_dim = self.projection_dim

        self.text_projection = nn.Linear(self.text_dim, self.projection_dim, bias=False)
        with torch.no_grad():
            self.text_projection.weight.copy_(self.clip.text_projection.weight)
        for param in self.text_projection.parameters():
            param.requires_grad = False

        self.gate_fc = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.gate_act = nn.Sigmoid()

        self.global_gate_fc = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.global_gate_act = nn.Sigmoid()

        nn.init.constant_(self.gate_fc.bias, -2.0)
        nn.init.constant_(self.global_gate_fc.bias, -2.0)

    def forward(self, input_ids, attention_mask, pixel_values):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        outputs = self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True
        )

        base_text_embeds = outputs.text_embeds
        base_img_embeds = outputs.image_embeds 

        text_last_hidden = outputs.text_model_output.last_hidden_state
        vision_all_hidden = outputs.vision_model_output.hidden_states
        hv_sequence, router_weights, target_visual_feats = self.hv_fusion(text_last_hidden, vision_all_hidden)

        eot_indices = input_ids.argmax(dim=-1)
        hv_output = hv_sequence[torch.arange(batch_size, device=device), eot_indices]
        hv_embeds_projected = self.text_projection(hv_output)

        vis_weights = router_weights[:, :, 1:]

        vis_weights_global = vis_weights.mean(dim=1)
        vis_weights_global = vis_weights_global.detach()

        vis_weights_normalized = vis_weights_global / (vis_weights_global.sum(dim=-1, keepdim=True) + 1e-9)

        stacked_vis_feats = torch.stack(target_visual_feats, dim=2)
        weight_expanded = vis_weights_normalized.unsqueeze(1).unsqueeze(-1)
        fused_vision_tokens = torch.sum(stacked_vis_feats * weight_expanded, dim=2)

        concat_local = torch.cat([base_text_embeds, hv_embeds_projected], dim=-1)
        alpha = self.gate_act(self.gate_fc(concat_local))
        local_fused = base_text_embeds + alpha * hv_embeds_projected

        concat_global = torch.cat([base_text_embeds, base_img_embeds], dim=-1)
        beta = self.global_gate_act(self.global_gate_fc(concat_global))
        global_fused = base_text_embeds + beta * base_img_embeds

        fused_embeds = (local_fused + global_fused) / 2.0

        fused_embeds = F.normalize(fused_embeds, dim=-1)
        feat_text_norm = F.normalize(base_text_embeds, dim=-1)
        feat_image_norm = F.normalize(base_img_embeds, dim=-1)


        return fused_embeds, feat_text_norm, feat_image_norm, text_last_hidden, fused_vision_tokens, router_weights



