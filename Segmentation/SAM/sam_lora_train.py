import torch.nn as nn
import segment_anything.modeling as sam_modeling
from PIL import Image


class EncoderQKVLora(nn.Module):
    def __init__(self, qkv: nn.Linear, r: int):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features

        self.linear_a_q = nn.Linear(self.dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, self.dim, bias=False)
        self.linear_a_v = nn.Linear(self.dim, r, bias=False)
        self.linear_b_v = nn.Linear(r, self.dim, bias=False)

        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.zeros_(self.linear_b_v.weight)
        nn.init.normal_(self.linear_a_q.weight, std=0.01)
        nn.init.normal_(self.linear_a_v.weight, std=0.01)

    def forward(self, x):
        qkv = self.qkv(x)
        lora_q = self.linear_b_q(self.linear_a_q(x))
        lora_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += lora_q
        qkv[:, :, :, -self.dim:] += lora_v
        return qkv


class DecoderAttentionProjLora(nn.Module):
    def __init__(self, proj: nn.Linear, r: int, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = proj
        self.linear_a_q = nn.Linear(input_dim, r, bias=False)
        self.linear_b_q = nn.Linear(r, output_dim, bias=False)

        nn.init.zeros_(self.linear_b_q.weight)
        nn.init.normal_(self.linear_a_q.weight, std=0.01)

    def forward(self, x):
        return self.proj(x) + self.linear_b_q(self.linear_a_q(x))


class SamLora(nn.Module):
    def __init__(self, sam_model: sam_modeling.Sam, r: int):
        """
        LoRA adaptation for SAM

        Args:
            sam_model (sam_modeling.Sam): Pretrained SAM model.
            r (int): Rank for LoRA adaptation.
        """
        super(SamLora, self).__init__()

        # freeze original SAM model parameters
        for param in sam_model.parameters():
            param.requires_grad = False

        # add LoRA to each image encoder block attention QKV
        for block in sam_model.image_encoder.blocks:
            block.attn.qkv = EncoderQKVLora(block.attn.qkv, r)

        # add LoRA to each mask decoder transformer attention layer
        for layer in sam_model.mask_decoder.transformer.layers:
            for attn in [layer.self_attn, layer.cross_attn_token_to_image, layer.cross_attn_point_to_image]:
                input_dim, output_dim = attn.embedding_dim, attn.embedding_dim
                attn.q_proj = DecoderAttentionProjLora(
                    attn.q_proj, r, input_dim, output_dim)
                attn.v_proj = DecoderAttentionProjLora(
                    attn.v_proj, r, input_dim, output_dim)

        # add LoRA to mask decoder transformer final token to image attention
        decoder_trans_final_attn = sam_model.mask_decoder.transformer.final_attn_token_to_image
        final_attn_input_dim, final_attn_output_dim = decoder_trans_final_attn.embedding_dim, decoder_trans_final_attn.embedding_dim
        decoder_trans_final_attn.q_proj = DecoderAttentionProjLora(
            decoder_trans_final_attn.q_proj, r, final_attn_input_dim, final_attn_output_dim)
        decoder_trans_final_attn.v_proj = DecoderAttentionProjLora(
            decoder_trans_final_attn.v_proj, r, final_attn_input_dim, final_attn_output_dim)

        self.sam_model = sam_model

    def forward(self, batched_input, multimask_output, image_size):
        return self.sam_model(batched_input, multimask_output, image_size)
