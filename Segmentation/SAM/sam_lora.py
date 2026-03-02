import torch
import torch.nn as nn
import segment_anything.modeling as sam_modeling


class EncoderQKVLoRA(nn.Module):
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


class DecoderAttentionProjLoRA(nn.Module):
    def __init__(self, proj: nn.Linear, r: int, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = proj
        self.linear_a = nn.Linear(input_dim, r, bias=False)
        self.linear_b = nn.Linear(r, output_dim, bias=False)

        nn.init.zeros_(self.linear_b.weight)
        nn.init.normal_(self.linear_a.weight, std=0.01)

    def forward(self, x):
        return self.proj(x) + self.linear_b(self.linear_a(x))


class SamLoRA(nn.Module):
    def __init__(self, sam_model: sam_modeling.Sam, r: int, finetune_img_encoder_lora=True,
                 finetune_img_encoder_n_blocks: int = 0,
                 finetune_mask_decoder=True, finetune_prompt_encoder=True):
        super().__init__()

        # freeze image encoder and mask decoder params, leave prompt encoder trainable
        # for automatic segmentation, default embedding in prompt encoder is used, so it needs to be trainable (See MedSAM paper)
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = False

        if not finetune_prompt_encoder:
            for param in sam_model.prompt_encoder.parameters():
                param.requires_grad = False

        if finetune_img_encoder_n_blocks > 0:
            # full fine-tuning of the last N image encoder blocks (no LoRA)
            for block in sam_model.image_encoder.blocks[-finetune_img_encoder_n_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        elif finetune_img_encoder_lora:
            # add LoRA to each image encoder block attention QKV
            for block in sam_model.image_encoder.blocks:
                block.attn.qkv = EncoderQKVLoRA(block.attn.qkv, r)

        # add LoRA to each mask decoder transformer attention layer
        if finetune_mask_decoder:
            for layer in sam_model.mask_decoder.transformer.layers:
                for attn in [layer.self_attn, layer.cross_attn_token_to_image, layer.cross_attn_image_to_token]:
                    input_dim, output_dim = attn.embedding_dim, attn.internal_dim
                    attn.q_proj = DecoderAttentionProjLoRA(
                        attn.q_proj, r, input_dim, output_dim)
                    attn.v_proj = DecoderAttentionProjLoRA(
                        attn.v_proj, r, input_dim, output_dim)

            # add LoRA to mask decoder transformer final token to image attention
            decoder_trans_final_attn = sam_model.mask_decoder.transformer.final_attn_token_to_image
            final_attn_input_dim, final_attn_output_dim = decoder_trans_final_attn.embedding_dim, decoder_trans_final_attn.internal_dim
            decoder_trans_final_attn.q_proj = DecoderAttentionProjLoRA(
                decoder_trans_final_attn.q_proj, r, final_attn_input_dim, final_attn_output_dim)
            decoder_trans_final_attn.v_proj = DecoderAttentionProjLoRA(
                decoder_trans_final_attn.v_proj, r, final_attn_input_dim, final_attn_output_dim)

        self.sam_model = sam_model

    def forward(self, batched_input, multimask_output):
        # return self.sam_model(batched_input, multimask_output)

        input_images = torch.stack(
            [self.sam_model.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.sam_model.image_encoder(input_images)

        outputs = []
        for image_record, curr_embedding in zip(batched_input, image_embeddings):
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.sam_model.postprocess_masks(
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.sam_model.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return None
