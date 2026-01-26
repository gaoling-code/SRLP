from transformers import T5Config, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Stack, __HEAD_MASK_WARNING_MSG
import copy
import os
import warnings
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)

import numpy as np
import os

def flatten_and_save_to_txt(array, file_path):
    array = array.cpu().numpy()
    bs = array.shape[0]
    flattened_array = array.reshape(bs, -1)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'a') as file:
        np.savetxt(file, flattened_array, fmt='%f')

    print(f"张量已保存到文件：{file_path}")

def info_nce_loss(text_sem, image_sem, temperature=0.07):
    bs, seq_length = text_sem.size(0), text_sem.size(1)
    text_sem = text_sem.view(bs, -1)  
    image_sem = image_sem.view(bs, -1) 

    similarity_scores = torch.matmul(text_sem, image_sem.T) 
    logits = F.softmax(similarity_scores / temperature, dim=1) 

    exp_neg_sum = torch.sum(logits, dim=0) 
    exp_pos = torch.diagonal(logits) 
    losses = exp_pos / exp_neg_sum  
    loss = torch.sum(losses, dim=0) 

    return loss

class T5ForMultimodalGenerationMCCoT(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config, patch_size, padding_idx, save_dir, vot_num,alpha):
        super().__init__(config)
        self.model_dim = config.d_model 
        self.vot_num=vot_num 
        self.alpha=alpha 
        self.padding_idx = padding_idx 

        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.patch_num, self.patch_dim = patch_size 

        self.image_dense = nn.Linear(self.patch_dim, config.d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(1536, config.vocab_size, bias=False)

        self.e_text_modal = nn.Linear(config.d_model, config.d_model, bias=False)
        self.e_text_sem = nn.Linear(config.d_model, config.d_model, bias=False)
        self.d_text = nn.Linear(2*config.d_model, config.d_model, bias=False)

        self.e_image_modal = nn.Linear(config.d_model, config.d_model, bias=False)
        self.e_image_sem = nn.Linear(config.d_model, config.d_model, bias=False)
        self.d_image = nn.Linear(2*config.d_model, config.d_model, bias=False)

        self.mha_layer_hs_ie = torch.nn.MultiheadAttention(embed_dim=config.hidden_size, kdim=config.hidden_size, vdim=config.hidden_size, num_heads=1, batch_first=True)
        self.gate_dense_hs_ie = nn.Linear(2*config.hidden_size, config.hidden_size)

        self.post_init()
        
        self.model_parallel = False
        self.device_map = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, 
        image_ids=None, 
        attention_mask: Optional[torch.FloatTensor] = None, 
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None, 
        model_trained = True, 
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        CoT_ids: Optional[torch.LongTensor] = None, 
        CoT_attention_mask: Optional[torch.FloatTensor] = None, 
        subject=None, 
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache          
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 

        if head_mask is not None and decoder_head_mask is None: 
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None: 
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True, 
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput): 
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0] 
        image_embedding = self.image_dense(image_ids) 


        padding_rows = hidden_states.shape[1] - image_embedding.shape[1] 
        zeros_coord = torch.zeros((image_embedding.size(0), padding_rows, image_embedding.size(2)))
        zeros_coord = zeros_coord.cuda(device=image_embedding.get_device())
        image_embedding = torch.cat((image_embedding, zeros_coord), dim=1) 

        text_modal = self.e_text_modal(hidden_states) 
        text_sem = self.e_text_sem(hidden_states) 
        text_mod_sem = torch.cat([text_modal, text_sem], dim=-1) 
        hidden_states_heat = self.d_text(text_mod_sem) 

        image_modal = self.e_image_modal(image_embedding) 
        image_sem = self.e_image_sem(image_embedding) 
        image_mod_sem = torch.cat([image_modal, image_sem], dim=-1) 
        image_embedding_heat = self.d_image(image_mod_sem) 

        text_sem_for_info_nce = text_sem
        image_sem_for_info_nce = image_sem

        text_sem_att, _ = self.mha_layer_hs_ie(hidden_states, text_sem, text_sem) 
        merge = torch.cat([hidden_states, text_sem_att], dim=-1) 
        gate = self.sigmoid(self.gate_dense_hs_ie(merge)) 
        hidden_states = (1 - gate) * hidden_states + gate * text_sem_att 

        image_att, _ = self.mha_layer(hidden_states, image_embedding, image_embedding) 
        merge = torch.cat([hidden_states, image_att], dim=-1) 
        gate = self.sigmoid(self.gate_dense(merge)) 
        hidden_states = (1 - gate) * hidden_states + gate * image_att 

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels) 

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,  
            encoder_attention_mask=attention_mask, 
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0] 

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)  # [bs, 64, 768]

        lm_logits = self.lm_head(sequence_output) 

        loss = None
        if labels is not None:
            loss_CEL = CrossEntropyLoss(ignore_index=-100)
            loss_MSE = MSELoss(reduction='mean') 
 
            loss_inf= loss_CEL(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)) 
            pp1 = 1
            pp2 = 0.01
            loss_autoencoder = pp1 * loss_MSE(hidden_states, hidden_states_heat) + pp1 * loss_MSE(image_embedding, image_embedding_heat) + pp2 * loss_MSE(text_sem_for_info_nce, image_sem_for_info_nce)

            criterion_TML = nn.TripletMarginLoss(margin=1.0, p=2)
            text_sem_for_info_nce = text_sem_for_info_nce.squeeze(1) 
            image_sem_for_info_nce = image_sem_for_info_nce.squeeze(1) 
            image_sem_for_info_nce_heat = torch.cat([image_sem_for_info_nce[1:], image_sem_for_info_nce[:1]], dim=0)
            loss_TML = criterion_TML(text_sem_for_info_nce, image_sem_for_info_nce, image_sem_for_info_nce_heat) 

            pn1 = 0.1
            pn2 = 0.1
            loss = loss_inf + pn1 * loss_autoencoder + pn2 * loss_TML

            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
