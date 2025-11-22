import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import warnings
from typing import List, Optional, Tuple, Union
from transformers import MBartForConditionalGeneration, T5ForConditionalGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
)
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class LMMultiHead(nn.Module):
    def __init__(self, 
                 model_type,
                 model_path,
                 len_token,
                 ids_remove_motion=None,
                 ids_remove_hand=None,
                 ids_remove_rhand=None,
                 num_heads=3,
                 eos_idx=0
                 ):
        
        super().__init__()
        self.num_heads = num_heads
        self.model_type = model_type
        if 't5' in model_type:
            self.main_lm = T5ForConditionalGeneration.from_pretrained(model_path)
        elif 'mbart' in model_type:
            self.main_lm = MBartForConditionalGeneration.from_pretrained(model_path)
        self.main_lm.resize_token_embeddings(len_token)

        self.ids_remove_motion = ids_remove_motion
        self.ids_remove_hand = ids_remove_hand
        self.ids_remove_rhand = ids_remove_rhand
        self.eos_idx = eos_idx

        self.alpha_hand = 0.4 #1/3
        self.mask_body = torch.zeros(len_token)
        self.mask_body[ids_remove_motion] = float('-inf')
        self.mask_lhand = torch.zeros(len_token)
        self.mask_lhand[ids_remove_hand] = float('-inf')
        self.mask_rhand = torch.zeros(len_token)
        self.mask_rhand[ids_remove_rhand] = float('-inf')
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        labels_hand: Optional[torch.LongTensor] = None,
        labels_rhand: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        if 't5' in self.model_type:
            return self.forward_t5(input_ids,
                                    attention_mask,
                                    decoder_input_ids,
                                    decoder_attention_mask,
                                    head_mask,
                                    decoder_head_mask,
                                    cross_attn_head_mask,
                                    encoder_outputs,
                                    past_key_values,
                                    inputs_embeds,
                                    decoder_inputs_embeds,
                                    labels,
                                    labels_hand,
                                    labels_rhand,
                                    use_cache,
                                    output_attentions,
                                    output_hidden_states,
                                    return_dict
                                )
        elif 'mbart' in self.model_type:
            return self.forward_mbart(input_ids,
                                    attention_mask,
                                    decoder_input_ids,
                                    decoder_attention_mask,
                                    head_mask,
                                    decoder_head_mask,
                                    cross_attn_head_mask,
                                    encoder_outputs,
                                    past_key_values,
                                    inputs_embeds,
                                    decoder_inputs_embeds,
                                    labels,
                                    labels_hand,
                                    labels_rhand,
                                    use_cache,
                                    output_attentions,
                                    output_hidden_states,
                                    return_dict
                                )
        
    def forward_t5(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        labels_hand: Optional[torch.LongTensor] = None,
        labels_rhand: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.main_lm.config.use_cache
        return_dict = return_dict if return_dict is not None else self.main_lm.config.use_return_dict

        encoder = self.main_lm.encoder
        decoder = self.main_lm.decoder

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.main_lm.config.num_layers == self.main_lm.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
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

        # if self.main_lm.model_parallel:
        #     torch.cuda.set_device(self.main_lm.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self.main_lm._shift_right(labels)
            if labels_hand is not None:
                # get decoder inputs from shifting lm labels to the right
                decoder_input_ids_hand = self.main_lm._shift_right(labels_hand)
            if labels_rhand is not None:
                decoder_input_ids_rhand = self.main_lm._shift_right(labels_rhand)

        # Set device for model parallelism
        # if self.main_lm.model_parallel:
        #     torch.cuda.set_device(self.main_lm.decoder.first_device)
        #     hidden_states = hidden_states.to(self.main_lm.decoder.first_device)
        #     if decoder_input_ids is not None:
        #         decoder_input_ids = decoder_input_ids.to(self.main_lm.decoder.first_device)
        #     if attention_mask is not None:
        #         attention_mask = attention_mask.to(self.main_lm.decoder.first_device)
        #     if decoder_attention_mask is not None:
        #         decoder_attention_mask = decoder_attention_mask.to(self.main_lm.decoder.first_device)

        # Decode
        decoder_outputs = decoder(
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

        decoder_outputs_hand = decoder(
            input_ids=decoder_input_ids_hand,
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
        sequence_output_hand = decoder_outputs_hand[0]

        if self.num_heads > 2:
            decoder_outputs_rhand = decoder(
                input_ids=decoder_input_ids_rhand,
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
            sequence_output_rhand = decoder_outputs_rhand[0]

        # Set device for model parallelism
        # if self.main_lm.model_parallel:
        #     torch.cuda.set_device(self.main_lm.encoder.first_device)
        #     self.main_lm.lm_head = self.main_lm.lm_head.to(self.main_lm.encoder.first_device)
        #     sequence_output = sequence_output.to(self.main_lm.lm_head.weight.device)

        if self.main_lm.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            d_model = self.main_lm.model_dim

            sequence_output = sequence_output * (d_model**-0.5)
            sequence_output_hand = sequence_output_hand * (d_model**-0.5)
            if self.num_heads > 2:
                sequence_output_rhand = sequence_output_rhand * (d_model**-0.5)

        lm_logits = self.main_lm.lm_head(sequence_output)
        lm_logits_hand = self.lm_head_hand(sequence_output_hand)
        if self.num_heads > 2:
            lm_logits_rhand = self.lm_head_rhand(sequence_output_rhand)

        loss = loss_hand = loss_rhand = None
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        if labels is not None:
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        
        if labels_hand is not None:
            labels_hand = labels_hand.to(lm_logits_hand.device)
            loss_hand = loss_fct(lm_logits_hand.view(-1, lm_logits_hand.size(-1)), labels_hand.view(-1))
        if labels_rhand is not None:
            labels_rhand = labels_rhand.to(lm_logits_rhand.device)
            loss_rhand = loss_fct(lm_logits_rhand.view(-1, lm_logits_rhand.size(-1)), labels_rhand.view(-1))

        return {'loss': loss,
                'loss_hand': loss_hand,
                'loss_rhand': loss_rhand,
                'output': Seq2SeqLMOutput(
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
                }


    def inference_decoder(
            self,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_input_ids_hand: Optional[torch.LongTensor] = None,
            decoder_input_ids_rhand: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            labels_hand: Optional[torch.LongTensor] = None,
            labels_rhand: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ):

        decoder = self.main_lm.get_decoder()
        token_embeds = decoder.get_input_embeddings()

        if self.model_type in ['mbart_multi']:
            decoder_embeds_body = token_embeds(decoder_input_ids)  #B,T,C
            decoder_embeds_lhand = token_embeds(decoder_input_ids_hand)  #B,T,C
            decoder_embeds_rhand = token_embeds(decoder_input_ids_rhand)  #B,T,C
            decoder_inputs_embeds = (1-2*self.alpha_hand)*decoder_embeds_body + self.alpha_hand*decoder_embeds_lhand + self.alpha_hand*decoder_embeds_rhand
        elif self.model_type in ['mbart_multi_flatten']:
            decoder_inputs_embeds = token_embeds(decoder_input_ids)

        # Decode
        decoder_outputs = decoder(
            input_ids=None,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        lm_logits_body = lm_logits = self.main_lm.lm_head(sequence_output) + self.main_lm.final_logits_bias
        lm_logits_lhand = lm_logits_rhand = None
        if self.model_type in ['mbart_multi']:
            # print(lm_logits.shape, lm_logits.argmax(dim=-1))
            lm_logits_body = lm_logits + self.mask_body.to(lm_logits.device)
            lm_logits_lhand = lm_logits + self.mask_lhand.to(lm_logits.device)
            lm_logits_rhand = lm_logits + self.mask_rhand.to(lm_logits.device)

        return lm_logits_body, lm_logits_lhand, lm_logits_rhand


    def forward_mbart(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        labels_hand: Optional[torch.LongTensor] = None,
        labels_rhand: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.main_lm.config.use_cache
        return_dict = return_dict if return_dict is not None else self.main_lm.config.use_return_dict

        encoder = self.main_lm.get_encoder()

        if labels is not None:
            use_cache = False
            if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, pad_token_id=1)
                decoder_input_ids_hand = decoder_input_ids_rhand = None
                # print('decoder_input_ids: ', decoder_input_ids)
                if labels_hand is not None:
                    # get decoder inputs from shifting lm labels to the right
                    decoder_input_ids_hand = shift_tokens_right(labels_hand, pad_token_id=1)
                    # print('decoder_input_ids_hand: ', decoder_input_ids_hand)
                if labels_rhand is not None:
                    decoder_input_ids_rhand = shift_tokens_right(labels_rhand, pad_token_id=1)
                    # print('decoder_input_ids_rhand: ', decoder_input_ids_rhand)
        
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
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

        lm_logits_body, lm_logits_lhand, lm_logits_rhand = self.inference_decoder(encoder_hidden_states=hidden_states,
                                                                                    decoder_input_ids=decoder_input_ids,
                                                                                    decoder_input_ids_hand=decoder_input_ids_hand,
                                                                                    decoder_input_ids_rhand=decoder_input_ids_rhand,
                                                                                    decoder_attention_mask=decoder_attention_mask,
                                                                                    head_mask=decoder_head_mask,
                                                                                    cross_attn_head_mask=cross_attn_head_mask,
                                                                                    encoder_outputs=encoder_outputs,
                                                                                    past_key_values=past_key_values,
                                                                                    use_cache=use_cache,
                                                                                    output_attentions=output_attentions,
                                                                                    output_hidden_states=output_hidden_states,
                                                                                    return_dict=return_dict,
                                                                                    )

        loss = loss_hand = loss_rhand = torch.tensor(0.0).to(lm_logits_body.device)
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        if labels is not None:
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits_body.device)
            loss = loss_fct(lm_logits_body.view(-1, lm_logits_body.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        
        if labels_hand is not None:
            labels_hand = labels_hand.to(lm_logits_lhand.device)
            loss_hand = loss_fct(lm_logits_lhand.view(-1, lm_logits_lhand.size(-1)), labels_hand.view(-1))
        if labels_rhand is not None:
            labels_rhand = labels_rhand.to(lm_logits_rhand.device)
            loss_rhand = loss_fct(lm_logits_rhand.view(-1, lm_logits_rhand.size(-1)), labels_rhand.view(-1))
        # print('three loss', loss, loss_hand, loss_rhand)

        return {'loss': loss,
                'loss_hand': loss_hand,
                'loss_rhand': loss_rhand,
                # 'output': Seq2SeqLMOutput(
                #             loss=loss,
                #             logits=lm_logits,
                #             past_key_values=decoder_outputs.past_key_values,
                #             decoder_hidden_states=decoder_outputs.hidden_states,
                #             decoder_attentions=decoder_outputs.attentions,
                #             cross_attentions=decoder_outputs.cross_attentions,
                #             encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                #             encoder_hidden_states=encoder_outputs.hidden_states,
                #             encoder_attentions=encoder_outputs.attentions,
                #         )
                }
        

    def generate(
        self,
        inputs=None,
        attention_mask=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=None,
        assistant_model=None,
        streamer=None,
        negative_prompt_ids=None,
        negative_prompt_attention_mask=None,
        decoder_start_token_id=None,
        decoder_start_token_id_hand=None,
        decoder_start_token_id_rhand=None,
        **kwargs,
    ):
        
        max_length = kwargs.get('max_length', 100)
        temperature = kwargs.get('temperature', 1.0)
        do_sample = False #kwargs.get('do_sample', True)

        if 'mbart' in self.model_type:
            encoder = self.main_lm.get_encoder()
        elif 't5' in self.model_type:
            encoder = self.main_lm.encoder

        encoder_outputs = encoder(
                input_ids=inputs,
                attention_mask=attention_mask,
            )
        encoder_hidden_states = encoder_outputs[0]
        
        # outputs_re = self.main_lm.generate(
        #         encoder_outputs=encoder_outputs,
        #         max_length=kwargs.get('max_length', None),
        #         num_beams=kwargs.get('num_beams', None),
        #         do_sample=kwargs.get('do_sample', None),
        #         bad_words_ids=self.ids_remove_motion,
        #         decoder_start_token_id=decoder_start_token_id,
        #     )

        with torch.no_grad():
            decoder_input_ids = decoder_start_token_id
            decoder_input_ids_hand = decoder_start_token_id
            decoder_input_ids_rhand = decoder_start_token_id
            finished = torch.tensor([False]*decoder_start_token_id.shape[0]).to(decoder_start_token_id.device)

            for i in range(max_length):
                # if the sequence context is growing too long we must crop it at block_size
                # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

                # forward the model to get the logits for the index in the sequence
                logits_body, logits_lhand, logits_rhand = self.inference_decoder(encoder_hidden_states=encoder_hidden_states,
                                                            decoder_input_ids=decoder_input_ids,
                                                            decoder_input_ids_hand=decoder_input_ids_hand,
                                                            decoder_input_ids_rhand=decoder_input_ids_rhand,
                                                            )
                # pluck the logits at the final step and scale by desired temperature
                logits_body = logits_body[:, -1, :] / temperature
                if self.model_type == 'mbart_multi_flatten':
                    #body -> lhand -> rhand
                    if i%3 == 0:
                        logits_body = logits_body + self.mask_body.to(logits_body.device)
                    elif i%3 == 1:
                        logits_body = logits_body + self.mask_lhand.to(logits_body.device)
                    elif i%3 == 2:
                        logits_body = logits_body + self.mask_rhand.to(logits_body.device)
                probs_body = F.softmax(logits_body, dim=-1)

                if logits_lhand is not None:
                    logits_lhand = logits_lhand[:, -1, :] / temperature
                    probs_lhand = F.softmax(logits_lhand, dim=-1)
                    logits_rhand = logits_rhand[:, -1, :] / temperature
                    probs_rhand = F.softmax(logits_rhand, dim=-1)
                
                # sample from the distribution
                if do_sample:
                    idx_next_body = torch.multinomial(probs_body, num_samples=1)
                    if logits_lhand is not None:
                        idx_next_lhand = torch.multinomial(probs_lhand, num_samples=1)
                        idx_next_rhand = torch.multinomial(probs_rhand, num_samples=1)
                else:
                    idx_next_body = torch.argmax(probs_body, dim=1, keepdim=True)
                    idx_next_body[finished] = self.eos_idx
                    if logits_lhand is not None:
                        idx_next_lhand = torch.argmax(probs_lhand, dim=1, keepdim=True)
                        idx_next_rhand = torch.argmax(probs_rhand, dim=1, keepdim=True)
                        idx_next_lhand[finished] = idx_next_rhand[finished] = self.eos_idx
                    
                    finished = torch.any(idx_next_body.squeeze(-1) == self.eos_idx, dim=-1)
                    
                # append sampled index to the running sequence and continue
                decoder_input_ids = torch.cat((decoder_input_ids, idx_next_body), dim=1)
                if logits_lhand is not None:
                    decoder_input_ids_hand = torch.cat((decoder_input_ids_hand, idx_next_lhand), dim=1)
                    decoder_input_ids_rhand = torch.cat((decoder_input_ids_rhand, idx_next_rhand), dim=1)

        return {'outputs_re': decoder_input_ids,
                'outputs_hand': decoder_input_ids_hand,
                'outputs_rhand': decoder_input_ids_rhand
                }
