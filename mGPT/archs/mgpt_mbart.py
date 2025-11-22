import os, pickle
from typing import List, Union
import numpy as np
import math, json
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import MBartTokenizer
import tokenizers
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb
from mGPT.archs.lm_multihead import LMMultiHead
from mGPT.archs.lm_multihead_part import LMMultiHead_Part


def get_tokens_as_list(tokenizer, word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer([word], add_special_tokens=False).input_ids[0][0]
        tokens_list.append(tokenized_word)
    return tokens_list


def correct_lang_token(tokenizer, input_ids: torch.Tensor, token_length: torch.Tensor, data_src: List[str], part: str, target: bool, model_type: str='mbart_multi'):
    #assign correct language tokens, e.g., en_XX, zh_CN, en_ASL, zh_CSL
    B = input_ids.shape[0]
    x = torch.arange(B).to(input_ids.device)
    y = token_length - 1
    cor_lang = []
    if target:
        if model_type == 'mbart_multi_part':
            src2id = {'how2sign': {'body': 'en_ASL', 'lhand': 'en_ASL_lhand', 'rhand': 'en_ASL_rhand'}, 
                    'csl': {'body': 'zh_CSL', 'lhand': 'zh_CSL_lhand', 'rhand': 'zh_CSL_rhand'},
                    'phoenix': {'body': 'de_DGS', 'lhand': 'de_DGS_lhand', 'rhand': 'de_DGS_rhand'}}
        else:
            src2id = {'how2sign': {'body': 'en_ASL', 'lhand': 'en_ASL', 'rhand': 'en_ASL'}, 
                    'csl': {'body': 'zh_CSL', 'lhand': 'zh_CSL', 'rhand': 'zh_CSL'},
                    'phoenix': {'body': 'de_DGS', 'lhand': 'de_DGS', 'rhand': 'de_DGS'}}
    else:
        src2id = {'how2sign': 'en_XX', 'csl': 'zh_CN', 'phoenix': 'de_DE'}
    for s in data_src:
        if target:
            token = src2id[s][part]
        else:
            token = src2id[s]
        lang_id = tokenizer.convert_tokens_to_ids(token)
        cor_lang.append(lang_id)
    cor_lang = torch.tensor(cor_lang).to(input_ids.device)
    input_ids[x, y] = cor_lang


def make_decoder_input_ids(tokenizer, device, data_src, part, model_type='mbart_multi'):
    decoder_input_ids = []
    if model_type == 'mbart_multi_part':
        src2id = {'how2sign': {'body': 'en_ASL', 'lhand': 'en_ASL_lhand', 'rhand': 'en_ASL_rhand'}, 
                      'csl': {'body': 'zh_CSL', 'lhand': 'zh_CSL_lhand', 'rhand': 'zh_CSL_rhand'},
                      'phoenix': {'body': 'de_DGS', 'lhand': 'de_DGS_lhand', 'rhand': 'de_DGS_rhand'}}
    else:
        src2id = {'how2sign': {'body': 'en_ASL', 'lhand': 'en_ASL', 'rhand': 'en_ASL'}, 
                    'csl': {'body': 'zh_CSL', 'lhand': 'zh_CSL', 'rhand': 'zh_CSL'},
                    'phoenix': {'body': 'de_DGS', 'lhand': 'de_DGS', 'rhand': 'de_DGS'}}
    for s in data_src:
        lang_id = tokenizer.convert_tokens_to_ids(src2id[s][part])
        decoder_input_ids.append(lang_id)
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long, device=device).unsqueeze(-1)
    return decoder_input_ids


class Mbart_Based_MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        hand_codebook_size: int = 0,
        rhand_codebook_size: int = 0,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 256,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        num_heads: int = 2,  #number of lm heads
        **kwargs,
    ) -> None:

        super().__init__()
        assert 'mbart' in model_type
        # Parameters
        self.num_heads = num_heads
        self.m_codebook_size = motion_codebook_size
        self.hand_codebook_size = hand_codebook_size
        self.rhand_codebook_size = rhand_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage

        # Add motion tokens
        self.tokenizer = MBartTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == 'mbart_multi_part':
            new_lang_token = ['en_ASL', 'en_ASL_lhand', 'en_ASL_rhand', 'zh_CSL', 'zh_CSL_lhand', 'zh_CSL_rhand', 'de_DGS', 'de_DGS_lhand', 'de_DGS_rhand']
        else:
            new_lang_token = ['en_ASL', 'zh_CSL', 'de_DGS']
        self.tokenizer.add_tokens(new_lang_token, special_tokens=True)
        all_motion_str = [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)]
        all_hand_str = [f'<hand_id_{i}>' for i in range(self.hand_codebook_size + 3)] if hand_codebook_size>0 else []
        all_rhand_str = [f'<rhand_id_{i}>' for i in range(self.rhand_codebook_size + 3)] if rhand_codebook_size>0 else []
        self.tokenizer.add_tokens(all_motion_str + all_hand_str + all_rhand_str)
        self.lang_token_ids = list(map(self.tokenizer.convert_tokens_to_ids, ['en_XX', 'zh_CN', 'de_DE', '<mask>']+new_lang_token))

        # set map ids
        with open(os.path.join(model_path, 'map_ids.pkl'), 'rb') as f:
            self.tok_id_to_emb_id = pickle.load(f)
        idx = len(self.tok_id_to_emb_id)
        for tok in [*new_lang_token, *all_motion_str, *all_hand_str, *all_rhand_str]:
            tok_id = self.tokenizer.convert_tokens_to_ids(tok)
            self.tok_id_to_emb_id[tok_id] = idx
            idx += 1
        self.emb_id_to_tok_id = {v:k for k,v in self.tok_id_to_emb_id.items()}
        self.eos_idx = self.tok_id_to_emb_id[self.tokenizer.convert_tokens_to_ids('</s>')]

        # restrict output vocab
        tokenizer_with_prefix_space = MBartTokenizer.from_pretrained(model_path, add_prefix_space=True, legacy=True)
        tokenizer_with_prefix_space.add_tokens(new_lang_token, special_tokens=True)
        tokenizer_with_prefix_space.add_tokens(all_motion_str + all_hand_str + all_rhand_str)
        all_motion_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_motion_str)
        all_hand_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_hand_str)
        all_rhand_ids = get_tokens_as_list(tokenizer_with_prefix_space, all_rhand_str)
        ids_remove_motion = list(set(tokenizer_with_prefix_space.get_vocab().values()) - set(all_motion_ids) - set([0,1,2,3]) - set(self.lang_token_ids))
        ids_remove_motion = [[self.tok_id_to_emb_id[x]] for x in ids_remove_motion if x in self.tok_id_to_emb_id]
        # print(ids_remove_motion)
        ids_remove_hand = list(set(tokenizer_with_prefix_space.get_vocab().values()) - set(all_hand_ids) - set([0,1,2,3]) - set(self.lang_token_ids))
        ids_remove_hand = [[self.tok_id_to_emb_id[x]] for x in ids_remove_hand if x in self.tok_id_to_emb_id]
        ids_remove_rhand = list(set(tokenizer_with_prefix_space.get_vocab().values()) - set(all_rhand_ids) - set([0,1,2,3]) - set(self.lang_token_ids))
        ids_remove_rhand = [[self.tok_id_to_emb_id[x]] for x in ids_remove_rhand if x in self.tok_id_to_emb_id]

        # Instantiate language model
        self.model_type = model_type
        if model_type == 'mbart_multi_part':
            self.language_model = LMMultiHead_Part(
                                        model_type,
                                        model_path, 
                                        num_heads=self.num_heads,
                                        len_token=len(self.tok_id_to_emb_id),
                                        ids_remove_motion=ids_remove_motion,
                                        ids_remove_hand=ids_remove_hand,
                                        ids_remove_rhand=ids_remove_rhand
                                    )
        else:
            self.language_model = LMMultiHead(
                                            model_type,
                                            model_path, 
                                            num_heads=self.num_heads,
                                            len_token=len(self.tok_id_to_emb_id),
                                            ids_remove_motion=ids_remove_motion,
                                            ids_remove_hand=ids_remove_hand,
                                            ids_remove_rhand=ids_remove_rhand,
                                            eos_idx=self.eos_idx
                                        )
        self.lm_type = 'encdec'
        # elif model_type == "gpt2":
        #     self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
        #     self.lm_type = 'dec'
        # else:
        #     raise ValueError("type must be either seq2seq or conditional")

        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.num_kws_per_sen = 3 if self.model_type == 'mbart_multi' else 0
        self.max_len_per_part = 10
        self.name2kws = {}
        for split in ['train', 'val', 'test']:
            # load pre-extracted keywords
            with open(f'scripts/name2kws_{split}.json', 'r') as f:
                data = json.load(f)
                self.name2kws.update(data)
        with open('scripts/word2code.json', 'r') as f:
            self.word2code = json.load(f)


    def map_ids(self, input_ids: torch.Tensor, direction: str ='token_to_emb'):
        assert direction in ['token_to_emb', 'emb_to_token']
        if direction == 'token_to_emb':
            mapping = self.tok_id_to_emb_id
        else:
            mapping = self.emb_id_to_tok_id

        B, N = input_ids.shape
        unk_idx = self.tokenizer.convert_tokens_to_ids('<unk>')
        for i in range(B):
            for j in range(N):
                try:
                    input_ids[i][j] = mapping[input_ids[i][j].item()]
                except:
                    input_ids[i][j] = unk_idx


    def get_kw_strings(self, name, src):
        output_strings = []
        for n, sr in zip(name, src):
            cur_string = ""
            if n not in self.name2kws:
                output_strings.append(cur_string)
                continue

            kws = self.name2kws[n][:self.num_kws_per_sen]
            for i in range(len(kws)):
                kw = kws[i]
                # cur_string += f" [{kw}]: "

                if sr in ['how2sign', 'openasl']:
                    cur_string += f" Key word {i+1}, "
                elif sr == 'csl':
                    cur_string += f" 关键词 {i+1}, "
                elif sr == 'phoenix':
                    cur_string += f" Schlüsselwort {i+1}, "
                else:
                    raise NotImplementedError
                
                mo_tokens = self.word2code[kw]['body']
                hand_tokens = self.word2code[kw]['lhand']
                rhand_tokens = self.word2code[kw]['rhand']
                if len(mo_tokens) <= self.max_len_per_part:
                    valid_idx = [_ for _ in range(len(mo_tokens))]
                else:
                    valid_idx = np.linspace(0, len(mo_tokens)-1, self.max_len_per_part, dtype=int)
                mo_tokens = [mo_tokens[i] for i in valid_idx]
                hand_tokens = [hand_tokens[i] for i in valid_idx]
                rhand_tokens = [rhand_tokens[i] for i in valid_idx]

                for mo_t, hand_t, rhand_t in zip(mo_tokens, hand_tokens, rhand_tokens):
                    cur_string += f'<motion_id_{mo_t}><hand_id_{hand_t}><rhand_id_{rhand_t}>'
                    
            output_strings.append(cur_string)
        return output_strings


    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict, src: List[str], name: List[str]):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, lengths, tasks, src, name)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, lengths, tasks)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
        src: List[str],
        name: List[str]
    ):
        # print(src)
        # Tensor to string
        # print('motion tok shape: ', motion_tokens.shape)
        if motion_tokens.ndim == 2:
            motion_strings = self.motion_token_to_string(motion_tokens, lengths, pattern='motion')
        else:
            if self.model_type == 'mbart_multi_flatten':
                motion_strings = []
                for i in range(len(lengths)):
                    cur_len = lengths[i]
                    cur_motion_tokens = motion_tokens[i, :cur_len, :]
                    cur_motion_str = ""
                    for bc, lhc, rhc in zip(cur_motion_tokens[:, 0], cur_motion_tokens[:, 1], cur_motion_tokens[:, 2]):
                        cur_motion_str += f'<motion_id_{int(bc)}><hand_id_{int(lhc)}><rhand_id_{int(rhc)}>'
                    motion_strings.append(cur_motion_str)
                # print(motion_strings)
            else:
                motion_strings = self.motion_token_to_string(motion_tokens[..., 0], lengths, pattern='motion')

        # print('motion_strings: ', motion_strings)
        if self.hand_codebook_size > 0:
            hand_strings = self.motion_token_to_string(motion_tokens[..., 1], lengths, pattern='hand')
            # print('hand_strings: ', hand_strings)
        if self.rhand_codebook_size > 0:
            rhand_strings = self.motion_token_to_string(motion_tokens[..., 2], lengths, pattern='rhand')
            # print('rhand_strings: ', rhand_strings)

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        inputs, outputs = self.template_fulfill(tasks, lengths,
                                                motion_strings, texts, pattern='motion')
        # print('inputs: ', inputs, 'outputs: ', outputs)
        if self.hand_codebook_size > 0:
            outputs_hand = outputs_rhand = None
            _, outputs_hand = self.template_fulfill(tasks, lengths,
                                                hand_strings, texts, pattern='hand')
            # print('outputs_hand: ', outputs_hand)
        if self.rhand_codebook_size > 0:
            _, outputs_rhand = self.template_fulfill(tasks, lengths,
                                            rhand_strings, texts, pattern='rhand')
            # print('outputs_rhand: ', outputs_rhand)

        # keyword motions
        if self.num_kws_per_sen > 0:
            kw_strings = self.get_kw_strings(name, src)
            for i in range(len(inputs)):
                inputs[i] += kw_strings[i]
            # print(inputs)

        # print(texts)
        # train inputs
        source_encoding = self.tokenizer(inputs,
                                         padding='longest',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt",
                                         return_length=True)

        source_attention_mask = source_encoding.attention_mask.to(
            motion_tokens.device)
        source_input_ids = source_encoding.input_ids.to(motion_tokens.device)
        token_len = source_encoding.length.to(motion_tokens.device)
        correct_lang_token(self.tokenizer, source_input_ids, token_len, src, part=None, target=False, model_type=self.model_type)
        self.map_ids(source_input_ids, direction='token_to_emb')
        # print('src input ids: ', source_input_ids)

        # train labels body
        target_inputs = self.tokenizer(outputs,
                                        padding='longest',
                                        max_length=self.max_length,
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True,
                                        return_tensors="pt",
                                        return_length=True)
        labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = target_inputs.attention_mask.to(motion_tokens.device)
        token_len = target_inputs.length.to(motion_tokens.device)
        correct_lang_token(self.tokenizer, labels_input_ids, token_len, src, part='body', target=True, model_type=self.model_type)
        labels_input_ids[labels_input_ids == 0] = -100
        self.map_ids(labels_input_ids, direction='token_to_emb')
        # print('labels: ', labels_input_ids)

        # train labels left hand
        labels_input_ids_hand = labels_input_ids_rhand = None
        if self.hand_codebook_size > 0 and self.model_type != 'mbart_multi_flatten':
            target_inputs_hand = self.tokenizer(outputs_hand,
                                        padding='longest',
                                        max_length=self.max_length,
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True,
                                        return_tensors="pt",
                                        return_length=True)
            labels_input_ids_hand = target_inputs_hand.input_ids.to(motion_tokens.device)
            token_len = target_inputs_hand.length.to(motion_tokens.device)
            correct_lang_token(self.tokenizer, labels_input_ids_hand, token_len, src, part='lhand', target=True, model_type=self.model_type)
            labels_input_ids_hand[labels_input_ids_hand==0] = -100
            self.map_ids(labels_input_ids_hand, direction='token_to_emb')
            # print('labels hand: ', labels_input_ids_hand)

        # train labels right hand
        if self.rhand_codebook_size > 0 and self.model_type != 'mbart_multi_flatten':
            target_inputs_rhand = self.tokenizer(outputs_rhand,
                                    padding='longest',
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_attention_mask=True,
                                    add_special_tokens=True,
                                    return_tensors="pt",
                                    return_length=True)
            labels_input_ids_rhand = target_inputs_rhand.input_ids.to(motion_tokens.device)
            token_len = target_inputs_rhand.length.to(motion_tokens.device)
            correct_lang_token(self.tokenizer, labels_input_ids_rhand, token_len, src, part='rhand', target=True, model_type=self.model_type)
            labels_input_ids_rhand[labels_input_ids_rhand==0] = -100
            self.map_ids(labels_input_ids_rhand, direction='token_to_emb')
            # print('labels rhand: ', labels_input_ids_rhand)

        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
            if condition == 'supervised' else None,
            labels=labels_input_ids,
            labels_hand=labels_input_ids_hand,
            labels_rhand=labels_input_ids_rhand,
            decoder_attention_mask=lables_attention_mask
            if condition == 'supervised' else None,
        )

        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):
        self.tokenizer.padding_side = "right"

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        condition = random.choice(
            ['text', 'motion', 'supervised', 'supervised', 'supervised'])

        if condition == 'text':
            labels = texts
        elif condition == 'motion':
            labels = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            labels = []
            for i in range(len(inputs)):
                labels.append(inputs[i] + ' \n ' + outputs[i] +
                              self.tokenizer.eos_token)

        # Tokenize
        inputs = self.tokenizer(labels,
                                padding='longest',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")

        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        outputs = self.language_model(input_ids=labels_input_ids,
                                      attention_mask=lables_attention_mask,
                                      labels=inputs["input_ids"])

        return outputs

    def generate_direct(self,
                        texts: List[str],
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None,
                        src: List[str] = None,
                        name: List[str] = None):

        # Device
        try:
            self.device = self.language_model.device
        except:
            self.device = self.language_model.main_lm.device

        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + " \n " for text in texts]

        if self.num_kws_per_sen > 0:
            kw_strings = self.get_kw_strings(name, src)
            for i in range(len(texts)):
                texts[i] += kw_strings[i]
            # print(texts)

        source_encoding = self.tokenizer(texts,
                                         padding='longest',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt",
                                         return_length=True)

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)
        token_len = source_encoding.length.to(self.device)
        correct_lang_token(self.tokenizer, source_input_ids, token_len, src, part=None, target=False, model_type=self.model_type)
        self.map_ids(source_input_ids, direction='token_to_emb')
        # print('gen src input: ', source_input_ids)

        # make decoder input ids as language token
        decoder_start_token_id = make_decoder_input_ids(self.tokenizer, self.device, src, part='body', model_type=self.model_type)
        self.map_ids(decoder_start_token_id, direction='token_to_emb')
        decoder_start_token_id_hand = decoder_start_token_id_rhand = None
        if self.hand_codebook_size > 0 and self.model_type != 'mbart_multi_flatten':
            decoder_start_token_id_hand = make_decoder_input_ids(self.tokenizer, self.device, src, part='lhand', model_type=self.model_type)
            self.map_ids(decoder_start_token_id_hand, direction='token_to_emb')
        if self.rhand_codebook_size > 0 and self.model_type != 'mbart_multi_flatten':
            decoder_start_token_id_rhand = make_decoder_input_ids(self.tokenizer, self.device, src, part='rhand', model_type=self.model_type)
            self.map_ids(decoder_start_token_id_rhand, direction='token_to_emb')
        # print('decoder_start_token_id: ', decoder_start_token_id)
        # print('decoder_start_token_id_hand: ', decoder_start_token_id_hand)
        # print('decoder_start_token_id_rhand: ', decoder_start_token_id_rhand)

        
        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
                decoder_start_token_id=decoder_start_token_id,
                decoder_start_token_id_hand=decoder_start_token_id_hand,
                decoder_start_token_id_rhand=decoder_start_token_id_rhand,
            )
        elif self.lm_type == 'dec':
            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                max_new_tokens=max_length)
            self.tokenizer.padding_side = 'left'
        
        outputs_tokens_hand = cleaned_text_hand = outputs_tokens_rhand = cleaned_text_rhand = None
        if 'multi' in self.model_type:
            if self.model_type != 'mbart_multi_flatten':
                # print('ops_re: ', outputs['outputs_re'])
                self.map_ids(outputs['outputs_re'], direction='emb_to_token')
                outputs_string = self.tokenizer.batch_decode(outputs['outputs_re'], skip_special_tokens=True)
                # print('string re: ', outputs_string)
                outputs_tokens, cleaned_text = self.motion_string_to_token(outputs_string, pattern='motion')
                # print('outputs_tokens re: ', outputs_tokens)

                # print('ops_hand: ', outputs['outputs_hand'])
                self.map_ids(outputs['outputs_hand'], direction='emb_to_token')
                outputs_string_hand = self.tokenizer.batch_decode(outputs['outputs_hand'], skip_special_tokens=True)
                # print('string hand: ', outputs_string_hand)
                outputs_tokens_hand, cleaned_text_hand = self.motion_string_to_token(outputs_string_hand, pattern='hand')
                # print('outputs_tokens hand: ', outputs_tokens_hand)
                
                # print('ops_rhand: ', outputs['outputs_rhand'])
                self.map_ids(outputs['outputs_rhand'], direction='emb_to_token')
                outputs_string_rhand = self.tokenizer.batch_decode(outputs['outputs_rhand'], skip_special_tokens=True)
                # print('string rhand: ', outputs_string_rhand)
                outputs_tokens_rhand, cleaned_text_rhand = self.motion_string_to_token(outputs_string_rhand, pattern='rhand')
                # print('outputs_tokens rhand: ', outputs_tokens_rhand)
                # print('lnag token ids: ', self.lang_token_ids)
            else:
                self.map_ids(outputs['outputs_re'], direction='emb_to_token')
                outputs_string_all = self.tokenizer.batch_decode(outputs['outputs_re'], skip_special_tokens=True)
                outputs_string, outputs_string_hand, outputs_string_rhand = [], [], []
                for i in range(len(outputs_string_all)):
                    cur_str = outputs_string_all[i]
                    tokens = cur_str.split(' ')
                    cur_body_str, cur_lhand_str, cur_rhand_str = [], [], []
                    for tok in tokens:
                        if 'motion_id' in tok:
                            cur_body_str.append(tok)
                        elif 'rhand_id' in tok:
                            cur_rhand_str.append(tok)
                        elif 'hand_id' in tok:
                            cur_lhand_str.append(tok)
                    outputs_string.append(' '.join(cur_body_str))
                    outputs_string_hand.append(' '.join(cur_lhand_str))
                    outputs_string_rhand.append(' '.join(cur_rhand_str))
                outputs_tokens, cleaned_text = self.motion_string_to_token(outputs_string, pattern='motion')
                outputs_tokens_hand, cleaned_text_hand = self.motion_string_to_token(outputs_string_hand, pattern='hand')
                outputs_tokens_rhand, cleaned_text_rhand = self.motion_string_to_token(outputs_string_rhand, pattern='rhand')
                # print(outputs_tokens, outputs_tokens_hand, outputs_tokens_rhand)

        else:
            self.map_ids(outputs['outputs_re'], direction='emb_to_token')
            outputs_string = self.tokenizer.batch_decode(outputs['outputs_re'], skip_special_tokens=True)
            # print('string re: ', outputs_string)
            outputs_tokens, cleaned_text = self.motion_string_to_token(outputs_string, pattern='motion')

        return {'outputs_tokens': outputs_tokens,
                'cleaned_text': cleaned_text,
                'outputs_tokens_hand': outputs_tokens_hand,
                'cleaned_text_hand': cleaned_text_hand,
                'outputs_tokens_rhand': outputs_tokens_rhand,
                'cleaned_text_rhand': cleaned_text_rhand
                }

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             hand_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None,
                             src: List[str] = None,
                             name: List[str]=None):

        try:
            self.device = self.language_model.device
        except:
            self.device = self.language_model.main_lm.device

        if task in ["t2m", "m2m", "pred", "inbetween"]:
            assert texts is not None
            motion_strings = [''] * len(texts)
            if not with_len:
                if tasks is None:
                    tasks = [{
                        'input':
                        ['<Caption_Placeholder>'],
                        'output': ['']
                    }] * len(texts)

                lengths = [0] * len(texts)
            else:
                tasks = [{
                    'input': [
                        'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                    ],
                    'output': ['']
                }] * len(texts)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts,
                                                    stage)
            # print(inputs)

            gen_results = self.generate_direct(inputs,
                                                max_length=100,
                                                num_beams=1,
                                                do_sample=True,
                                                src=src,
                                                name=name)

            return gen_results

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(
                motion_tokens, lengths, pattern='motion')
            hand_strings = None
            if hand_tokens is not None and len(hand_tokens) > 0:
                hand_strings = self.motion_token_to_string(
                    hand_tokens, lengths, pattern='hand')

            if not with_len:
                tasks = [{
                    'input': ['Generate text: <Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            gen_results = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
                src=src
                # bad_words_ids=self.bad_words_ids
            )
            return gen_results["cleaned_text"]

    def motion_token_to_string(self, motion_token, lengths: List[int], pattern: str = 'motion'):
        motion_string = []
        for i in range(len(motion_token)):
            if type(motion_token[i]) == List:
                motion_list = motion_i[:lengths[i]]
            else:
                motion_i = motion_token[i].cpu(
                ) if motion_token[i].device.type == 'cuda' else motion_token[i]
                # print(motion_i, lengths)
                motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (''.join([f'<{pattern}_id_{int(i)}>' for i in motion_list])))
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor, pattern: str = 'motion'):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (''.join([f'<{pattern}_id_{int(i)}>' for i in motion_list])))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str], pattern: str = 'motion'):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = ''.join(motion_string[i].split(' '))
            string_list = string.split('><')
            try:
                token_list = [
                    int(i.split('_')[-1].replace('>', ''))
                    for i in string_list
                ]
            except:
                token_list = [0]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str, text: str, pattern: str = 'motion'):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        token_length = length / self.down_t
        predict_head = int(token_length * self.predict_ratio + 1)
        masked_head = int(token_length * self.inbetween_ratio + 1)
        masked_tail = int(token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><{pattern}_id_{self.m_codebook_size+1}>'
        motion_predict_last = f'<{pattern}_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:])

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<{pattern}_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        # if random.random() < self.quota_ratio:
            # text = f'\"{text}\"'

        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>',
            motion_string).replace('<Frame_Placeholder>', f'{length}').replace(
                '<Second_Placeholder>', '%.1f' % seconds).replace(
                    '<Motion_Placeholder_s1>', motion_predict_head).replace(
                        '<Motion_Placeholder_s2>',
                        motion_predict_last).replace(
                            '<Motion_Placeholder_Masked>', motion_masked)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         stage='test',
                         pattern='motion'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i], pattern=pattern))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i], pattern=pattern))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr, pattern='motion'):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<{pattern}_id_{self.m_codebook_size}><{pattern}_id_0><{pattern}_id_{self.m_codebook_size+1}>'

        return f'<{pattern}_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<{pattern}_id_{self.m_codebook_size+1}>'

    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
