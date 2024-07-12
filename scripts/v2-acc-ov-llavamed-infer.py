# Standard library imports
import collections
import csv
import json
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from datasets import load_dataset
import torch
import openvino as ov
from openvino.runtime import opset13
import nncf

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava import LlavaLlamaModel
from transformers import (
    AutoConfig,
    AutoTokenizer,
    CLIPImageProcessor,
    StoppingCriteria,
    TextStreamer,
)
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

    
class OVLlavaMedForCausalLM(GenerationMixin):
    def __init__(self, core, model_dir, device, use_im_start_end=True, im_patch_token=32001, im_start_token=32002, im_end_token=32003):
        self.image_encoder = core.compile_model(model_dir / "int8_image_encoder.xml", device)
        self.token_embed = core.compile_model(model_dir / "token_embed.xml", device)
        self.model = core.read_model(model_dir / "llava_with_past.xml")
        self.input_names = {
            key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)
        }
        self.output_names = {
            idx: key for idx, key in enumerate(self.model.outputs)
        }
        self.key_value_input_names = [
            key for key in list(self.input_names)[1:] if key != "beam_idx"
        ]
        self.key_value_output_names = [
            key for key in list(self.output_names)[1:]
        ]
        self.stateful = len(self.key_value_input_names) == 0
        compiled_model = core.compile_model(self.model, device)
        self.request = compiled_model.create_infer_request()
        self.config = AutoConfig.from_pretrained(model_dir)
        self.generation_config = GenerationConfig.from_model_config(self.config)
        self.main_input_name = "input_ids"
        self.device = torch.device("cpu")
        self.num_pkv = 2
        self.use_im_start_end = use_im_start_end,
        self.im_patch_token = im_patch_token
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.next_beam_idx = None

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True

    def __call__(
        self,
        input_ids: torch.LongTensor,
        images: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        prefix_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        return self.forward(
            input_ids, images, attention_mask, prefix_mask, past_key_values
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        images: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        prefix_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """General inference method"""
        inputs = {}
        if past_key_values is not None:
            inputs = {}
            if not self.stateful:
                past_key_values = tuple(
                    past_key_value
                    for pkv_per_layer in past_key_values
                    for past_key_value in pkv_per_layer
                )
                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))
            input_ids = np.array(input_ids)[:, -1:]
            inputs_embeds = self.token_embed(input_ids)[0]
            inputs["inputs_embeds"] = inputs_embeds
            if "beam_idx" in self.input_names:
                inputs["beam_idx"] = (
                    self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
                )
        else:
            inputs = self.prepare_multimodal_input(
            input_ids, images, attention_mask
            )

        # Run inference
        global first_token_latency
        global other_tokens_latency_list
        # Run inference
        start_time = time.time()
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()
        end_time = time.time()
        
        if past_key_values is None:
            first_token_latency = end_time - start_time
        else:
            other_tokens_latency_list.append(end_time - start_time)
            
        logits = torch.from_numpy(self.request.get_tensor(self.output_names[0]).data)

        if not self.stateful:

            # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
            past_key_values = tuple(
                self.request.get_tensor(key).data for key in self.key_value_output_names
            )
            # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
            past_key_values = tuple(
                past_key_values[i : i + self.num_pkv]
                for i in range(0, len(past_key_values), self.num_pkv)
            )
        else:
            past_key_values = ((),)
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


    def prepare_multimodal_input(self, input_ids, images, attention_mask):
        """Preprocessing function for embedding multimodal data"""
        inputs = {}
        inputs_embeds = self.token_embed(input_ids)[0]
        batch_size = input_ids.shape[0]
        if not self.stateful:
            for input_name in self.key_value_input_names:
                model_inputs = self.model.input(input_name)
                shape = model_inputs.get_partial_shape()
                shape[0] = batch_size
                if shape[2].is_dynamic:
                    shape[2] = 0
                else:
                    shape[1] = 0
                inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())
        else:
            self.request.reset_state()
            # Set initial value for the next beam_idx input that will be used at the current iteration
            # and will be optionally updated by _reorder_cache at the next iterations if beam_search is used
            self.next_beam_idx = np.arange(batch_size, dtype=int)

        if images is None:
            inputs["inputs_embeds"] = inputs_embeds
            if "beam_idx" in self.input_names:
                inputs["beam_idx"] = (
                    self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
                )
            return inputs
        res = self.image_encoder(images)
        image_features = res[0]
        dummy_image_features = res[1]

        new_input_embeds = []
        cur_image_idx = 0
        for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
            if (cur_input_ids == self.im_patch_token).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                new_input_embeds.append(cur_input_embeds)
                continue
            if self.use_im_start_end:
                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == self.im_start_token).sum() != (cur_input_ids == self.im_end_token).sum():
                    raise ValueError("The number of image start tokens and image end tokens should be the same.")
                image_start_tokens = np.where(cur_input_ids == self.im_start_token)[0]

                for image_start_token_pos in image_start_tokens:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    
                    if cur_input_ids[image_start_token_pos + num_patches + 1] != self.im_end_token:
                        raise ValueError("The image end token should follow the image start token.")
                    cur_new_input_embeds = np.concatenate(cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:], dim=0)
                    cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
            else:
                cur_image_features = image_features[cur_image_idx]
                num_patches = cur_image_features.shape[0]
                if (cur_input_ids == self.im_patch_token).sum() != num_patches:
                    raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = np.where(cur_input_ids == self.im_patch_token)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != np.arange(mask_index_start, mask_index_start+num_patches, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                cur_new_input_embeds = np.concatenate((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), axis=0)
                new_input_embeds.append(cur_new_input_embeds)
        inputs_embeds = np.stack(new_input_embeds, axis=0)
        inputs["inputs_embeds"] = inputs_embeds
        if "beam_idx" in self.input_names:
            inputs["beam_idx"] = (
                self.next_beam_idx if self.next_beam_idx is not None else np.arange(batch_size, dtype=int)
            )
        return inputs


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        This function is used during running GenerationMixin.generate for preparing model specific inputs for
        each generation step
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            self.past_len += input_ids.shape[1]
        else:
            self.past_len = input_ids.shape[1]
        attention_mask = kwargs.get(
            "attention_mask",
            torch.ones(input_ids.shape[0],  self.past_len),
        )
        if not kwargs.get("use_cache", True):
            raise NotImplementedError("Llama with prefix_lm=True does not support use_cache=False.")
        else:
            prefix_mask = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prefix_mask": prefix_mask,
            "past_key_values": past_key_values,
            "images": kwargs.get("images", None),
        }

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called.
        This is required to match `past_key_values` with the correct beam_idx at every generation step.
        """

        # from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
        return tuple(
            tuple(np.take(past_state, beam_idx, 0) for past_state in layer_past)
            for layer_past in past_key_values
        )


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def evaluate(pred, output_file="eval_results.jsonl"):
    closed_scores = collections.defaultdict(list)
    open_scores = collections.defaultdict(list)
    evaluation_results = []

    for pred_item in pred:
        gt_value = pred_item['gt_text'].lower().strip()
        pred_value = pred_item['pred_text'].lower().strip()

        answer_type = 'CLOSED' if gt_value in ['yes', 'no'] else 'OPEN'
        eval_result = 0

        if answer_type == 'OPEN':
            open_scores['q_id'].append(pred_item['question_id'])
            if gt_value in pred_value:
                open_scores['hit'].append(1)
                eval_result = 1
            else:
                open_scores['hit'].append(0)
 
        elif answer_type == 'CLOSED':
            closed_scores['q_id'].append(pred_item['question_id'])
            if 'yes' in pred_value or 'no' in pred_value:
                if gt_value in pred_value:
                    closed_scores['hit'].append(1)
                    eval_result = 1
                else:
                    closed_scores['hit'].append(0)
            else:
                closed_scores['hit'].append(0)

        evaluation_result = {
            "question_id": pred_item['question_id'],
            "prompt": pred_item['prompt'],
            "gt_text": pred_item['gt_text'],
            "pred_text": pred_item['pred_text'],
            "eval_result": eval_result
        }
        evaluation_results.append(evaluation_result)

    open_score = sum(open_scores['hit']) / len(open_scores['hit']) if len(open_scores['hit']) != 0 else 0.0
    closed_score = sum(closed_scores['hit']) / len(closed_scores['hit']) if len(closed_scores['hit']) != 0 else 0.0

    num_open, num_close = len(open_scores['hit']), len(closed_scores['hit']) 

    
    eval_results_summary = {
        "output_file": output_file,
        "num_open": num_open,
        "num_close": num_close,
        "open_type_accuracy": {
            "percentage": f"{open_score * 100:.2f}%",
            "correct": sum(open_scores['hit']),
            "total": len(open_scores['hit'])
        },
        "close_type_accuracy": {
            "percentage": f"{closed_score * 100:.2f}%",
            "correct": sum(closed_scores['hit']),
            "total": len(closed_scores['hit'])
        }
    }

    with open(f"eval_summary_{output_file}", 'w') as f:
        f.write(json.dumps(eval_results_summary, indent=4))
    
    print(f"\n\n{output_file}")
    print(json.dumps(eval_results_summary, indent=4))
            
    # Write evaluation results to a file
    with open(output_file, "w") as f:
        for result in evaluation_results:
            f.write(json.dumps(result) + "\n")
           
#          
# Inference Code for Medical Dataset :flaviagiammarinov/vqa-rad
#

####
base_dir = Path("/home/sdp/ravi/openvino_notebooks/notebooks/257-llava-multimodal-chatbot-v2/LLaVA-Med")

delta_weights_arr = ["vqa_rad", "ckpt2"]
wt_precision_arr = ["fp32", "int8", "int4"]

# Mapping for delta weights
delta_weights_mapping = {
    "vqa_rad": "data_RAD-9epoch_delta",
    "ckpt2": "llava_med_in_text_60k_ckpt2_delta"
}

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
mm_use_im_start_end = True

config_mm_vision_tower = "openai/clip-vit-large-patch14"
image_processor = CLIPImageProcessor.from_pretrained(config_mm_vision_tower)

# Load the dataset (not in streaming mode)
dataset = load_dataset("flaviagiammarino/vqa-rad")

# Get the total number of samples in the test split
test_num_samples = len(dataset["test"])
print("Total number of samples in the test split:", test_num_samples)

num_examples_to_evaluate = test_num_samples
# num_examples_to_evaluate = 3

retry_phrases = [
    'having access', 'without access', 'have access', 'access to the image', 
    'access to the actual image', 'without the actual image', 'not mentioned in the figure caption',
    'not mentioned in the caption', 'not provided in the figure caption', 'not specified in the caption',
    'not specified in the image caption', 'without seeing the', 'cannot see the', 'unable to view', 
    'unable to provide specific details', 'sorry', 'cannot directly view', 'cannot view the image', 
    'cannot view the actual image','image itself is not available', 'image is not available'
]

core = ov.Core()
device = "AUTO"
    
for delta_weights in delta_weights_arr:
    pt_llava_med = base_dir / f"llava_med_model_{delta_weights}"
    tokenizer = AutoTokenizer.from_pretrained(pt_llava_med)
    
    for wt_precision in wt_precision_arr:
        ov_out_path = base_dir / f"ov_llava_med_{delta_weights}_{wt_precision}"
        # ov_out_path, device.value, vision_config.use_im_start_end, vision_config.im_patch_token, vision_config.im_start_token, vision_config.im_end_token)
        # ov_llava_med AUTO True 32001 32002 32003
        ov_model = OVLlavaMedForCausalLM(core, ov_out_path, device)

        correct_answers = 0
        max_retries = 3
        results = []

        for i, example in enumerate(dataset["test"]):
            
            if i > num_examples_to_evaluate:
                break

            image = example["image"]
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]

            input_prompt = example["question"]
            expected_answer = example["answer"]

            conv_mode = "multimodal"
            conv = conv_templates[conv_mode].copy()
            roles = ("user", "assistant")

            if mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + input_prompt
            else:
                inp = DEFAULT_IMAGE_TOKEN + "\n" + input_prompt
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)

            prompt = conv.get_prompt()
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str, "##"]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            if delta_weights == "vqa_rad":
                max_new_tokens = 16
            else:
                max_new_tokens = 64
                
            print(f"Max new tokens: {max_new_tokens}")
            print(f"Question: {input_prompt}")
            print(f"Expected answer: {expected_answer}")

            retries = 0
            while retries < max_retries:
                other_tokens_latency_list = []
                first_token_latency = 0
                print("Model Answer:")
                
                start_time = time.time()
                output_ids = ov_model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=max_new_tokens,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Decode the generated text
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Find the 2nd part after "Assistant:" and Remove "\n##" at the end, which is model answer.
                if "###Assistant:" in generated_text:
                    generated_answer = generated_text.split("###Assistant:")[2][:-3].strip()
                else:
                    generated_answer = generated_text.strip().split()[0].lower()

                # Check if the generated text contains any retry phrases
                if not any(key in generated_text.lower() for key in retry_phrases):
                    break
                retries += 1
                print(f"Retrying inference... ({retries}/{max_retries})")

            num_tokens_generated = len(other_tokens_latency_list) + 1  # +1 for the first token
            avg_other_token_latency = sum(other_tokens_latency_list) / len(other_tokens_latency_list) if other_tokens_latency_list else 0

            # Append result for evaluation in the specified format
            result = {
                "question_id": i,
                "prompt": input_prompt,
                "gt_text": expected_answer,
                "pred_text": generated_answer,
                "max_new_tokens": max_new_tokens,
                "num_tokens_generated": num_tokens_generated,
                "first_token_latency": first_token_latency,
                "avg_other_token_latency": avg_other_token_latency,
                "execution_time": execution_time,
                "model_id": f"ov_llava_med_{delta_weights}_{wt_precision}",
            }
            results.append(result)

        # Save the results in JSONL format for evaluation later if needed.
        with open(f"results_{delta_weights}_{wt_precision}.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        # Calculate averages
        average_first_token_latency = sum(r["first_token_latency"] for r in results) / len(results)
        average_avg_other_token_latency = sum(r["avg_other_token_latency"] for r in results) / len(results)
        total_execution_time = sum(r["execution_time"] for r in results) 
        average_execution_time = total_execution_time / len(results)
        avg_num_tokens_generated = sum(r["num_tokens_generated"] for r in results) / len(results)
        
        # Save the averages to a file
        averages = {
            "delta_weights": delta_weights,
            "wt_precision": wt_precision,
            "num_samples_processed": len(results),
            "avg_num_tokens_generated": avg_num_tokens_generated,
            "average_first_token_latency": average_first_token_latency,
            "average_other_token_latency": average_avg_other_token_latency,
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time,
        }

        print(f"\n\nperf_averages_{delta_weights}_{wt_precision}:")
        print(json.dumps(averages, indent=4))

        with open(f"perf_averages_{delta_weights}_{wt_precision}.json", "w") as f:
            json.dump(averages, f, indent=4)
            
        # Perform evaluation
        evaluate(results, output_file=f"eval_results_{delta_weights}_{wt_precision}.jsonl")