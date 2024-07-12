from transformers.generation import GenerationConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoConfig
import numpy as np
import torch
from typing import Optional, Tuple
import openvino as ov

from typing import Optional, Tuple, List
from llava.model.llava import LlavaLlamaModel
import nncf
from openvino.runtime import opset13
import numpy as np


def model_has_state(ov_model: ov.Model):
    # TODO: Provide a better way based on the variables availability, but OV Python API doesn't expose required methods
    return len(ov_model.get_sinks()) > 0


def model_has_input_output_name(ov_model: ov.Model, name: str):
    """
    Helper function for checking that model has specified input or output name

    Parameters:
      ov_model (ov.Model):   # TODO: Can we derive the dimensions from the model topology?
      name (str):
          name of input or output

    Returns:
      True if input or output with requested name exists else False
    """
    return name in sum([list(t.get_names()) for t in ov_model.inputs + ov_model.outputs], [])

def fuse_cache_reorder(
    ov_model: ov.Model, not_kv_inputs: List[str], key_value_input_names: List[str], gather_dim: int
):
    """
    Fuses reored_cache during generate cycle into ov.Model. Used with stateful models, because we can not modify model state directly.

    Adds a new beam_idx parameter and Gather op per each kv-cache input in a given model.
    Should be run before make_stateful. Implements optimumum's _reorder_cache
    inside the model in the beginning of each iteration.
    Gather works along given gather_dim dimension that may vary from model to model.
    KV-cache inputs are identified based on names in key_value_input_names.
    Append the new beam_idx parameter to not_kv_inputs.

    Parameters:
      ov_model (`ov.Model`):
          openvino model for processing
      not_kv_inputs (`List[str]`):
          list of input nodes in model that not related to past key values
      key_value_input_names (`List[str]`):
          list of names for key value input layers
      gather_dim (int):
          dimension for gathering cache during reorder pass
    """

    if model_has_input_output_name(ov_model, "beam_idx"):
        raise ValueError("Model already has fused cache")
    input_batch = ov_model.input("inputs_embeds").get_partial_shape()[0]
    beam_idx = opset13.parameter(name="beam_idx", dtype=ov.Type.i32, shape=ov.PartialShape([input_batch]))
    beam_idx.output(0).get_tensor().add_names({"beam_idx"})  # why list is not accepted?
    ov_model.add_parameters([beam_idx])
    not_kv_inputs.append(ov_model.inputs[-1])
    # Go over all cache parameters and fuse _reorder_cache with indices provided by the new parameter beam_idx
    for input_name in key_value_input_names:
        parameter_output_port = ov_model.input(input_name)
        consumers = parameter_output_port.get_target_inputs()
        gather = opset13.gather(parameter_output_port, beam_idx, opset13.constant(gather_dim))
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))
    ov_model.validate_nodes_and_infer_types()


def build_state_initializer(ov_model: ov.Model, batch_dim: int):
    """
    Build initialization ShapeOf Expression for all ReadValue ops

    Parameters:
      ov_model (ov.Model):
          openvino model
      batch_dim (int):
          index of dimension corresponding to batch size
    """
    input_ids = ov_model.input("inputs_embeds")
    batch = opset13.gather(opset13.shape_of(input_ids, output_type="i64"), opset13.constant([0]), opset13.constant(0))
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            dims = [dim.min_length for dim in list(op.get_output_partial_shape(0))]
            dims[batch_dim] = batch
            dims = [opset13.constant(np.array([dim], dtype=np.int64)) if isinstance(dim, int) else dim for dim in dims]
            shape = opset13.concat(dims, axis=0)
            broadcast = opset13.broadcast(opset13.constant(0.0, dtype=op.get_output_element_type(0)), shape)
            op.set_arguments([broadcast])
    ov_model.validate_nodes_and_infer_types()


def make_stateful(
    ov_model: ov.Model,
    not_kv_inputs: List[str],
    key_value_input_names: List[str],
    key_value_output_names: List[str],
    batch_dim: int,
    num_attention_heads: int,
    num_beams_and_batch: int = None,
):
    """
    Hides kv-cache inputs and outputs inside the model as variables.

    Parameters:
        ov_model (ov.Model):
            openvino model
        not_kv_inputs (`List[str]`):
            list of input nodes in model that not related to past key values
        key_value_input_names (`List[str]`):
            list of names for key value input layers
        key_value_output_names (`List[str]`):
            list of names for key value input layers
        batch_dim (int):
            index of batch dimension in key value layers
        num_attention_heads (int):
            number of attention heads for batch dimension initialization
        num_beams_an_batch (int):
            precalculated number of beams and batch for shapes initialization
    """
    from openvino._offline_transformations import apply_make_stateful_transformation

    input_output_map = {}
    # TODO: Can we derive the dimensions from the model topology?

    if num_beams_and_batch is not None:
        # Set batch size for input_ids and attention mask to avoid dynamic dimension got propagated from the end of the model back to ReadValue
        for input in not_kv_inputs:
            shape = input.get_partial_shape()
            if shape.rank.get_length() <= 2:  # == 1 for beam_index
                shape[0] = num_beams_and_batch
                input.get_node().set_partial_shape(shape)
            else:
                log.warn(f"Rank of {input.get_any_name()} input of the model is not 2, batch size is not set")

    for kv_name_pair in zip(key_value_input_names, key_value_output_names):
        input_output_map[kv_name_pair[0]] = kv_name_pair[1]
        if num_beams_and_batch is not None:
            input = ov_model.input(kv_name_pair[0])
            shape = input.get_partial_shape()
            shape[batch_dim] = num_beams_and_batch * num_attention_heads
            input.get_node().set_partial_shape(shape)

    if num_beams_and_batch is not None:
        # Re-validation model if shapes are altered above
        ov_model.validate_nodes_and_infer_types()

    apply_make_stateful_transformation(ov_model, input_output_map)
    if num_beams_and_batch is None:
        build_state_initializer(ov_model, batch_dim)


def patch_stateful(ov_model):
    key_value_input_names = [
        key.get_any_name() for key in ov_model.inputs[1:]
    ]
    key_value_output_names = [
        key.get_any_name() for key in ov_model.outputs[1:]
    ]
    not_kv_inputs = [
        input for input in ov_model.inputs if not any(name in key_value_input_names for name in input.get_names())
    ]
    if not key_value_input_names or not key_value_output_names:
        return
    batch_dim =  0
    num_attention_heads = 1

    fuse_cache_reorder(ov_model, not_kv_inputs, key_value_input_names, batch_dim)
    make_stateful(
        ov_model, not_kv_inputs, key_value_input_names, key_value_output_names, batch_dim, num_attention_heads, None
    )

llava_wc_parameters = dict(mode=nncf.CompressWeightsMode.INT4_ASYM, group_size=128, ratio=0.8)


class ModelWithPastWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.config.model_type = "llama"
        self.model.to_bettertransformer()
        self.llama = super(LlavaLlamaModel, model.model).forward

    def forward(self, inputs_embeds, past_key_values:Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]=None):
        outputs  = self.llama(inputs_embeds=inputs_embeds, past_key_values=past_key_values, use_cache=True)
        hidden_states = outputs[0]
        logits = self.model.lm_head(hidden_states)
        return logits, outputs.past_key_values

    
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
        self.request.start_async(inputs, share_inputs=True)
        self.request.wait()

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
                    # import pdb; pdb.set_trace()
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

from transformers import StoppingCriteria


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
    

from pathlib import Path
from transformers import CLIPImageProcessor
from transformers import AutoTokenizer
config_mm_vision_tower = "openai/clip-vit-large-patch14"
image_processor = CLIPImageProcessor.from_pretrained(config_mm_vision_tower)

pt_llava_med = Path("/home/sdp/ravi/openvino_notebooks/notebooks/257-llava-multimodal-chatbot/LLaVA-Med/llava_med_model")
tokenizer = AutoTokenizer.from_pretrained(pt_llava_med)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
mm_use_im_start_end = True


core = ov.Core()
device = "AUTO"
ov_out_path = Path("/home/sdp/ravi/openvino_notebooks/notebooks/257-llava-multimodal-chatbot/LLaVA-Med/ov_llava_med")
# ov_out_path, device.value, vision_config.use_im_start_end, vision_config.im_patch_token, vision_config.im_start_token, vision_config.im_end_token)
# ov_llava_med AUTO True 32001 32002 32003
ov_model = OVLlavaMedForCausalLM(core, ov_out_path, device)

import requests
from PIL import Image
from io import BytesIO


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

print("\nGeneral Converstation Sample\n")

image_file = "https://llava-vl.github.io/static/images/view.jpg"

image = load_image(image_file)
image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]

text_message = "What are the things I should be cautious about when I visit here?"
print(f"Question: {text_message}")
#image

from transformers import TextStreamer
from llava.conversation import conv_templates, SeparatorStyle


# Prepare
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
conv_mode = "multimodal"

conv = conv_templates[conv_mode].copy()
roles = ("user", "assistant")

if mm_use_im_start_end:
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text_message
else:
    inp = DEFAULT_IMAGE_TOKEN + "\n" + text_message
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print("Answer:")

output_ids = ov_model.generate(
    input_ids,
    images=image_tensor,
    do_sample=True,
    temperature=0.2,
    max_new_tokens=128,
    streamer=streamer,
    use_cache=True,
    stopping_criteria=[stopping_criteria],
)


print("\nMedical Converstation Sample\n")
# Medical Conversation
from datasets import load_dataset

dataset = load_dataset("flaviagiammarino/vqa-rad", streaming=True)

example = next(iter(dataset["train"]))
image = example["image"]
image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"]

text_message = example["question"]
print(f"Question: {text_message}")
print(f"Expected answer: {example['answer']}")
#image

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
conv_mode = "multimodal"

conv = conv_templates[conv_mode].copy()
roles = ("user", "assistant")

if mm_use_im_start_end:
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text_message
else:
    inp = DEFAULT_IMAGE_TOKEN + "\n" + text_message
conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)

prompt = conv.get_prompt()
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
keywords = [stop_str, "##"]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
print("Model Answer:")

output_ids = ov_model.generate(
    input_ids,
    images=image_tensor,
    do_sample=True,
    temperature=0.2,
    max_new_tokens=128,
    streamer=streamer,
    use_cache=True,
    stopping_criteria=[stopping_criteria],
)

output_ids = ov_model.generate(
    input_ids,
    images=image_tensor,
    do_sample=True,
    temperature=0.2,
    max_new_tokens=128,
    streamer=streamer,
    use_cache=True,
    stopping_criteria=[stopping_criteria],
)


