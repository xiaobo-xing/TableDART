import os
from huggingface_hub import login

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nltk_data_path = os.path.join(project_root, "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_path
token = os.getenv("HF_TOKEN")
login(token)
import torch
import torch.nn as nn
import re
import json
import requests
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    GenerationConfig,
)
import shortuuid
from project_config.config import cfg


def _parse_json_and_extract_answer(json_string_batch):
    """Parse JSON responses and extract answer field from expert outputs."""
    if isinstance(json_string_batch, str):
        json_string_batch = [json_string_batch]
    extracted_answers = []
    for json_string in json_string_batch:
        if not json_string:
            extracted_answers.append("")
            continue
        try:
            match = re.search(
                '{\\s*\\"answer\\"\\s*:\\s*.*?}\\s*', json_string, re.DOTALL
            )
            if match:
                json_payload_str = match.group(0)
            else:
                json_payload_str = json_string
            data = json.loads(json_payload_str)
            answer = str(answer).strip().strip('"').strip(".")
            if answer is None:
                extracted_answers.append(json_string)
            elif isinstance(answer, list):
                extracted_answers.append(str(answer[0]) if answer else "")
            else:
                extracted_answers.append(str(answer))
        except json.JSONDecodeError:
            extracted_answers.append(json_string)
        except Exception:
            extracted_answers.append(json_string)
    return extracted_answers if len(extracted_answers) > 1 else extracted_answers[0]


class BaseExpert(nn.Module):
    """
    Base class for all expert models.
    Each expert produces intermediate states for gating network feature extraction.
    """
    def __init__(self, model_id_or_path, cost=1.0):
        super().__init__()
        self.model_id_or_path = model_id_or_path
        self.cost = cost  # Cost metric for routing (latency or efficiency)
        self.device = torch.device(cfg["TRAINING"]["DEVICE"])
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.gate_feature_dim = 0  # Dimension of features for gating network

    def _load_model_resources(self):
        raise NotImplementedError

    def extract_gate_features_and_intermediate_state(self, *args, **kwargs):
        """Extract features for gating network and intermediate model state."""
        raise NotImplementedError

    def continue_generation_from_state(self, *args, **kwargs):
        """Resume generation from saved intermediate state."""
        raise NotImplementedError

    def generate_full(self, *args, **kwargs):
        """Full generation pipeline from input to output."""
        raise NotImplementedError

    def forward_for_loss(self, *args, **kwargs):
        """Forward pass for computing training loss."""
        raise NotImplementedError


class TableGPT2Expert(BaseExpert):
    """
    Text-based expert using TableGPT2 model.
    Processes flattened table representations in text format.
    """
    def __init__(self, model_path=cfg["MODEL"]["TEXT_EXPERT_ID"], cost=1.2):
        super().__init__(model_path, cost)
        self._load_model_resources()
        self.gate_feature_dim = self.model.config.hidden_size

    def _load_model_resources(self):
        """Load pretrained TableGPT2 model and tokenizer."""
        print(f"INFO: Initializing TableGPT2Expert from {self.model_id_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id_or_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id_or_path,
            torch_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        config_max_pos_embed = getattr(
            self.model.config, "max_position_embeddings", 2048
        )
        original_model_config_max_length = getattr(self.model.config, "max_length", 20)
        if (
            not isinstance(original_model_config_max_length, int)
            or original_model_config_max_length < config_max_pos_embed
        ):
            print(
                f"WARN: TableGPT2 model.config.max_length ({original_model_config_max_length}) is too small or problematic."
            )
            self.model.config.max_length = config_max_pos_embed
            print(
                f"INFO: TableGPT2 model.config.max_length has been overridden to: {self.model.config.max_length}"
            )
        else:
            print(
                f"INFO: TableGPT2 model.config.max_length ({original_model_config_max_length}) seems adequate."
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        print(
            f"INFO: TableGPT2 loaded (device map: {getattr(self.model, 'hf_device_map', 'auto')})"
        )

    def _create_generation_config(self, override):
        cfg_merge = {**cfg["GENERATION"], **override}
        gen_conf = {
            "max_new_tokens": cfg_merge.get("MAX_NEW_TOKENS", 20),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "repetition_penalty": 1.2,
            "length_penalty": 0.8,
        }
        if cfg_merge.get("NUM_BEAMS", 1) > 1:
            gen_conf.update(
                {
                    "num_beams": cfg_merge["NUM_BEAMS"],
                    "do_sample": False,
                    "early_stopping": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 1,
                }
            )
        print(f"[INFO TableGPT2] _create_generation_config (final): {gen_conf}")
        return gen_conf

    def extract_gate_features_and_intermediate_state(self, prompts_batch):
        tokens = self.tokenizer(
            prompts_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=(self.tokenizer.model_max_length or 4096)
            - cfg["GENERATION"].get("MAX_NEW_TOKENS", 150),
        ).to(self.model.device)
        embeds = self.model.get_input_embeddings()(tokens.input_ids)
        mask = tokens.attention_mask.unsqueeze(-1)
        pooled = (embeds * mask).sum(1) / mask.sum(1).clamp(min=1e-09)
        state = {"inputs_embeds": embeds, "attention_mask": tokens.attention_mask}
        return (pooled, state)

    def continue_generation_from_state(
        self, intermediate_state, target_len=None, for_loss=False, **override
    ):
        if for_loss and target_len is not None:
            out = self.model(
                inputs_embeds=intermediate_state["inputs_embeds"],
                attention_mask=intermediate_state["attention_mask"],
            )
            logits = out.logits
            L = logits.shape[1]
            if L < target_len:
                pad = torch.zeros(
                    logits.shape[0], target_len - L, logits.shape[2], device=self.device
                )
                return torch.cat([logits, pad], dim=1)
            return logits[:, :target_len, :]
        gen_kwargs = self._create_generation_config(override)
        prompt_len_for_debug_only = intermediate_state["inputs_embeds"].shape[1]
        effective_model_max_len = getattr(self.model.config, "max_length", 2048)
        requested_max_new = gen_kwargs.get(
            "max_new_tokens", cfg["GENERATION"]["MAX_NEW_TOKENS"]
        )
        gen_kwargs["max_length"] = prompt_len_for_debug_only + requested_max_new
        if gen_kwargs["max_length"] > effective_model_max_len:
            gen_kwargs["max_length"] = effective_model_max_len
        with torch.no_grad():
            ids = self.model.generate(
                inputs_embeds=intermediate_state["inputs_embeds"],
                attention_mask=intermediate_state["attention_mask"],
                **gen_kwargs,
            )
        results = []
        for seq_idx, generated_sequence_tensor in enumerate(ids):
            raw_decoded_text = ""
            if generated_sequence_tensor.numel() > 0:
                if generated_sequence_tensor.shape[0] > prompt_len_for_debug_only:
                    new_tokens = generated_sequence_tensor[prompt_len_for_debug_only:]
                    raw_decoded_text = self.tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    ).strip()
                else:
                    raw_decoded_text = self.tokenizer.decode(
                        generated_sequence_tensor, skip_special_tokens=True
                    ).strip()
            else:
                pass
            if raw_decoded_text:
                results.append(raw_decoded_text)
            else:
                results.append("")
        return results[0] if ids.shape[0] == 1 else results

    def _extract_first_sentence(self, text):
        try:
            import nltk
            from nltk.tokenize import sent_tokenize
            import os

            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            nltk_data_path = os.path.join(project_root, "nltk_data")
            os.makedirs(nltk_data_path, exist_ok=True)
            if nltk_data_path not in nltk.data.path:
                nltk.data.path.insert(0, nltk_data_path)
            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                print("INFO: Downloading NLTK punkt tokenizer")
                nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
            text = text.strip()
            if not text:
                return text
            sentences = sent_tokenize(text)
            if sentences and sentences[0].strip():
                return sentences[0].strip()
            else:
                return text
        except ImportError:
            try:
                import spacy

                try:
                    nlp = spacy.load("en_core_web_sm")
                except OSError:
                    nlp = spacy.blank("en")
                    nlp.add_pipe("sentencizer")
                text = text.strip()
                if not text:
                    return text
                doc = nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents]
                if sentences and sentences[0]:
                    return sentences[0]
                else:
                    return text
            except ImportError:
                text = text.strip()
                if not text:
                    return text
                import re

                abbreviations = [
                    "Mr",
                    "Mrs",
                    "Ms",
                    "Dr",
                    "Prof",
                    "Jr",
                    "Sr",
                    "vs",
                    "etc",
                    "Inc",
                    "Ltd",
                    "Co",
                ]
                temp_text = text
                for abbr in abbreviations:
                    temp_text = re.sub(
                        f"\\b{abbr}\\.",
                        f"{abbr}__DOT__",
                        temp_text,
                        flags=re.IGNORECASE,
                    )
                sentences = re.split("[.!?]+\\s+", temp_text)
                if sentences and sentences[0].strip():
                    first_sentence = sentences[0].replace("__DOT__", ".")
                    if first_sentence and (not first_sentence[-1] in ".!?"):
                        first_sentence += "."
                    return first_sentence.strip()
                return text


class TableLLavaExpert(BaseExpert):
    def __init__(self, model_path=cfg["MODEL"]["VLM_EXPERT_ID"], cost=2.5):
        super().__init__(model_path, cost)
        self.gate_feature_dim = None
        self._load_model_resources()
        if self.gate_feature_dim is None:
            print(
                f"CRITICAL WARN: LLaVA gate_feature_dim could not be auto-detected. Using default (4096)."
            )
            self.gate_feature_dim = 4096

    def _load_model_resources(self):
        print(f"INFO: Initializing TableLLavaExpert from {self.model_id_or_path}")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id_or_path)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_id_or_path,
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            print(
                f"INFO: TableLLaVA model ({self.model_id_or_path}) and processor loaded. Device map: {(self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else self.model.device)}"
            )
            if hasattr(self.model.config, "text_config") and hasattr(
                self.model.config.text_config, "hidden_size"
            ):
                self.gate_feature_dim = self.model.config.text_config.hidden_size
                print(
                    f"INFO: LLaVA gate_feature_dim set from model.config.text_config.hidden_size: {self.gate_feature_dim}"
                )
            elif hasattr(self.model, "language_model") and hasattr(
                self.model.language_model.config, "hidden_size"
            ):
                self.gate_feature_dim = self.model.language_model.config.hidden_size
                print(
                    f"INFO: LLaVA gate_feature_dim set from model.language_model.config.hidden_size: {self.gate_feature_dim}"
                )
            elif hasattr(self.model.config, "hidden_size"):
                self.gate_feature_dim = self.model.config.hidden_size
                print(
                    f"INFO: LLaVA gate_feature_dim set from model.config.hidden_size (less common): {self.gate_feature_dim}"
                )
            else:
                print(
                    f"WARN: Could not automatically determine LLaVA LLM hidden_size from config."
                )
                self.gate_feature_dim = 4096
                print(
                    f"INFO: Using assumed LLaVA LLM hidden size for gate_feature_dim: {self.gate_feature_dim}. VERIFY THIS."
                )
            if self.gate_feature_dim is None:
                raise AttributeError(
                    "Failed to determine LLaVA's language model hidden size for gate_feature_dim."
                )
        except Exception as e:
            print(f"ERROR: Failed to load TableLLaVA model or processor: {e}")
            raise e

    def _create_generation_config_llava(self, generation_config_override):
        merged_cfg = {**cfg["GENERATION"], **generation_config_override}
        final_params = {
            "max_new_tokens": merged_cfg.get("MAX_NEW_TOKENS", 256),
            "eos_token_id": self.processor.tokenizer.eos_token_id
            if hasattr(self.processor, "tokenizer")
            and self.processor.tokenizer.eos_token_id is not None
            else self.model.config.eos_token_id,
            "pad_token_id": self.processor.tokenizer.pad_token_id
            if hasattr(self.processor, "tokenizer")
            and self.processor.tokenizer.pad_token_id is not None
            else self.model.config.pad_token_id,
            "do_sample": merged_cfg.get("DO_SAMPLE", False),
        }
        if final_params["do_sample"]:
            final_params["temperature"] = merged_cfg.get("TEMPERATURE", 0.2)
        return {k: v for k, v in final_params.items() if v is not None}

    def _prepare_llava_inputs(self, prompts_text_parts_batch, image_paths_batch):
        batch_images_pil = []
        valid_prompts = []
        valid_indices = []
        img_size = getattr(self.model.config.vision_config, "image_size", 224)
        for i, (p_text, img_path) in enumerate(
            zip(prompts_text_parts_batch, image_paths_batch)
        ):
            current_image_pil = None
            if img_path and os.path.exists(img_path):
                try:
                    current_image_pil = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(
                        f"WARN: LLaVA couldn't load image {img_path}: {e}. Using placeholder."
                    )
            if current_image_pil is None:
                current_image_pil = Image.new(
                    "RGB", (img_size, img_size), (128, 128, 128)
                )
            batch_images_pil.append(current_image_pil)
            valid_prompts.append(p_text)
            valid_indices.append(i)
        if not batch_images_pil:
            bs = len(prompts_text_parts_batch)
            dummy_text_inputs = self.processor(
                text=["<image>\nNo valid image."] * bs,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            dummy_pixel_values = torch.zeros(
                (bs, 3, img_size, img_size),
                device=self.model.device,
                dtype=self.model.dtype,
            )
            return (
                {
                    "input_ids": dummy_text_inputs.input_ids,
                    "attention_mask": dummy_text_inputs.attention_mask,
                    "pixel_values": dummy_pixel_values,
                },
                [],
                [],
            )
        full_chat_prompts = []
        for p_text in valid_prompts:
            conversation = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": p_text}],
                }
            ]
            full_chat_prompts.append(
                self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True
                )
            )
        inputs = self.processor(
            text=full_chat_prompts,
            images=batch_images_pil,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length or 2048,
        ).to(self.model.device)
        return (inputs, valid_indices, batch_images_pil)

    def extract_gate_features_and_intermediate_state(
        self, prompts_text_parts_batch, image_paths_batch
    ):
        inputs, valid_indices, _ = self._prepare_llava_inputs(
            prompts_text_parts_batch, image_paths_batch
        )
        batch_size_total = len(prompts_text_parts_batch)
        pixel_values_for_tower = inputs.pixel_values.to(
            dtype=self.model.vision_tower.dtype
            if hasattr(self.model.vision_tower, "dtype")
            else self.model.dtype
        )
        if not valid_indices and batch_size_total > 0:
            dummy_pooled_features = torch.zeros(
                batch_size_total,
                self.gate_feature_dim,
                device=self.device,
                dtype=self.model.dtype,
            )
            dummy_input_ids = self.processor(
                text=["<image>\nDummy"] * batch_size_total,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(self.device)
            dummy_state = {
                "input_ids": dummy_input_ids,
                "attention_mask": torch.ones_like(dummy_input_ids),
                "pixel_values": inputs.pixel_values,
                "projected_image_features_for_llm": torch.zeros(
                    batch_size_total,
                    1,
                    self.gate_feature_dim,
                    device=self.device,
                    dtype=self.model.dtype,
                ),
            }
            return (dummy_pooled_features, dummy_state)
        with torch.no_grad():
            image_features_output = self.model.vision_tower(
                pixel_values_for_tower, output_hidden_states=False
            )
            image_last_hidden_state = (
                image_features_output.last_hidden_state
                if hasattr(image_features_output, "last_hidden_state")
                else image_features_output[0]
            )
            projected_image_features = self.model.multi_modal_projector(
                image_last_hidden_state
            )
            pooled_visual_features_valid = projected_image_features.mean(dim=1)
            if pooled_visual_features_valid.shape[-1] != self.gate_feature_dim:
                print(
                    f"CRITICAL WARNING: LLaVA pooled projected features dim {pooled_visual_features_valid.shape[-1]} != self.gate_feature_dim {self.gate_feature_dim}."
                )
            pooled_visual_features = torch.zeros(
                batch_size_total,
                self.gate_feature_dim,
                device=self.device,
                dtype=self.model.dtype,
            )
            if valid_indices:
                pooled_visual_features[
                    torch.tensor(valid_indices, device=self.device)
                ] = pooled_visual_features_valid
        intermediate_state = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "pixel_values": inputs.pixel_values,
            "projected_image_features_for_llm": projected_image_features,
        }
        return (pooled_visual_features, intermediate_state)

    def continue_generation_from_state(
        self,
        intermediate_state,
        target_len=None,
        for_loss=False,
        **generation_kwargs_override,
    ):
        if for_loss and target_len is not None:
            model_inputs_for_forward = {
                "input_ids": intermediate_state["input_ids"],
                "attention_mask": intermediate_state["attention_mask"],
                "pixel_values": intermediate_state["pixel_values"].to(self.model.dtype),
            }
            outputs = self.model(**model_inputs_for_forward)
            logits = outputs.logits
            current_seq_len = logits.shape[1]
            if current_seq_len > target_len:
                return logits[:, :target_len, :]
            elif current_seq_len < target_len:
                padding = torch.zeros(
                    logits.shape[0],
                    target_len - current_seq_len,
                    logits.shape[2],
                    device=self.device,
                    dtype=logits.dtype,
                )
                return torch.cat([logits, padding], dim=1)
            return logits
        else:
            gen_params = self._create_generation_config_llava(
                generation_kwargs_override
            )
            model_inputs_for_generate = {
                "input_ids": intermediate_state["input_ids"],
                "attention_mask": intermediate_state["attention_mask"],
                "pixel_values": intermediate_state["pixel_values"].to(self.model.dtype),
            }
            with torch.no_grad():
                gen_ids = self.model.generate(**model_inputs_for_generate, **gen_params)
            final_batch_answers = []
            prompt_lens = [len(ids) for ids in intermediate_state["input_ids"]]
            for i in range(gen_ids.shape[0]):
                current_prompt_len = prompt_lens[i]
                raw_decoded_text = ""
                if gen_ids[i].numel() > 0:
                    if gen_ids[i].shape[0] > current_prompt_len:
                        output_ids_single = gen_ids[i, current_prompt_len:]
                        raw_decoded_text = self.processor.decode(
                            output_ids_single, skip_special_tokens=True
                        ).strip()
                    else:
                        raw_decoded_text = self.processor.decode(
                            gen_ids[i], skip_special_tokens=True
                        ).strip()
                else:
                    pass
                if raw_decoded_text:
                    final_batch_answers.append(raw_decoded_text.strip())
                else:
                    final_batch_answers.append("")
            return (
                final_batch_answers[0] if gen_ids.shape[0] == 1 else final_batch_answers
            )

    def generate_full(
        self, prompt_text_part_single, image_path_single, **generation_kwargs_override
    ):
        gen_params = self._create_generation_config_llava(generation_kwargs_override)
        img_size = getattr(self.model.config.vision_config, "image_size", 224)
        current_image_pil = None
        if not image_path_single or not os.path.exists(image_path_single):
            print(
                f"WARN: Image not found for TableLLaVA full generation: {image_path_single}. Using placeholder."
            )
            current_image_pil = Image.new("RGB", (img_size, img_size), (128, 128, 128))
        else:
            try:
                current_image_pil = Image.open(image_path_single).convert("RGB")
            except Exception as e:
                return f"[Error: Image loading failed: {e}]"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text_part_single},
                ],
            }
        ]
        llava_full_chat_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        raw_decoded_text = "[Error: TableLLaVA Full Generation failed pre-decode]"
        try:
            inputs = self.processor(
                text=llava_full_chat_prompt,
                images=current_image_pil,
                return_tensors="pt",
            ).to(self.model.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=self.model.vision_tower.dtype
                if hasattr(self.model.vision_tower, "dtype")
                else self.model.dtype
            )
            with torch.no_grad():
                gen_ids = self.model.generate(**inputs, **gen_params)
            prompt_len = inputs.input_ids.shape[1]
            output_ids = gen_ids[0, prompt_len:]
            raw_decoded_text = self.processor.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            return raw_decoded_text.strip()
        except Exception as e_gen:
            print(f"ERROR: TableLLaVA (full) generation failed: {e_gen}")
            import traceback

            traceback.print_exc()
            return f"[Error: TableLLaVA Full Generation failed - {e_gen}]"

    def forward_for_loss(self, **kwargs):
        return None


class Ovis2Expert(BaseExpert):
    def __init__(self, model_path=cfg["MODEL"]["VLM_EXPERT_ID"], cost=3.0):
        super().__init__(model_path, cost)
        self.text_tokenizer = None
        self.visual_tokenizer = None
        self.image_token_id = None
        self.gate_feature_dim = 6144
        self._load_model_resources()
        print(
            f"INFO: Ovis2Expert initialized with gate feature dimension: {self.gate_feature_dim}"
        )

    def _load_model_resources(self):
        print(f"INFO: Initializing Ovis2Expert from {self.model_id_or_path}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                torch_dtype=torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                multimodal_max_length=32768,
                trust_remote_code=True,
            ).to(self.device)
            self.text_tokenizer = self.model.get_text_tokenizer()
            self.visual_tokenizer = self.model.get_visual_tokenizer()
            if self.text_tokenizer is None:
                raise ValueError("Failed to get text_tokenizer from Ovis2 model")
            if self.visual_tokenizer is not None and hasattr(
                self.visual_tokenizer, "to"
            ):
                self.visual_tokenizer = self.visual_tokenizer.to(self.model.device)
            else:
                print(f"WARN: visual_tokenizer is None or doesn't have 'to' method")
            if hasattr(self.model, "vision_tower") and hasattr(
                self.model.vision_tower, "to"
            ):
                self.model.vision_tower = self.model.vision_tower.to(self.model.device)
            self.image_token_id = self.text_tokenizer.convert_tokens_to_ids("<image>")
            if self.image_token_id is None:
                self.image_token_id = 151646
                print(
                    f"WARN: Could not look up '<image>' token ID. Using hardcoded fallback value: {self.image_token_id}. This is expected for some versions of Ovis/Qwen."
                )
            self.text_tokenizer.padding_side = "left"
            if self.text_tokenizer.pad_token is None:
                if self.text_tokenizer.eos_token is not None:
                    self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
                else:
                    self.text_tokenizer.pad_token = "<|endoftext|>"
            if self.text_tokenizer.eos_token is None:
                self.text_tokenizer.eos_token = "<|endoftext|>"
            if self.text_tokenizer.pad_token_id is None:
                if self.text_tokenizer.eos_token_id is not None:
                    self.text_tokenizer.pad_token_id = self.text_tokenizer.eos_token_id
                else:
                    self.text_tokenizer.pad_token_id = 151643
            if self.text_tokenizer.eos_token_id is None:
                self.text_tokenizer.eos_token_id = 151645
            if hasattr(self.model, "config"):
                self.model.config.pad_token_id = self.text_tokenizer.pad_token_id
                self.model.config.eos_token_id = self.text_tokenizer.eos_token_id
            if hasattr(self.model, "generation_config"):
                if self.model.generation_config.pad_token_id is None:
                    self.model.generation_config.pad_token_id = self.text_tokenizer.pad_token_id
                if self.model.generation_config.eos_token_id is None:
                    self.model.generation_config.eos_token_id = self.text_tokenizer.eos_token_id
            print(
                f"INFO: Ovis2 model ({self.model_id_or_path}) loaded to device: {self.model.device}"
            )
            print(
                f"INFO: Ovis2 tokenizer pad_token_id={self.text_tokenizer.pad_token_id}, eos_token_id={self.text_tokenizer.eos_token_id}"
            )
        except Exception as e:
            print(f"ERROR: Failed to load Ovis2 model or tokenizers: {e}")
            raise e

    def _create_generation_config(self, generation_config_override):
        merged_cfg = {**cfg["GENERATION"], **generation_config_override}
        eos_token_id = self.model.generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        return {
            "max_new_tokens": merged_cfg.get("MAX_NEW_TOKENS", 1024),
            "do_sample": merged_cfg.get("DO_SAMPLE", False),
            "top_p": None,
            "top_k": None,
            "temperature": None,
            "repetition_penalty": None,
            "use_cache": True,
            "eos_token_id": eos_token_id,
            "pad_token_id": self.text_tokenizer.pad_token_id,
        }

    def extract_gate_features_and_intermediate_state(
        self, prompts_text_parts_batch, image_paths_batch
    ):
        batch_size = len(prompts_text_parts_batch)
        pooled_features_list = []
        intermediate_states = {
            "input_ids": [],
            "inputs_embeds": [],
            "attention_mask": [],
            "new_attention_mask": [],
            "pixel_values": [],
        }
        was_training = self.model.training
        self.model.eval()
        img_size = getattr(self.model.config, "image_size", 448)
        with torch.no_grad():
            for i in range(batch_size):
                prompt_text = prompts_text_parts_batch[i]
                image_path = image_paths_batch[i]
                query = f"<image>\n{prompt_text}"
                try:
                    image = Image.open(image_path).convert("RGB")
                except Exception:
                    image = Image.new("RGB", (img_size, img_size), (128, 128, 128))
                images = [image]
                _, input_ids, pixel_values = self.model.preprocess_inputs(
                    query, images, max_partition=9
                )
                input_ids = input_ids.to(self.model.device).unsqueeze(0)
                attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
                if pixel_values is not None:
                    if self.visual_tokenizer is not None and hasattr(
                        self.visual_tokenizer, "device"
                    ):
                        pixel_values = pixel_values.to(
                            dtype=self.visual_tokenizer.dtype,
                            device=self.visual_tokenizer.device,
                        )
                    else:
                        pixel_values = pixel_values.to(
                            device=self.model.device, dtype=self.model.dtype
                        )
                else:
                    pixel_values = torch.zeros(
                        9,
                        3,
                        img_size,
                        img_size,
                        device=self.model.device,
                        dtype=self.model.dtype,
                    )
                if pixel_values is None:
                    print(
                        f"WARN: pixel_values is None for sample {i}, creating zero tensor"
                    )
                    pixel_values = torch.zeros(
                        9,
                        3,
                        img_size,
                        img_size,
                        device=self.model.device,
                        dtype=self.model.dtype,
                    )
                pooled_feature = None
                if self.visual_tokenizer is not None:
                    try:
                        feat = self.visual_tokenizer.encode(pixel_values)
                        pooled_feature = feat.mean(dim=[0, 1]).unsqueeze(0)
                    except Exception as e:
                        print(
                            f"WARN: visual_tokenizer.encode failed for sample {i}: {e}; using zero features."
                        )
                        pooled_feature = torch.zeros(
                            1,
                            self.gate_feature_dim,
                            device=self.model.device,
                            dtype=self.model.dtype,
                        )
                else:
                    pooled_feature = torch.zeros(
                        1,
                        self.gate_feature_dim,
                        device=self.model.device,
                        dtype=self.model.dtype,
                    )
                pooled_features_list.append(pooled_feature)
                pixel_values_list = [pixel_values]
                _, inputs_embeds, _, new_attention_mask = self.model.merge_multimodal(
                    text_input_ids=input_ids,
                    pixel_values=pixel_values_list,
                    text_attention_masks=attention_mask,
                    text_labels=None,
                )
                intermediate_states["input_ids"].append(input_ids)
                intermediate_states["inputs_embeds"].append(inputs_embeds)
                intermediate_states["attention_mask"].append(attention_mask)
                intermediate_states["new_attention_mask"].append(new_attention_mask)
                intermediate_states["pixel_values"].append(pixel_values)
        if was_training:
            self.model.train()
        batch_pooled_features = torch.cat(pooled_features_list, dim=0)
        batch_intermediate_state = {
            "input_ids": intermediate_states["input_ids"],
            "inputs_embeds": intermediate_states["inputs_embeds"],
            "attention_mask": intermediate_states["attention_mask"],
            "new_attention_mask": intermediate_states["new_attention_mask"],
            "pixel_values": intermediate_states["pixel_values"],
        }
        return (batch_pooled_features, batch_intermediate_state)

    def continue_generation_from_state(
        self, intermediate_state, **generation_kwargs_override
    ):
        gen_params = self._create_generation_config(generation_kwargs_override)
        if isinstance(intermediate_state["pixel_values"], list):
            pixel_values = intermediate_state["pixel_values"][0]
            input_ids = intermediate_state["input_ids"][0]
            attention_mask = intermediate_state["attention_mask"][0]
        else:
            pixel_values = intermediate_state["pixel_values"]
            input_ids = intermediate_state["input_ids"]
            attention_mask = intermediate_state["attention_mask"]
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            try:
                pixel_values_for_generate = [pixel_values]
                output_ids = self.model.generate(
                    input_ids,
                    pixel_values=pixel_values_for_generate,
                    attention_mask=attention_mask,
                    **gen_params,
                )
                prompt_len = input_ids.shape[1]
                if output_ids.shape[1] > prompt_len:
                    newly_generated_ids = output_ids[0, prompt_len:]
                elif output_ids.shape[1] == prompt_len:
                    print(
                        f"[WARNING] No new tokens generated! output_len={output_ids.shape[1]}, prompt_len={prompt_len}"
                    )
                    return "[No new tokens generated]"
                else:
                    newly_generated_ids = output_ids[0]
                decoded_text = self.text_tokenizer.decode(
                    newly_generated_ids, skip_special_tokens=True
                ).strip()
                if not decoded_text:
                    print(f"[WARNING] Decoded text is empty after all processing!")
                    return "[Empty decoded text]"
                return decoded_text
            except Exception as e:
                print(f"[ERROR] Ovis2Expert generation failed: {e}")
                return f"[Error: Ovis2Expert generation failed - {str(e)}]"
            finally:
                if was_training:
                    self.model.train()

    def generate_full(self, *args, **kwargs):
        raise NotImplementedError(
            "Ovis2Expert does not implement generate_full. Use extract + continue."
        )

    def forward_for_loss(self, *args, **kwargs):
        return None


class LateFusionExpert(BaseExpert):
    def __init__(
        self, model_name=cfg["MODEL"]["FUSION_EXPERT_MODEL_ID"], api_key=None, cost=0.3
    ):
        super().__init__(model_name, cost)
        self.api_key = api_key or cfg["MODEL"]["GEMINI_API_KEY"]
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            print(
                "WARN: Gemini API Key is not set. LateFusionExpert will be non-functional."
            )
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id_or_path}:generateContent?key={self.api_key}"
        self.headers = {"Content-Type": "application/json"}
        print(
            f"INFO: Initializing LateFusionExpert using Gemini API ({self.model_id_or_path})"
        )
        self.gate_feature_dim = 0

    def _convert_table_to_markdown_string(
        self, raw_table_data_single: dict | str
    ) -> str:
        if isinstance(raw_table_data_single, str):
            return raw_table_data_single
        elif (
            isinstance(raw_table_data_single, dict)
            and "header" in raw_table_data_single
            and ("rows" in raw_table_data_single)
        ):
            header = raw_table_data_single["header"]
            rows = raw_table_data_single["rows"]
            if not header or not isinstance(header, list):
                return ""
            markdown_parts = []
            markdown_parts.append("| " + " | ".join(map(str, header)) + " |")
            markdown_parts.append("| " + " | ".join(["---"] * len(header)) + " |")
            for row in rows:
                if isinstance(row, list) and len(row) == len(header):
                    markdown_parts.append("| " + " | ".join(map(str, row)) + " |")
            return "\n".join(markdown_parts)
        elif isinstance(raw_table_data_single, dict):
            try:
                return json.dumps(raw_table_data_single, indent=2)
            except TypeError:
                return str(raw_table_data_single)
        else:
            return str(raw_table_data_single)

    def generate_full(
        self,
        original_question_single: str,
        raw_table_data_single: dict | str,
        expert1_output_single: str,
        expert2_output_single: str,
        dataset_category: str | None = None,
        **generation_kwargs_override,
    ):
        full_table_data_str = self._convert_table_to_markdown_string(
            raw_table_data_single
        )
        if not full_table_data_str.strip():
            full_table_data_str = "[Table data could not be displayed or is empty]"
        extra_instruction = ""
        if dataset_category:
            dc = dataset_category.upper()
            print(f"INFO: Dataset category: {dc}")
            if "TABFACT" in dc:
                extra_instruction = '\nReminder: Generate a JSON response with an \'answer\' field containing either ["True"] or ["False"].'
            elif "INFOTABS" in dc:
                extra_instruction = '\nReminder: Generate a JSON response with an \'answer\' field containing exactly one of ["Entail"], ["Contradict"], or ["Neutral"].'
            elif "TABMWP" in dc:
                extra_instruction = "\nReminder: If a numeric answer is required, output only the number (no units)."
            elif "FETAQA" in dc:
                extra_instruction = "\nReminder: Provide a complete sentence as the answer, not just keywords or phrases."
        fusion_prompt = f"\n            You are a table question answering expert. You are given a question, the **full table data**, and two expert answers:\n            - Expert 1 is a text-based LLM, good at understanding table structure and language.\n            - Expert 2 is a vision-language model (VLM), good at extracting information from table images.\n\n            Your task:\n            - Analyze the **structure and content of the provided Full Table Data**.\n            - **Important: Do NOT directly answer the question using information from the Full Table Data.** Your main role is to synthesize the answers from Expert 1 and Expert 2.\n            - Carefully compare both expert answers.\n            - If the answers are complementary, merge their key information.\n            - If there is a conflict, use your understanding of the **table's structure (derived from Full Table Data)** and the **expert outputs** to reason about and select or generate the most accurate answer.\n            - Your response must be extremely concise and directly answer the original question, with no explanation, no prefix, and no extra words.{extra_instruction}\n\n            Question:\n            {original_question_single}\n\n            Full Table Data:\n            {full_table_data_str}\n\n            Expert 1 (Text LLM) Answer:\n            {expert1_output_single}\n\n            Expert 2 (VLM) Answer:\n            {expert2_output_single}\n\n            Final Answer:\n            "
        final_gen_cfg = {**cfg["GENERATION"], **generation_kwargs_override}
        fusion_max_tokens = generation_kwargs_override.get(
            "MAX_NEW_TOKENS",
            cfg["MODEL"].get(
                "FUSION_MAX_NEW_TOKENS", final_gen_cfg.get("MAX_NEW_TOKENS", 9999)
            ),
        )
        body = {
            "contents": [{"parts": [{"text": fusion_prompt}]}],
            "generationConfig": {
                "temperature": final_gen_cfg.get("TEMPERATURE", 0.0),
                "maxOutputTokens": fusion_max_tokens,
            },
        }
        if not self.api_key or self.api_key == "YOUR_GEMINI_API_KEY_HERE":
            return "[Error: Gemini API key not configured]"
        try:
            resp = requests.post(
                self.api_url, headers=self.headers, json=body, timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                text = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                return text or "[Error: Empty Gemini response]"
            detail = ""
            try:
                detail = resp.json().get("error", {}).get("message", "")
            except Exception:
                pass
            return f"[Error {resp.status_code}: {detail}]"
        except requests.exceptions.RequestException as e_req:
            return f"[Error: request failed – {e_req}]"
        except Exception as e:
            return f"[Error: unexpected fusion error – {e}]"

    def _load_model_resources(self):
        pass

    def extract_gate_features_and_intermediate_state(self, *args, **kwargs):
        return (None, None)

    def continue_generation_from_state(self, *args, **kwargs):
        raise NotImplementedError("LateFusionExpert uses generate_full only.")

    def forward_for_loss(self, **kwargs):
        return None
