"""Model loading and generation shared by evaluation and the demo.

The base model and the LoRA adapter are loaded once. Base-model outputs are
produced with the adapter temporarily disabled, so both conditions use the
exact same weights, quantisation and decoding settings.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.hardware import compute_dtype


class SupportAgent:
    def __init__(self, base_model: str, adapter: Optional[str] = None, use_4bit: bool = True):
        if adapter is not None and not Path(adapter, "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"No trained adapter found at '{adapter}'. Did training finish? "
                "It saves the adapter to <out>/final."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(adapter or base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cuda = torch.cuda.is_available()
        dtype = compute_dtype()
        quant = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            if use_4bit and cuda
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=quant, dtype=dtype, device_map="auto" if cuda else None
        )
        self.has_adapter = adapter is not None
        if self.has_adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        self.model = model

    @torch.no_grad()
    def generate(
        self,
        messages: list[dict],
        use_adapter: bool = True,
        max_new_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        if use_adapter and not self.has_adapter:
            raise ValueError("no adapter loaded")
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)

        do_sample = temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.9)

        ctx = self.model.disable_adapter() if (self.has_adapter and not use_adapter) else nullcontext()
        with ctx:
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
