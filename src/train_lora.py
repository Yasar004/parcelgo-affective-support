"""QLoRA finetuning of a small open instruct model on the prepared data.

The base model is loaded in 4-bit (NF4) and only low-rank adapters on the
attention and MLP projections are trained, so a 1.5B-parameter model fits on
a free Colab T4 (16 GB). The loss is computed on the assistant completion only
(the system prompt and customer message are masked out).

Usage:
    python -m src.train_lora --data data/processed --out outputs/qwen-support-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer

from src.hardware import compute_dtype as pick_dtype
from src.hardware import native_bf16, patch_trl_for_older_gpus, upcast_trainable_params

from src.prompts import DEFAULT_MODEL  # noqa: E402


def load_model(model_name: str, use_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16 = native_bf16()  # False on Colab's T4, which only emulates bf16
    compute_dtype = pick_dtype()
    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if use_4bit
        else None
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant,
        dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, tokenizer, bf16


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("outputs/support-lora"))
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-4bit", action="store_true", help="train in full precision LoRA instead of QLoRA")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    patch_trl_for_older_gpus()
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.data / "train.jsonl"), "validation": str(args.data / "val.jsonl")},
    )
    # Keep only the fields TRL needs; labels stay in the files for analysis.
    dataset = dataset.select_columns(["prompt", "completion"])

    model, tokenizer, bf16 = load_model(args.model, use_4bit=not args.no_4bit)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    config = SFTConfig(
        output_dir=str(args.out),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # float < 1 = fraction of total steps (transformers >= 5)
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    upcast_trainable_params(trainer.model)  # must run after SFTTrainer casts adapters to bf16
    trainer.model.print_trainable_parameters()
    trainer.train()

    final_dir = args.out / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    (final_dir / "train_args.json").write_text(json.dumps({**vars(args), "data": str(args.data), "out": str(args.out)}, indent=2))
    print(f"adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
