import argparse
import math
import random
from typing import List, Optional, Tuple

from unsloth import FastLanguageModel
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--resume", default=False, action="store_true", help="resume last checkpoint")
    parser.add_argument("--hub", default=False, action="store_true", help="push to hub")
    parser.add_argument("--rescale", default=False, action="store_true", help="rescale codebook loss")
    parser.add_argument("--p-val", type=float, default=0.03, help="holdout ratio for validation split")

    parser.add_argument("--base", default="nytopop/1b_or_base")
    parser.add_argument("--read", default=False, action="store_true", help="use read examples")
    parser.add_argument("--improv", default=False, action="store_true", help="use improv examples")
    args = parser.parse_args()

    if not args.read and not args.improv:
        parser.error("at least one of [--read, --improv] are required")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=8192,
        load_in_4bit=False,
        full_finetuning=True,
        use_gradient_checkpointing="unsloth",
        dtype=None,
    )

    train, eval = [], []

    def labels(row):
        return dict(row, labels=row["input_ids"])

    if args.read:
        ds = load_dataset("nytopop/expresso-full-pruned", split="read")
        ds = ds.map(labels).train_test_split(test_size=args.p_val, seed=args.seed)
        train.append((ds["train"], 4))  # TODO: 12
        eval.append((ds["test"], 4))

    if args.improv:
        ds = load_dataset("nytopop/expresso-full-pruned", split="improv_4096")
        ds = ds.map(labels).train_test_split(test_size=args.p_val, seed=args.seed)
        train.append((ds["train"], 1))
        eval.append((ds["test"], 1))

    train_args = TrainingArguments(
        output_dir    = "output",
        eval_strategy = "epoch",
        eval_on_start = True,
        save_strategy = "epoch",
        load_best_model_at_end = True,

        # --- log steps to tensorboard ---
        logging_strategy = "steps",
        logging_steps    = 1,
        report_to        = "tensorboard",
        include_tokens_per_second     = True,
        include_num_input_tokens_seen = True,

        # --- hub settings ---
        push_to_hub      = args.hub,
        hub_private_repo = False,
        hub_strategy     = "every_save",

        # --- optimizer hyperparameters ---
        optim         = "lion_8bit",
        learning_rate = 1.85e-5,
        weight_decay  = 3e-2,
        adam_beta1    = 9.5e-1,
        adam_beta2    = 9.8e-1,

        # --- train run hyperparameters ---
        lr_scheduler_type = "cosine",
        warmup_ratio      = 5e-2,
        num_train_epochs  = 2,
        gradient_accumulation_steps = 15, # 12 min step

        # --- misc --- 
        per_device_train_batch_size = 1,  # NOTE: does nothing; we batch in mux
        per_device_eval_batch_size  = 1,  # NOTE: does nothing; we batch in mux
        seed = args.seed,
        bf16 = True,
        bf16_full_eval=True,
    )  # fmt: off

    if args.rescale:
        weight = torch.ones(model.config.vocab_size, device=model.device)
        weight[132362:136458] = 0.50  # 24hz
        weight[136458:144650] = 0.25  # 48hz
        weight[144650:148746] = 0.50  # 24hz
        weight[148746:156938] = 0.25  # 48hz

        def compute_loss_func(*args, **kwargs):
            return causal_lm_loss(*args, **kwargs, weight=weight)
    else:
        compute_loss_func = None

    trainer = MuxTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=eval,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
        args=train_args,
        compute_loss_func=compute_loss_func,
    )

    # start training!
    trainer.train(resume_from_checkpoint=args.resume)

    if not train_args.push_to_hub:
        trainer.save_model(train_args.output_dir)


def causal_lm_loss(outputs, labels, num_items_in_batch, weight=None):
    # shift for causal
    logits = outputs["logits"][..., :-1, :].contiguous()  # [B, T-1, C]
    labels = labels[..., 1:].contiguous()  # [B, T-1]

    [B, T, C] = logits.shape
    logits = logits.reshape(B * T, C)
    labels = labels.reshape(B * T)

    if num_items_in_batch is None:
        return F.cross_entropy(logits, labels, weight=weight, ignore_index=-100, reduction="mean")

    loss = F.cross_entropy(logits, labels, weight=weight, ignore_index=-100, reduction="sum")
    loss = loss / num_items_in_batch

    return loss


class MuxTrainer(Trainer):
    """A Trainer that multiplexes amongst datasets with individual batch sizes."""

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("MuxTrainer: training requires a train_dataset")

        return DataLoader(
            ConcatDataset([ds for ds, _ in self.train_dataset]),
            batch_sampler=Mux(self.train_dataset, seed=self.args.seed),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[List[Tuple[Dataset, int]]]) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("MuxTrainer: evaluation requires an eval_dataset")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        return DataLoader(
            ConcatDataset([ds for ds, _ in eval_dataset]),
            batch_sampler=Mux(eval_dataset, seed=self.args.seed),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class Mux(Sampler):
    """A batch Sampler that multiplexes amongst datasets with individual batch sizes."""

    def __init__(self, dss: List[Tuple[Dataset, int]], seed: int):
        self.dss = dss
        self.rng = random.Random(seed)

    def __iter__(self):
        ix, offset = [], 0

        for ds, k in self.dss:
            jx = list(range(offset, len(ds) + offset))
            self.rng.shuffle(jx)  # shuffle rows

            ix.extend([jx[i : i + k] for i in range(0, len(jx), k)])
            offset += len(ds)

        self.rng.shuffle(ix)  # shuffle batches

        return iter(ix)

    def __len__(self):
        return sum([math.ceil(len(ds) / k) for ds, k in self.dss])


if __name__ == "__main__":
    main()
