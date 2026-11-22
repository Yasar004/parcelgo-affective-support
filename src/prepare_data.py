"""Build the finetuning dataset from the Customer Support on Twitter corpus (TWCS).

Pipeline
1. Keep inbound customer tweets about delayed deliveries (keyword filter).
2. Pair each with the first reply written by the company's support account.
3. Clean both texts (handles, URLs, agent signatures) and drop replies that
   claim capabilities the ParcelGo chatbot lacks (checking tracking data,
   stating arrival dates, processing refunds), so finetuning does not teach
   behaviour the system prompt forbids.
4. Weakly label the customer tweet with an affect label and an escalation flag.
5. Balance the affect classes and write train/val/test JSONL files in TRL's
   conversational prompt-completion format.

Two labelers are available:
- `model` (default): pretrained Twitter irony detector + GoEmotions classifier,
  combined with delivery-context rules.
- `heuristic`: lexicon rules only; needs no downloads, useful for quick tests.

Download TWCS (`twcs.csv`) from Kaggle:
https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

Usage:
    python -m src.prepare_data --twcs data/twcs.csv --out data/processed
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from src.prompts import SYSTEM_PROMPT, format_target, makes_unsupported_claim

# --------------------------------------------------------------------------
# Filtering and cleaning
# --------------------------------------------------------------------------

DELAY_RE = re.compile(
    r"\b(late|delay(ed)?|still (not|hasn'?t|haven'?t|waiting)|where('?s| is) my "
    r"(order|package|parcel|delivery)|not (arrived|delivered|here)|never (arrived|came)|"
    r"missed delivery|supposed to (arrive|be delivered)|tracking)\b",
    re.IGNORECASE,
)
DELIVERY_RE = re.compile(r"\b(order|package|parcel|delivery|deliver\w*|shipping|shipment|courier)\b", re.IGNORECASE)

HANDLE_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
# Agent signatures at the end of brand replies, e.g. "^JD", "-AB", "*KC", "/SB"
SIGNATURE_RE = re.compile(r"\s*[\^\-\*/~]\s?[A-Z]{1,3}\d?\s*$")
WS_RE = re.compile(r"\s+")


def clean_text(text: str, strip_signature: bool = False) -> str:
    text = HANDLE_RE.sub("", text)
    text = URL_RE.sub("", text)
    if strip_signature:
        text = SIGNATURE_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def is_delay_complaint(text: str) -> bool:
    return bool(DELAY_RE.search(text)) and bool(DELIVERY_RE.search(text))


def load_pairs(
    twcs_path: Path, max_candidates: int | None = None, keep_unsupported_claims: bool = False
) -> pd.DataFrame:
    """Return (customer_text, reply_text) pairs for delay complaints."""
    df = pd.read_csv(
        twcs_path,
        usecols=["tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "string", "text": "string",
               "response_tweet_id": "string"},
    )
    df["inbound"] = df["inbound"].astype(str).str.lower().eq("true")

    # Conversation starters only: the first customer message best matches a
    # single-turn support query.
    customers = df[df["inbound"] & df["in_response_to_tweet_id"].isna()].copy()
    customers = customers[customers["text"].fillna("").map(is_delay_complaint)]
    customers = customers[customers["response_tweet_id"].notna()]
    if max_candidates:
        customers = customers.head(max_candidates)

    customers["first_response_id"] = (
        customers["response_tweet_id"].str.split(",").str[0].str.strip().astype("int64")
    )
    brand = df[~df["inbound"]][["tweet_id", "text"]].rename(
        columns={"tweet_id": "first_response_id", "text": "reply_raw"}
    )
    pairs = customers.merge(brand, on="first_response_id", how="inner")

    pairs["customer_text"] = pairs["text"].map(clean_text)
    pairs["reply_text"] = pairs["reply_raw"].map(lambda t: clean_text(t, strip_signature=True))
    pairs = pairs[(pairs["customer_text"].str.len() >= 15) & (pairs["reply_text"].str.len() >= 20)]
    if not keep_unsupported_claims:
        claims = pairs["reply_text"].map(makes_unsupported_claim)
        print(f"dropping {int(claims.sum())} of {len(pairs)} pairs whose reply claims unavailable capabilities")
        pairs = pairs[~claims]
    return pairs[["tweet_id", "customer_text", "reply_text"]].drop_duplicates("customer_text")


# --------------------------------------------------------------------------
# Weak labelling
# --------------------------------------------------------------------------

POSITIVE_WORDS_RE = re.compile(
    r"\b(great|thanks?( so much| a lot)?|love|wow|amazing|awesome|brilliant|fantastic|"
    r"perfect|wonderful|impressive|nice|well done|good job|bravo|genius)\b",
    re.IGNORECASE,
)
SARCASM_MARKERS_RE = re.compile(r"(🙄|🙃|👏|😒|/s\b|\byeah right\b|\boh great\b|\bso glad\b|\bjust what i needed\b)",
                                re.IGNORECASE)
NEGATIVE_CONTEXT_RE = re.compile(
    r"\b(late|delay\w*|still|again|waiting|never|weeks?|days?|lost|missing|nothing)\b", re.IGNORECASE
)
FRUSTRATION_RE = re.compile(
    r"\b(ridiculous|unacceptable|joke|worst|terrible|awful|useless|pathetic|disgusting|"
    r"fed up|sick of|annoy\w*|angry|furious|frustrat\w*|disappoint\w*|horrible|wtf|"
    r"seriously|no one|nobody|every time|third time|again)\b|!{2,}",
    re.IGNORECASE,
)
ESCALATION_RE = re.compile(
    r"\b(refund|money back|compensat\w*|cancel\w*|chargeback|lawyer|legal|sue|"
    r"complaint|manager|supervisor|report you|trading standards|ombudsman)\b",
    re.IGNORECASE,
)


def heuristic_affect(text: str) -> str:
    has_positive = bool(POSITIVE_WORDS_RE.search(text))
    has_negative_ctx = bool(NEGATIVE_CONTEXT_RE.search(text))
    if SARCASM_MARKERS_RE.search(text) or (has_positive and has_negative_ctx and not text.lower().startswith("thank")):
        return "sarcastic"
    caps_words = [w for w in text.split() if len(w) > 3 and w.isupper()]
    if FRUSTRATION_RE.search(text) or len(caps_words) >= 2:
        return "frustrated"
    return "neutral"


def heuristic_escalate(text: str, affect: str, anger_score: float = 0.0) -> str:
    if ESCALATION_RE.search(text) or anger_score >= 0.6:
        return "yes"
    if affect == "frustrated" and re.search(r"\b(again|third time|every time|weeks?)\b", text, re.IGNORECASE):
        return "yes"
    return "no"


class ModelLabeler:
    """Combines a Twitter irony classifier with a GoEmotions classifier.

    Classifier probabilities alone over-fire on Twitter text, so they are
    combined with delivery-context rules: irony only counts as sarcasm when the
    tweet also refers to a negative delivery situation.
    """

    IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
    EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
    NEGATIVE_EMOTIONS = ("anger", "annoyance", "disappointment", "disapproval")

    def __init__(self, device: int | str | None = None, batch_size: int = 64):
        from transformers import pipeline  # imported lazily: heuristic mode needs no torch

        kwargs = dict(top_k=None, truncation=True, batch_size=batch_size)
        if device is not None:
            # transformers >= 5 rejects the string "0"; accept "0", 0 or "cuda:0"
            kwargs["device"] = int(device) if str(device).isdigit() else device
        self.irony = pipeline("text-classification", model=self.IRONY_MODEL, **kwargs)
        self.emotion = pipeline("text-classification", model=self.EMOTION_MODEL, **kwargs)

    @staticmethod
    def _scores(result) -> dict[str, float]:
        return {r["label"].lower(): r["score"] for r in result}

    def label(self, texts: list[str]) -> list[tuple[str, str]]:
        irony_results = self.irony(texts)
        emotion_results = self.emotion(texts)
        labels = []
        for text, ir, em in zip(texts, irony_results, emotion_results):
            ir_s, em_s = self._scores(ir), self._scores(em)
            # Some checkpoint versions name the classes LABEL_0/LABEL_1 (1 = irony)
            irony_p = ir_s.get("irony", ir_s.get("label_1", 0.0))
            neg_p = max(em_s.get(e, 0.0) for e in self.NEGATIVE_EMOTIONS)
            anger = em_s.get("anger", 0.0)

            heur = heuristic_affect(text)
            if irony_p >= 0.6 and NEGATIVE_CONTEXT_RE.search(text) and (
                POSITIVE_WORDS_RE.search(text) or SARCASM_MARKERS_RE.search(text) or heur == "sarcastic"
            ):
                affect = "sarcastic"
            elif neg_p >= 0.3 or heur == "frustrated":
                affect = "frustrated"
            else:
                affect = "neutral"
            labels.append((affect, heuristic_escalate(text, affect, anger)))
        return labels


def heuristic_label(texts: list[str]) -> list[tuple[str, str]]:
    out = []
    for t in texts:
        affect = heuristic_affect(t)
        out.append((affect, heuristic_escalate(t, affect)))
    return out


# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def to_record(customer_text: str, reply_text: str, affect: str, escalate: str, tweet_id: int) -> dict:
    """TRL conversational prompt-completion record (loss only on the completion)."""
    return {
        "id": int(tweet_id),
        "affect": affect,
        "escalate": escalate,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": customer_text},
        ],
        "completion": [{"role": "assistant", "content": format_target(affect, escalate, reply_text)}],
    }


def balance(records: list[dict], per_class: int, seed: int) -> list[dict]:
    """Downsample each affect class to at most `per_class` examples."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for r in records:
        by_class.setdefault(r["affect"], []).append(r)
    out = []
    for items in by_class.values():
        rng.shuffle(items)
        out.extend(items[:per_class])
    rng.shuffle(out)
    return out


def split(records: list[dict], val_frac: float, test_frac: float) -> dict[str, list[dict]]:
    n = len(records)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    return {
        "test": records[:n_test],
        "val": records[n_test:n_test + n_val],
        "train": records[n_test + n_val:],
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--twcs", type=Path, required=True, help="path to twcs.csv")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--labeler", choices=["model", "heuristic"], default="model")
    p.add_argument("--per-class", type=int, default=1500, help="max examples per affect class")
    p.add_argument("--max-candidates", type=int, default=None, help="cap on filtered tweets (for quick runs)")
    p.add_argument("--keep-unsupported-claims", action="store_true",
                   help="keep company replies that claim tracking access, refunds etc. (off by default)")
    p.add_argument("--oversample-sarcastic", type=int, default=1,
                   help="repeat sarcastic TRAIN examples this many times (val/test untouched)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--device", default=None, help="e.g. 0 or cuda:0 for the model labeler")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pairs = load_pairs(args.twcs, args.max_candidates, args.keep_unsupported_claims)
    print(f"delay-complaint pairs after filtering: {len(pairs)}")

    texts = pairs["customer_text"].tolist()
    if args.labeler == "model":
        labels = ModelLabeler(device=args.device).label(texts)
    else:
        labels = heuristic_label(texts)

    records = [
        to_record(c, r, a, e, tid)
        for (tid, c, r), (a, e) in zip(pairs[["tweet_id", "customer_text", "reply_text"]].itertuples(index=False), labels)
    ]
    print("label distribution before balancing:", dict(Counter(r["affect"] for r in records)))

    records = balance(records, args.per_class, args.seed)
    splits = split(records, args.val_frac, args.test_frac)
    if args.oversample_sarcastic > 1:
        # Oversample after splitting so duplicates never leak into val/test.
        sarcastic = [r for r in splits["train"] if r["affect"] == "sarcastic"]
        splits["train"] = splits["train"] + sarcastic * (args.oversample_sarcastic - 1)
        random.Random(args.seed).shuffle(splits["train"])
    for name, recs in splits.items():
        write_jsonl(args.out / f"{name}.jsonl", recs)
        dist = dict(Counter(r["affect"] for r in recs))
        esc = sum(r["escalate"] == "yes" for r in recs)
        print(f"{name}: {len(recs)} examples, affect={dist}, escalate=yes: {esc}")

    meta = {"labeler": args.labeler, "oversample_sarcastic": args.oversample_sarcastic, "keep_unsupported_claims": args.keep_unsupported_claims, "per_class": args.per_class, "seed": args.seed, "source": str(args.twcs)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
