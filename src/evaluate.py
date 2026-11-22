"""Compare the base model and the LoRA-finetuned model.

Two evaluation sets:
- `scenarios`: the hand-written user scenarios and crash tests in
  data/scenarios.json (gold labels written by hand).
- `test`: the held-out TWCS test split (weak labels, so treat its scores as
  agreement with the labeler rather than as ground truth).

Automatic metrics per condition: format compliance, affect accuracy and
macro-F1, sarcasm recall, escalation accuracy, and the rates of unauthorised
promises and unsupported capability claims (keyword heuristics).
Response quality cannot be scored automatically, so a Markdown sheet is
written for manual rating on the group's shared criteria: tone, usefulness,
factual grounding and capability compliance.

Usage:
    python -m src.evaluate --adapter outputs/support-lora/final --set scenarios
    python -m src.evaluate --adapter outputs/support-lora/final --set test --limit 200
    python -m src.evaluate --set scenarios            # base model only
    python -m src.evaluate --rescore results/scenarios_<timestamp>   # recompute metrics, no GPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from src.prompts import (
    AFFECT_LABELS,
    DEFAULT_MODEL,
    build_messages,
    makes_unauthorised_promise,
    makes_unsupported_claim,
    parse_output,
)


def load_scenarios(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["scenarios"]


def load_test_split(path: Path, limit: int | None) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            items.append({
                "id": str(r["id"]), "scenario": "twcs_test", "probe": "",
                "message": r["prompt"][-1]["content"],
                "gold_affect": r["affect"], "gold_escalate": r["escalate"],
            })
    return items[:limit] if limit else items


def score(rows: list[dict]) -> dict:
    n = len(rows)
    # Malformed answers count as wrong rather than being dropped.
    pred_aff = [r["affect"] or "invalid" for r in rows]
    gold_aff = [r["gold_affect"] for r in rows]
    pred_esc = [r["escalate"] or "invalid" for r in rows]
    gold_esc = [r["gold_escalate"] for r in rows]
    return {
        "n": n,
        "format_compliance": sum(r["well_formed"] for r in rows) / n,
        "affect_accuracy": accuracy_score(gold_aff, pred_aff),
        "affect_macro_f1": f1_score(gold_aff, pred_aff, labels=list(AFFECT_LABELS), average="macro", zero_division=0),
        "sarcasm_recall": (
            sum(p == "sarcastic" for p, g in zip(pred_aff, gold_aff) if g == "sarcastic")
            / max(1, gold_aff.count("sarcastic"))
        ),
        "escalate_accuracy": accuracy_score(gold_esc, pred_esc),
        "unauthorised_promise_rate": sum(r["unauthorised_promise"] for r in rows) / n,
        "unsupported_claim_rate": sum(r["unsupported_claim"] for r in rows) / n,
    }


def rating_sheet(rows_by_cond: dict[str, list[dict]]) -> str:
    """Markdown table for manual rating, one row per item and condition."""
    lines = [
        "# Manual rating sheet",
        "",
        "Same criteria as the group's other experiments. Rate each reply 1-5 for:",
        "",
        "- **Tone**: fits the customer's affect; acknowledges frustration, does not answer sarcasm as praise or copy it",
        "- **Usefulness**: gives an appropriate, concrete next step",
        "- **Grounding**: states only what the chatbot can know (no invented tracking status, dates or policies)",
        "- **Capability**: stays within ParcelGo's limits (no tracking access, no refunds, no hand-over)",
        "",
        "The `Flags` column shows the automatic heuristics: P = unauthorised promise, C = unsupported claim.",
        "",
        "| ID | Probe | Condition | Gold (affect/esc) | Pred (affect/esc) | Flags | Reply | Tone | Useful. | Ground. | Capab. |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    conds = list(rows_by_cond)
    for i in range(len(rows_by_cond[conds[0]])):
        for c in conds:
            r = rows_by_cond[c][i]
            reply = (r["reply"] or r["raw_output"]).replace("|", "\\|").replace("\n", " ")
            flags = "".join(f for f, on in (("P", r["unauthorised_promise"]), ("C", r["unsupported_claim"])) if on)
            lines.append(
                f"| {r['id']} | {r['probe']} | {c} | {r['gold_affect']}/{r['gold_escalate']} | "
                f"{r['affect']}/{r['escalate']} | {flags} | {reply} |  |  |  |  |"
            )
    return "\n".join(lines) + "\n"


def rescore(run_dir: Path) -> None:
    """Recompute parsing, flags and metrics from saved raw outputs (no model needed)."""
    rows_by_cond: dict[str, list[dict]] = {}
    with (run_dir / "outputs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            parsed = parse_output(r["raw_output"])
            r.update(
                affect=parsed.affect, escalate=parsed.escalate, reply=parsed.reply,
                well_formed=parsed.well_formed,
                unauthorised_promise=makes_unauthorised_promise(parsed, r["raw_output"]),
                unsupported_claim=makes_unsupported_claim(parsed.reply or r["raw_output"]),
            )
            rows_by_cond.setdefault(r["condition"], []).append(r)
    write_results(run_dir, rows_by_cond, per_scenario=any(r["scenario"] != "twcs_test"
                                                          for rows in rows_by_cond.values() for r in rows))
    print(f"rescored {run_dir}")


def write_results(out_dir: Path, rows_by_cond: dict[str, list[dict]], per_scenario: bool) -> None:
    with (out_dir / "outputs.jsonl").open("w", encoding="utf-8") as f:
        for rows in rows_by_cond.values():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {cond: score(rows) for cond, rows in rows_by_cond.items()}
    if per_scenario:
        for cond, rows in rows_by_cond.items():
            for sc in sorted({r["scenario"] for r in rows}):
                metrics[f"{cond}/{sc}"] = score([r for r in rows if r["scenario"] == sc])
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "rating_sheet.md").write_text(rating_sheet(rows_by_cond), encoding="utf-8")
    print("\n" + json.dumps({k: v for k, v in metrics.items() if "/" not in k}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="path to the trained LoRA adapter (omit for base only)")
    p.add_argument("--set", choices=["scenarios", "test"], default="scenarios")
    p.add_argument("--scenarios", type=Path, default=Path("data/scenarios.json"))
    p.add_argument("--test-file", type=Path, default=Path("data/processed/test.jsonl"))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding (reproducible)")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--rescore", type=Path, default=None, help="recompute metrics for an existing run folder")
    args = p.parse_args()

    if args.rescore:
        rescore(args.rescore)
        return

    from src.generation import SupportAgent  # lazy: --rescore needs no torch

    items = load_scenarios(args.scenarios) if args.set == "scenarios" else load_test_split(args.test_file, args.limit)
    agent = SupportAgent(args.model, args.adapter)
    conditions = {"base": False} | ({"finetuned": True} if args.adapter else {})

    rows_by_cond: dict[str, list[dict]] = {}
    for cond, use_adapter in conditions.items():
        rows = []
        for item in items:
            messages = build_messages(item["message"], item.get("history"))
            raw = agent.generate(messages, use_adapter=use_adapter, temperature=args.temperature)
            parsed = parse_output(raw)
            rows.append({
                **{k: item.get(k) for k in ("id", "scenario", "probe", "message", "gold_affect", "gold_escalate")},
                "condition": cond,
                "raw_output": raw,
                "affect": parsed.affect,
                "escalate": parsed.escalate,
                "reply": parsed.reply,
                "well_formed": parsed.well_formed,
                "unauthorised_promise": makes_unauthorised_promise(parsed, raw),
                "unsupported_claim": makes_unsupported_claim(parsed.reply or raw),
            })
            print(f"[{cond}] {item['id']}: {parsed.affect}/{parsed.escalate} | {(parsed.reply or raw)[:90]}")
        rows_by_cond[cond] = rows

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = args.out / f"{args.set}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_results(out_dir, rows_by_cond, per_scenario=args.set == "scenarios")
    print(f"results written to {out_dir}")


if __name__ == "__main__":
    main()
