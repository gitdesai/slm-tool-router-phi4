import os, json, yaml, argparse, random, re
from typing import Any, Dict, List, Tuple, Callable

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer

# Temporary block
import accelerate
import inspect

if "keep_torch_compile" not in inspect.signature(accelerate.Accelerator.unwrap_model).parameters:
    _orig = accelerate.Accelerator.unwrap_model
    def _unwrap(self, model, *args, **kwargs):
        kwargs.pop("keep_torch_compile", None)
        return _orig(self, model, *args, **kwargs)
    accelerate.Accelerator.unwrap_model = _unwrap
#

# ----------------------------
# Repro / Seed
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# IO helpers
# ----------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ----------------------------
# tools.yaml parsing
# Your format: {service: [tool1, tool2, ...]}
# ----------------------------
def load_tools_yaml(tools_yaml: str) -> Tuple[Dict[str, List[str]], List[str]]:
    with open(tools_yaml, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    if not isinstance(obj, dict):
        raise ValueError("tools.yaml must be a dict of {service: [tool, ...]}")

    tools_by_service: Dict[str, List[str]] = {}
    flat: List[str] = []
    seen = set()

    for svc, tools in obj.items():
        if not isinstance(tools, list):
            continue
        tools_by_service[svc] = []
        for t in tools:
            if isinstance(t, str):
                tools_by_service[svc].append(t)
                if t not in seen:
                    flat.append(t)
                    seen.add(t)

    return tools_by_service, flat

def format_tools_grouped(tools_by_service: Dict[str, List[str]]) -> str:
    out = []
    for svc, tools in tools_by_service.items():
        out.append(f"{svc.upper()}:")
        for t in tools:
            out.append(f"- {t}")
        out.append("")
    return "\n".join(out).strip()

# ----------------------------
# Tool name normalization
# Supports datasets that might contain prefixed names like calendar.list_events
# Your tools.yaml is unprefixed like list_events
# ----------------------------
def normalize_tool_name(tool: Any) -> Any:
    if not isinstance(tool, str):
        return tool
    return tool.split(".")[-1].strip()

# ----------------------------
# Stratified split per tool (single), per sequence (multi)
# ----------------------------
def stratified_split(
    records: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], str],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        buckets.setdefault(key_fn(r), []).append(r)

    train, test = [], []
    for k, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        if n < 2:
            train.extend(items)  # can't split
            continue
        n_test = max(1, int(round(n * test_ratio)))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test

def key_single(rec: Dict[str, Any]) -> str:
    return normalize_tool_name(rec["output"]["tool"])

def key_multi(rec: Dict[str, Any]) -> str:
    steps = rec["output"]["steps"]
    seq = [normalize_tool_name(s["tool"]) for s in steps]
    return " :: ".join(seq)

# ----------------------------
# Prompts
# ----------------------------
SINGLE_HINT = """Return ONLY valid JSON:
{"tool":"<tool_name>","arguments":{...}}
No extra text. Use {} if arguments are unknown or you are unsure, still return valid JSON with {} as arguments."""

MULTI_HINT = """Return ONLY valid JSON:
{"steps":[{"tool":"<tool_name>","arguments":{...}}, ...]}
1 to 3 steps max. No extra text. If you are unsure about arguments for any step, use {}."""

def build_prompt_single(user_input: str, tools_by_service: Dict[str, List[str]]) -> str:
    return f"""You select the correct Google Workspace MCP tool.

Available tools (grouped):
{format_tools_grouped(tools_by_service)}

Rules:
- Choose exactly ONE tool from the list above.
- Output strict JSON only.
- If arguments are not specified, output an empty object {{}}.

{SINGLE_HINT}

User request: {user_input}
JSON:
"""

def build_prompt_multi(user_input: str, tools_by_service: Dict[str, List[str]]) -> str:
    return f"""You plan an ordered multi-step sequence of Google Workspace MCP tool calls.

Available tools (grouped):
{format_tools_grouped(tools_by_service)}

Rules:
- Output 1 to 3 steps maximum.
- Each step tool must be from the list above.
- Output strict JSON only.

{MULTI_HINT}

User request: {user_input}
JSON:
"""

# ----------------------------
# JSON extraction (robust)
# ----------------------------
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = text.replace("```json", "```").replace("```JSON", "```")
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
    m = JSON_RE.search(text)
    if not m:
        raise ValueError("No JSON object found.")
    return json.loads(m.group(0))

@torch.no_grad()
def generate_json(model, tok, prompt: str, max_new_tokens: int = 256) -> Dict[str, Any]:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        pad_token_id=tok.eos_token_id,
    )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    return extract_json(decoded[len(prompt):] if decoded.startswith(prompt) else decoded)

# ----------------------------
# Model loading (QLoRA)
# ----------------------------
def load_base_model_and_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a GPU runtime for QLoRA (bitsandbytes 4-bit).")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(model)
    model.eval()
    return model, tok

def pick_lora_targets(model) -> List[str]:
    candidates = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    present = set()
    for name, _ in model.named_modules():
        for c in candidates:
            if name.endswith(c):
                present.add(c)
    return sorted(present) if present else candidates

# ----------------------------
# Evaluation
# ----------------------------
def eval_single(records, model, tok, tools_set: set, tools_by_service):
    total = len(records)
    ok_json = ok_in_set = ok_tool = 0
    details = []

    for r in records:
        gold_tool = normalize_tool_name(r["output"]["tool"])
        prompt = build_prompt_single(r["input"], tools_by_service)

        row = {"input": r["input"], "gold": {"tool": gold_tool, "arguments": r["output"].get("arguments", {})},
               "pred": None, "error": None}

        try:
            pred = generate_json(model, tok, prompt)
            pred_tool = normalize_tool_name(pred.get("tool"))
            pred_args = pred.get("arguments", {}) if isinstance(pred.get("arguments", {}), dict) else {}

            row["pred"] = {"tool": pred_tool, "arguments": pred_args}
            ok_json += 1
            if pred_tool in tools_set:
                ok_in_set += 1
            if pred_tool == gold_tool:
                ok_tool += 1

        except Exception as e:
            row["error"] = str(e)

        details.append(row)

    return {
        "summary": {
            "total": total,
            "json_valid_rate": ok_json / max(1, total),
            "tool_in_set_rate": ok_in_set / max(1, total),
            "tool_accuracy": ok_tool / max(1, total),
        },
        "details": details,
    }

def eval_multi(records, model, tok, tools_set: set, tools_by_service):
    total = len(records)
    ok_json = exact_seq = step_count_ok = first_step_ok = hallucinated = 0
    details = []

    for r in records:
        gold_steps = r["output"]["steps"]
        gold_seq = [normalize_tool_name(s["tool"]) for s in gold_steps]

        prompt = build_prompt_multi(r["input"], tools_by_service)
        row = {"input": r["input"], "gold": {"steps": [{"tool": t, "arguments": s.get("arguments", {})}
                                                      for t, s in zip(gold_seq, gold_steps)]},
               "pred": None, "error": None}

        try:
            pred = generate_json(model, tok, prompt)
            steps = pred.get("steps", [])
            if not isinstance(steps, list):
                raise ValueError("'steps' must be a list")

            pred_seq = []
            norm_steps = []
            for s in steps:
                if not isinstance(s, dict):
                    continue
                t = normalize_tool_name(s.get("tool"))
                a = s.get("arguments", {}) if isinstance(s.get("arguments", {}), dict) else {}
                pred_seq.append(t)
                norm_steps.append({"tool": t, "arguments": a})

            row["pred"] = {"steps": norm_steps}
            ok_json += 1

            if any((t not in tools_set) for t in pred_seq):
                hallucinated += 1
            if len(pred_seq) == len(gold_seq):
                step_count_ok += 1
            if pred_seq == gold_seq:
                exact_seq += 1
            if pred_seq and gold_seq and pred_seq[0] == gold_seq[0]:
                first_step_ok += 1

        except Exception as e:
            row["error"] = str(e)

        details.append(row)

    return {
        "summary": {
            "total": total,
            "json_valid_rate": ok_json / max(1, total),
            "hallucinated_tool_rate": hallucinated / max(1, total),
            "step_count_accuracy": step_count_ok / max(1, total),
            "sequence_exact_match": exact_seq / max(1, total),
            "first_step_accuracy": first_step_ok / max(1, total),
        },
        "details": details,
    }

# ----------------------------
# Fine-tuning (SFT + LoRA/QLoRA)
# ----------------------------
def sft_text_single(rec, tools_by_service):
    prompt = build_prompt_single(rec["input"], tools_by_service)
    target = json.dumps(
        {"tool": normalize_tool_name(rec["output"]["tool"]), "arguments": rec["output"].get("arguments", {})},
        ensure_ascii=False
    )
    return prompt + target

def sft_text_multi(rec, tools_by_service):
    prompt = build_prompt_multi(rec["input"], tools_by_service)
    target = json.dumps(
        {"steps": [{"tool": normalize_tool_name(s["tool"]), "arguments": s.get("arguments", {})}
                   for s in rec["output"]["steps"]]},
        ensure_ascii=False
    )
    return prompt + target

def finetune(train_records, model_id, tools_by_service, out_dir, epochs=2, seed=42, multistep=False):
    set_seed(seed)
    base, tok = load_base_model_and_tokenizer(model_id)

    targets = pick_lora_targets(base)
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(base, lora)

    texts = [sft_text_multi(r, tools_by_service) if multistep else sft_text_single(r, tools_by_service)
             for r in train_records]
    ds = Dataset.from_dict({"text": texts})

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        args=args,
        packing=False,
        max_seq_length=1024,
    )
    trainer.train()
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir

def load_with_adapter(model_id, adapter_dir):
    base, tok = load_base_model_and_tokenizer(model_id)
    tuned = PeftModel.from_pretrained(base, adapter_dir)
    tuned.eval()
    return tuned, tok

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools_yaml", default="tools.yaml")
    ap.add_argument("--single_jsonl", default="tool_routing_synthetic.jsonl")
    ap.add_argument("--multi_jsonl", default="tool_routing_multistep.jsonl")
    ap.add_argument("--model_id", default="microsoft/Phi-4-mini-instruct")
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--do_train", action="store_true")
    ap.add_argument("--do_multi", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Validate required files exist
    if not os.path.exists(args.tools_yaml):
        raise FileNotFoundError(f"Missing {args.tools_yaml}")
    if not os.path.exists(args.single_jsonl):
        raise FileNotFoundError(f"Missing {args.single_jsonl}")
    if args.do_multi and not os.path.exists(args.multi_jsonl):
        raise FileNotFoundError(f"Missing {args.multi_jsonl}")

    tools_by_service, tools_flat = load_tools_yaml(args.tools_yaml)
    tools_set = set(tools_flat)
    print(f"[INFO] tools loaded: {len(tools_flat)}")

    report_dir = os.path.join(args.out_dir, "reports")
    split_dir = os.path.join(args.out_dir, "splits")
    ensure_dir(report_dir)
    ensure_dir(split_dir)

    # -------- Phase 1: single --------
    single = read_jsonl(args.single_jsonl)
    train_s, test_s = stratified_split(single, key_fn=key_single, test_ratio=args.test_ratio, seed=args.seed)
    write_jsonl(os.path.join(split_dir, "train_single.jsonl"), train_s)
    write_jsonl(os.path.join(split_dir, "test_single.jsonl"), test_s)
    print(f"[SPLIT single] train={len(train_s)} test={len(test_s)}")

    if 1:
      base, tok = load_base_model_and_tokenizer(args.model_id)
      baseline_single = eval_single(test_s, base, tok, tools_set, tools_by_service)
      with open(os.path.join(report_dir, "baseline_single_summary.json"), "w", encoding="utf-8") as f:
        json.dump(baseline_single["summary"], f, indent=2)
      write_jsonl(os.path.join(report_dir, "baseline_single_details.jsonl"), baseline_single["details"])
      print("[BASELINE single]", json.dumps(baseline_single["summary"], indent=2))

    if args.do_train:
        adapter_single = os.path.join(args.out_dir, "adapter_single")
        print("[TRAIN] single-step...")
        finetune(train_s, args.model_id, tools_by_service, adapter_single, epochs=args.epochs, seed=args.seed, multistep=False)

        tuned, tok2 = load_with_adapter(args.model_id, adapter_single)
        post_single = eval_single(test_s, tuned, tok2, tools_set, tools_by_service)
        with open(os.path.join(report_dir, "post_single_summary.json"), "w", encoding="utf-8") as f:
            json.dump(post_single["summary"], f, indent=2)
        write_jsonl(os.path.join(report_dir, "post_single_details.jsonl"), post_single["details"])
        print("[POST single]", json.dumps(post_single["summary"], indent=2))

    base, tok = load_base_model_and_tokenizer(args.model_id)
    # -------- Phase 2: multi (optional) --------
    if args.do_multi:
        multi = read_jsonl(args.multi_jsonl)
        train_m, test_m = stratified_split(multi, key_fn=key_multi, test_ratio=args.test_ratio, seed=args.seed)
        write_jsonl(os.path.join(split_dir, "train_multi.jsonl"), train_m)
        write_jsonl(os.path.join(split_dir, "test_multi.jsonl"), test_m)
        print(f"[SPLIT multi] train={len(train_m)} test={len(test_m)}")

        baseline_multi = eval_multi(test_m, base, tok, tools_set, tools_by_service)
        with open(os.path.join(report_dir, "baseline_multi_summary.json"), "w", encoding="utf-8") as f:
            json.dump(baseline_multi["summary"], f, indent=2)
        write_jsonl(os.path.join(report_dir, "baseline_multi_details.jsonl"), baseline_multi["details"])
        print("[BASELINE multi]", json.dumps(baseline_multi["summary"], indent=2))

        if args.do_train:
            adapter_multi = os.path.join(args.out_dir, "adapter_multi")
            print("[TRAIN] multi-step...")
            finetune(train_m, args.model_id, tools_by_service, adapter_multi, epochs=max(1, args.epochs), seed=args.seed, multistep=True)

            tuned2, tok3 = load_with_adapter(args.model_id, adapter_multi)
            post_multi = eval_multi(test_m, tuned2, tok3, tools_set, tools_by_service)
            with open(os.path.join(report_dir, "post_multi_summary.json"), "w", encoding="utf-8") as f:
                json.dump(post_multi["summary"], f, indent=2)
            write_jsonl(os.path.join(report_dir, "post_multi_details.jsonl"), post_multi["details"])
            print("[POST multi]", json.dumps(post_multi["summary"], indent=2))

    print("[DONE] Outputs in:", args.out_dir)

if __name__ == "__main__":
    main()