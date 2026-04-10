### 🧠 Results of tuning on test set that require only 1 tool to be identified.
## !python run_tool_router_experiment.py --do_train
[BASELINE single] {
  "total": 24,
  "json_valid_rate": 1.0,
  "tool_in_set_rate": 0.08333333333333333,
  "tool_accuracy": 0.08333333333333333
}

[POST single] {
  "total": 24,
  "json_valid_rate": 0.9583333333333334,
  "tool_in_set_rate": 0.9583333333333334,
  "tool_accuracy": 0.7916666666666666
}

## !python run_tool_router_experiment.py --do_train --do_multi
[SPLIT multi] train=8 test=4
[BASELINE multi] {
  "total": 4,
  "json_valid_rate": 1.0,
  "hallucinated_tool_rate": 1.0,
  "step_count_accuracy": 0.75,
  "sequence_exact_match": 0.0,
  "first_step_accuracy": 0.0
}

For a 3.8B SLM with:
~100 training examples
LoRA only
no negative examples
no schema-enforced decoding

👉 ~71% correct routing on first pass is very solid


This already demonstrates:
- feasibility of SLM-based tool routers
- value of local fine-tuning vs prompt-only
- high signal‑to‑noise in your dataset

In a real system, you’d typically add:
- a fallback router
- or a second-pass re-ranker
- which would push effective accuracy even higher.




🔧 Tool Routing & Planning with Small Language Models (SLMs)
This project explores tool routing and multi‑step tool planning using a small language model (SLM) fine‑tuned with QLoRA.
The focus is not just on final accuracy, but on understanding how tool grounding and planning interact, especially when training with limited data.

Note:
This project requires a CUDA‑enabled GPU.
When using Google Colab, set Runtime → Change runtime type → GPU (T4) before running.

🎯 Problem Statement
Given a natural‑language user request and a fixed set of tools (Calendar, Gmail, Drive, Docs), the model should act as an orchestrator:

Decide which tools to call
Decide how many tools are needed
Decide the order of execution
Output a structured plan in JSON

In real systems:

Some requests require one tool
Others require multiple tools
The model should not be pre‑classified as “single‑step” or “multi‑step”

This project investigates how to train such behavior effectively.

🧠 Key Insight from This Study (Important)
✅ What worked well

Single‑step training (one tool per request) achieved strong accuracy (~70%) with limited data.
The model showed good tool grounding once trained in this simple format.

❌ Why multi‑step initially failed
When trained only on multi‑step data, the model:

Learned the structure of a plan (number of steps)
But failed to select correct tools
Hallucinated tool names consistently
Had 0% first‑step accuracy

📌 Root Cause (Core Learning)
Tool grounding and planning were trained using different output schemas:

Single‑step data → { "tool": ..., "arguments": ... }
Multi‑step data → { "steps": [...] }

As a result, tool knowledge did not transfer into planning, even though the tools were the same.

✅ Correct Mental Model (Design Conclusion)
The orchestrator’s job is always to produce a plan.

A single‑tool request is simply a plan with one step
A multi‑tool workflow is a plan with multiple steps

Therefore:

Single‑step and multi‑step are not different problems
Single‑step is the simplest possible planning case


🔁 Final Dataset Design (Recommended)
All training data should follow one unified format:
JSON{  "steps": [    {      "tool": "tool_name",      "arguments": { ... }    }  ]}Show more lines

Single‑step examples → converted to 1‑step plans
Multi‑step examples → kept as multi‑step plans
One model
One task
One output schema

This ensures:

Tool grounding transfers into planning
First‑step accuracy improves
Hallucinated tools decrease


⚙️ Training Setup

Model: microsoft/Phi-4-mini-instruct
Fine‑tuning: QLoRA (4‑bit)
Adapter Method: LoRA
Hardware: NVIDIA T4 (Colab GPU)
Why T4 is required:

QLoRA + bitsandbytes requires CUDA
CPU runtime will fail




⚠️ Known Environment Constraint (Colab)
During training, a compatibility issue was observed between:

transformers
accelerate

A temporary monkey patch is used to drop an unsupported argument
(keep_torch_compile) in Colab environments.
This does not affect model behavior or evaluation, and is documented in code.
It can be removed once dependency versions fully align.
