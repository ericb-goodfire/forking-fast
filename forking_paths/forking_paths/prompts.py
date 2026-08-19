"""Prompt construction for MMLU, faithful to the authors' reference repo.

Base (completion) model  -> raw zero-shot CoT prompt (repo `prompt_mmlu`).
Instruct model           -> same question content wrapped in the Llama-3 chat
                            template with a generation prompt.

Each model is analyzed *as it is actually used* (the plan's chosen framing);
the confound between instruction-tuning and prompt format is documented in the
report.
"""
from __future__ import annotations

# Reference repo `prompt_mmlu` (forking_paths/prompts.py), verbatim structure.
PROMPT_MMLU_BASE = """\
Question:
{question}

Choices:
A) {A}
B) {B}
C) {C}
D) {D}

Answer:
Let's think step by step."""

# For the instruct model we deliver the same content as a user turn. We keep the
# explicit "Let's think step by step" CoT cue so both models run the same task.
INSTRUCT_USER_TEMPLATE = """\
{question}

A) {A}
B) {B}
C) {C}
D) {D}

Think step by step, then end with "The answer is (X)" where X is A, B, C, or D."""

# Answer-elicitation suffix used by the logit-read fallback (repo `therefore_ABCD`).
THEREFORE_ABCD = "\nTherefore, among A through D, the answer is ("


def format_mmlu_base(question: str, choices) -> str:
    return PROMPT_MMLU_BASE.format(
        question=question, A=choices[0], B=choices[1], C=choices[2], D=choices[3]
    )


def format_mmlu_instruct_user(question: str, choices) -> str:
    return INSTRUCT_USER_TEMPLATE.format(
        question=question, A=choices[0], B=choices[1], C=choices[2], D=choices[3]
    )


# Llama-3 chat template, built manually from special-token strings so we do not
# depend on tokenizer.apply_chat_template (which requires jinja2, not present in
# the cluster job image). The Llama-3 tokenizer maps these literal special-token
# strings to their reserved ids, so tokenizing with add_special_tokens=False
# yields exactly the chat-formatted prompt the instruct model expects.
LLAMA3_CHAT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def build_prompt_ids(tokenizer, question, choices, is_instruct: bool):
    """Return the formatted prompt token-ids for the given model mode."""
    if is_instruct:
        user = format_mmlu_instruct_user(question, choices)
        text = LLAMA3_CHAT.format(content=user)
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return list(ids)
    text = format_mmlu_base(question, choices)
    # base model: no special chat template; add BOS.
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return list(ids)


# ---------------------------------------------------------------------------
# DeepSeek-R1-Distill-Llama-8B (reasoning-native) prompt.
#
# R1-Distill's chat template (tokenizer_config.json) wraps the user turn and,
# with add_generation_prompt=True, appends "<｜Assistant｜><think>\n" so the
# assistant response BEGINS inside a thinking block. DeepSeek's usage guidance:
# no system prompt, put all instructions in the user turn. We build the template
# from literal special-token strings (same approach as LLAMA3_CHAT) so we do not
# depend on tokenizer.apply_chat_template / jinja2 in the job image.
#
# The exact special-token surface strings are VERIFIED at runtime against the
# tokenizer (see run_track / verify_think) before use.
R1_USER_TEMPLATE = INSTRUCT_USER_TEMPLATE  # identical MMLU user content

# R1 special tokens (surface strings). BOS + user turn + assistant + <think>.
R1_BOS = "<｜begin▁of▁sentence｜>"
R1_USER = "<｜User｜>"
R1_ASSISTANT = "<｜Assistant｜>"
R1_THINK_OPEN = "<think>"
R1_THINK_CLOSE = "</think>"


def build_prompt_ids_r1(tokenizer, question, choices):
    """Return R1-Distill prompt token-ids: BOS + user turn + assistant + <think>\\n.

    Mirrors R1-Distill's chat template with add_generation_prompt=True, built
    from literal special-token strings so generation begins inside <think>.
    """
    user = R1_USER_TEMPLATE.format(
        question=question, A=choices[0], B=choices[1], C=choices[2], D=choices[3]
    )
    text = f"{R1_BOS}{R1_USER}{user}{R1_ASSISTANT}{R1_THINK_OPEN}\n"
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(ids)


# Forcing string for the R1 reasoning-native forced-answer readout. Placed AFTER
# the </think> close so the model reads its committed answer post-deliberation.
FORCE_MC_R1 = "\n\nThe final answer (out of options A/B/C/D) is: "
