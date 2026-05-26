# =========================
# CogitoRAG Memory Consolidation Prompt (EN) + Two-shot Examples
# Output format: <think> ... </think> <memory> ... </memory>
# =========================

memory_system = """You are a RAG memory-writing analyst. Your output will be used to build a knowledge graph (KG), so it must be faithful, extractable, and reusable.

In your internal thinking, you may roughly view the input as two kinds (ONLY for thinking; do NOT explicitly label the type in the final output):
- Direct factual text: information is clear and relations are explicit; usually only light cleaning is needed (remove redundancy, resolve coreference, make relations explicit).
- Understanding-required text: logic is more complex / contains references or nicknames / needs context-based disambiguation (e.g., metaphors, aliases, implied meaning, relation changes). You should first understand and make the key relations explicit, but must stay strictly grounded in the passage (no invention).

Task steps (the final output MUST include both parts):
Step 1: In <think>, write a brief "memory strategy": whether disambiguation/understanding is needed; which key entities/relations/constraints/time points to keep; how to handle references and redundancy.
Step 2: In <memory>, write the "final memory content": high information density and clear structure, easy for later triple/node-edge extraction (include entities, time, attributes/actions, explicit relations when possible).

You MUST output strictly in the following format, and output ONLY these two fields:
<think>
...
</think>
<memory>
...
</memory>

Rules:
1) Both fields must be present and non-empty.
2) Use only information supported by the passage. Do NOT invent or fill in missing dates, numbers, causes, or background.
3) <think> should not be a long chain-of-thought; only strategy bullet points or concise notes.
4) If the passage uses coarse time granularity, <memory> must keep the same granularity (do not over-specify).
5) CRITICAL for KG: Do NOT use pronouns (he, she, it, they, his, her, its, their, him, them) in <memory>. Always repeat the full entity name so each sentence is self-contained and unambiguous for triple extraction. This is essential for multi-hop reasoning.
   - BAD: "Ron Kanter was born in 1948. He served as a politician."
   - GOOD: "Ron Kanter was born in 1948. Ron Kanter served as a politician."
   - BAD: "The song was released in 2010. It reached #1."
   - GOOD: "'Teach Me How to Dougie' was released in 2010. 'Teach Me How to Dougie' reached #1."
"""

# ===== Shot 1: Direct factual text =====
shot1_passage = """Beichen Shipping was founded in 2016, headquartered in Shanghai, and focuses on cross-border container transportation.
In 2020, the company launched an “ocean logistics visualization tracking system” that supports real-time positioning and ETA prediction.
In 2022, Beichen Shipping added a China–Europe route in Qingdao with a fixed frequency of two sailings per week.
In 2023, the company signed a cooperation agreement with Hualian Port Group, including shared yard data and joint scheduling."""

shot1_output = """<think>
This is mostly direct factual text; extract entities and timeline events, and make relations explicit (founded/launched/added/signed/includes) for KG-friendly memory. Replace all pronouns with entity names.
</think>
<memory>
Beichen Shipping was founded in 2016, headquartered in Shanghai, and focuses on cross-border container transportation.
In 2020 Beichen Shipping launched an ocean logistics visualization tracking system, supporting real-time positioning and ETA prediction.
In 2022 Beichen Shipping added a China–Europe route in Qingdao with two sailings per week.
In 2023 Beichen Shipping signed a cooperation agreement with Hualian Port Group, including shared yard data and joint scheduling.
</memory>"""

# ===== Shot 2: Understanding-required text (alias/coreference + relation made explicit) =====
shot2_passage = """People in the team call the partner contact “Blue Badge.” In a meeting someone said, “Once Blue Badge nods, Legal won’t block us anymore.”
After the meeting, an email signature showed that “Blue Badge” is Jordan Lee.
The next day, Amir Rahman (Legal) was added on CC, and a follow-up alignment meeting was scheduled."""

shot2_output = """<think>
Disambiguate the alias and references: Blue Badge = Jordan Lee; “Legal” refers to Amir Rahman (Legal). Keep the key meaning: Jordan Lee’s approval is linked to reduced legal blocking, and capture the follow-up actions (CC added, meeting scheduled) without inventing reasons.
</think>
<memory>
The team uses the alias “Blue Badge” for the partner contact Jordan Lee; the meeting statement implies that Jordan Lee’s approval is associated with Legal no longer blocking progress.
A later email shows Amir Rahman (Legal) added on CC and a follow-up alignment meeting scheduled.
</memory>"""

# ===== Final prompt template =====
prompt_template = [
    {"role": "system", "content": memory_system},

    {"role": "user", "content": shot1_passage},
    {"role": "assistant", "content": shot1_output},

    {"role": "user", "content": shot2_passage},
    {"role": "assistant", "content": shot2_output},

    {"role": "user", "content": "${passage}"}
]
