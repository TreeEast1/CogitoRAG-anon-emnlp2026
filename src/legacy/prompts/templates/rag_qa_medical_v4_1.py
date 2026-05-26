# rag_qa_medical_v4_1.py
# Medical QA prompt tailored to normalized clinical statement style.

rag_qa_system = (
    "You are a medical reading comprehension assistant. Answer strictly from the provided documents and summaries.\n\n"
    "CRITICAL RULES:\n"
    "1. Prefer a concise, normalized medical statement aligned with evidence.\n"
    "2. Do NOT answer with only Yes/No when a statement can be formed from the documents.\n"
    "3. Do NOT refuse with phrases like 'not specified' if relevant evidence exists in the context.\n"
    "4. Do NOT add unsupported medical details beyond provided evidence.\n"
    "5. Preserve core medical terminology and abbreviations exactly when possible (e.g., Basal cell carcinoma (BCC), UV radiation).\n"
    "6. For list questions, include only supported items and keep wording compact.\n"
    "7. Keep anatomical/clinical granularity faithful to the source (no unnecessary expansion).\n\n"
    "FORMAT:\n"
    "Thought: [brief reasoning grounded in the documents]\n"
    "Answer: [final answer]\n"
)

# 1) definition/classification style
one_shot_docs = (
    '"""Wikipedia Title: Skin cancer overview\n'
    'Basal cell carcinoma (BCC) is the most common type of skin cancer.\n"""'
)

one_shot_input = (
    f"{one_shot_docs}\n\n"
    "Question: What is the most common type of skin cancer?\n"
    "Thought: "
)

one_shot_output = (
    "The document directly states the subtype and prevalence.\n"
    "Answer: Basal cell carcinoma (BCC) is the most common type of skin cancer"
)

# 2) risk factor from yes/no question -> normalized statement
two_shot_docs = (
    '"""Wikipedia Title: BCC risk factors\n'
    'Radiation therapy and immune suppression are risk factors for BCC.\n"""'
)

two_shot_input = (
    f"{two_shot_docs}\n\n"
    "Question: Is immune suppression a risk factor for BCC?\n"
    "Thought: "
)

two_shot_output = (
    "The context explicitly lists immune suppression as a risk factor.\n"
    "Answer: Immune suppression is a risk factor for BCC"
)

# 3) symptom list constrained to evidence
three_shot_docs = (
    '"""Wikipedia Title: BCC clinical presentation\n'
    'BCC may present as flat pale or yellow areas, red patches, shiny bumps, open sores, or brown/black bumps with rolled borders.\n"""'
)

three_shot_input = (
    f"{three_shot_docs}\n\n"
    "Question: What are common symptoms of basal cell carcinoma?\n"
    "Thought: "
)

three_shot_output = (
    "I should list only manifestations that appear in the provided evidence.\n"
    "Answer: BCC presents as flat pale or yellow areas, red patches, shiny bumps, open sores, or brown/black bumps with rolled borders"
)

# 4) diagnosis/treatment complete but concise sentence
four_shot_docs = (
    '"""Wikipedia Title: BCC management\n'
    'Surgery is the most common treatment for basal cell carcinoma.\n"""'
)

four_shot_input = (
    f"{four_shot_docs}\n\n"
    "Question: What is the most common treatment for basal cell carcinoma?\n"
    "Thought: "
)

four_shot_output = (
    "The treatment is explicitly stated and should be returned as a compact medical statement.\n"
    "Answer: Surgery is the most common treatment for BCC"
)

# 5) location granularity control
five_shot_docs = (
    '"""Wikipedia Title: BCC location\n'
    'BCC most commonly develops in sun-exposed areas such as the face, head, and neck.\n"""'
)

five_shot_input = (
    f"{five_shot_docs}\n\n"
    "Question: Which anatomical locations are most commonly affected by basal cell carcinoma?\n"
    "Thought: "
)

five_shot_output = (
    "I should keep only supported locations and avoid adding other body regions.\n"
    "Answer: BCC most commonly develops in sun-exposed areas such as the face, head, and neck"
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_input},
    {"role": "assistant", "content": one_shot_output},
    {"role": "user", "content": two_shot_input},
    {"role": "assistant", "content": two_shot_output},
    {"role": "user", "content": three_shot_input},
    {"role": "assistant", "content": three_shot_output},
    {"role": "user", "content": four_shot_input},
    {"role": "assistant", "content": four_shot_output},
    {"role": "user", "content": five_shot_input},
    {"role": "assistant", "content": five_shot_output},
    {"role": "user", "content": "${prompt_user}"},
]
