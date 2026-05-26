# rag_qa_novel_v4_1.py
# Novel QA prompt tailored to short entity / location / name answers.

rag_qa_system = (
    "You are a literary reading comprehension assistant. Answer strictly from the provided documents and summaries.\n\n"
    "CRITICAL RULES:\n"
    "1. If the question asks for a person, place, object, region, or feature name, prefer a short entity phrase.\n"
    "2. Do NOT add a trailing period for entity-style answers.\n"
    "3. Do NOT expand short answers into full descriptive sentences unless the question clearly requires a sentence.\n"
    "4. Match the semantic target exactly: who -> person name, where/which region -> place name, which feature/landmark -> target entity.\n"
    "5. For multi-entity questions, output a clean concise list of the supported entities only.\n"
    "6. Avoid decorative narrative wording, scene description, or unsupported elaboration.\n"
    "7. If the evidence itself is a sentence but the question asks for an entity, compress to the target entity.\n\n"
    "FORMAT:\n"
    "Thought: [brief reasoning grounded in the documents]\n"
    "Answer: [final answer]\n"
)

# 1) alias / mapping short answer
one_shot_docs = (
    '"""Wikipedia Title: An Unsentimental Journey through Cornwall\n'
    'The plant known scientifically as Erica vagans is referred to as Cornish heath.\n"""'
)

one_shot_input = (
    f"{one_shot_docs}\n\n"
    "Question: In the narrative of 'An Unsentimental Journey through Cornwall', which plant known scientifically as Erica vagans is also referred to by another common name, and what is that name?\n"
    "Thought: "
)

one_shot_output = (
    "The question asks for the common name, so the answer should be the short mapped entity only.\n"
    "Answer: Cornish heath"
)

# 2) relationship / who question -> person name only
two_shot_docs = (
    '"""Wikipedia Title: Royal visit account\n'
    'Baron Von Pawel-Rammingen married Princess Frederica of Hanover.\n"""'
)

two_shot_input = (
    f"{two_shot_docs}\n\n"
    "Question: Within the account of the royal visit to St. Michael's Mount in Cornwall, who is identified as the person who married Princess Frederica of Hanover?\n"
    "Thought: "
)

two_shot_output = (
    "The target is the person, not the whole relation sentence.\n"
    "Answer: Baron Von Pawel-Rammingen"
)

# 3) location / region answer without sentence punctuation
three_shot_docs = (
    '"""Wikipedia Title: Historic sites\n'
    'Mont St. Michel is located in Normandy.\n"""'
)

three_shot_input = (
    f"{three_shot_docs}\n\n"
    "Question: According to the narrative's discussion of historic sites, in which region of France is Mont St. Michel located?\n"
    "Thought: "
)

three_shot_output = (
    "The question asks for the region name only.\n"
    "Answer: Normandy"
)

# 4) multi-entity listing kept compact
four_shot_docs = (
    '"""Wikipedia Title: Kynance Cove\n'
    'Asparagus Island is located near or within Kynance Cove. Gull Rock is located near or within Kynance Cove. Bellows is located in Kynance Cove.\n"""'
)

four_shot_input = (
    f"{four_shot_docs}\n\n"
    "Question: In the account of the travelers' visit to Kynance Cove, what natural landmarks are described as being located near or within Kynance Cove?\n"
    "Thought: "
)

four_shot_output = (
    "This is a list question, so I should output only the supported landmarks in a clean short list.\n"
    "Answer: Asparagus Island, Gull Rock, and Bellows"
)

# 5) sentence vs phrase discrimination
five_shot_docs = (
    '"""Wikipedia Title: Housel Cove\n'
    'Penolver is located near Housel Cove.\n"""'
)

five_shot_input = (
    f"{five_shot_docs}\n\n"
    "Question: During the travelers' exploration of the Cornish coastline in the narrative, what notable geographical feature is located near Housel Cove?\n"
    "Thought: "
)

five_shot_output = (
    "Even though the evidence is a sentence, the question asks for the feature name, so I should return only the entity phrase.\n"
    "Answer: Penolver"
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
