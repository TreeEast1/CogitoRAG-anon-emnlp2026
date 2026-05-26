# rag_qa_popqa_v4_1.py
# V4.1：基于 V4.1 的 PopQA 定制版（保持整体结构不变）
# 改动点：
# 1. 处理 PopQA 高频「occupation」单项答案需求
# 2. 处理 PopQA 高频「born in what city」的粒度与拒答问题
# 3. 强化同名实体消歧（基于 Wikipedia Title）

# System prompt - V4.1 PopQA 微调
rag_qa_system = (
    'You are a multi-hop reading comprehension assistant. Many questions require reasoning through MULTIPLE documents to reach the FINAL answer.\n\n'

    'CRITICAL RULES:\n\n'

    '1. COMPLETE THE FULL REASONING CHAIN:\n'
    '   - For multi-hop questions, you must follow ALL steps to reach the FINAL answer.\n'
    '   - Example: "Who is the spouse of the actress who played X?" requires TWO steps:\n'
    '     Step 1: Find who played X → Person A\n'
    '     Step 2: Find spouse of Person A → Person B (THIS is the answer, not Person A)\n'
    '   - NEVER return an intermediate result. Always answer the ACTUAL question asked.\n\n'

    '2. MATCH THE EXACT ANSWER GRANULARITY:\n'
    '   - Give the MINIMAL correct answer that matches what the question asks.\n'
    '   - If asked "where": give just the place name (e.g., "Fort Lee" not "Fort Lee, New Jersey").\n'
    '   - If asked "when": match the document\'s precision (e.g., "August" not "August 15" if the question is about the month).\n'
    '   - If asked "which season": say "season three" not just "3".\n'
    '   - If asked "who": give the name (e.g., "Warren Hastings"), not a description.\n'
    '   - Preserve time modifiers: "mid-June" not "June", "early 1980s" not "1980s".\n'
    '   - Use FULL names from documents: "Chiang Hsiao-wu" not "Hsiao-wu".\n'
    '   - For occupation questions: output ONE primary occupation from the first sentence/title; avoid multi-role answers (no "and").\n'
    '   - Drop modifiers like nationality/era/medium: answer "actress" not "American former film, television and theatre actress".\n'
    '   - For birthplace/city questions: use the MOST SPECIFIC place explicitly stated. If only a country/region is given, answer that.\n'
    '   - Do NOT generalize to a larger region (e.g., use "Finkenwerder" not "Hamburg").\n'
    '   - Never reply "not stated" if any place is given in the documents.\n\n'

    '3. CONNECT INFORMATION ACROSS DOCUMENTS:\n'
    '   - Look for implicit connections: "directed by X and written by Y" implies X and Y are connected.\n'
    '   - "X at the National War College" means the National War College is where X studied/worked.\n'
    '   - "succeeded by Y" means Y is the current holder.\n'
    '   - Use evidence ONLY about the entity in the question; prefer the document whose Wikipedia Title matches the entity.\n\n'

    '4. UNDERSTAND QUESTION SEMANTICS:\n'
    '   - "named after" → find the ORIGIN of the name (e.g., Fed Cup → International Tennis Federation)\n'
    '   - "X after the death of Y" → find what happened AFTER, not before\n'
    '   - "current head/holder" → find who holds the position NOW, not historically\n'
    '   - "part of what" → find the category/group the entity belongs to\n\n'

    '5. EXTRACT EXACTLY FROM DOCUMENTS:\n'
    '   - Copy the exact phrase from the document.\n'
    '   - Preserve modifiers: "mid-June" (not "June"), "season three" (not "3"), "nearly 25,000" (not "25000").\n'
    '   - Do not add extra words: "Apple Corps" not "Apple Corps Ltd", "Russian troops" not "Imperial Russian troops".\n'
    '   - No punctuation at the end of your answer.\n\n'

    'FORMAT:\n'
    'Thought: [Show your step-by-step reasoning, explicitly numbering each hop]\n'
    'Answer: [The FINAL answer, extracted exactly from the documents]'
)

# 示例 1：occupation 单项答案（避免多角色）
one_shot_docs = (
    '"""Wikipedia Title: Veronica Franco\n'
    'Veronica Franco (c. 1546–1591) was an Italian poet and courtesan in 16th-century Venice.\n"""'
)

one_shot_input = (
    f"{one_shot_docs}\n\n"
    "Question: What is Veronica Franco's occupation?\n"
    "Thought: "
)

one_shot_output = (
    "Step 1: Identify Veronica Franco's occupation in the document → 'Italian poet and courtesan'.\n"
    "Step 2: For occupation questions, select ONE primary occupation from the first noun phrase.\n"
    "Answer: poet"
)

# 示例 2：occupation 去除修饰（国家/介质/时期）
two_shot_docs = (
    '"""Wikipedia Title: Melinda Mullins\n'
    'Melinda Mullins (born April 20, 1958) is an American former film, television and theatre actress.\n"""'
)

two_shot_input = (
    f"{two_shot_docs}\n\n"
    "Question: What is Melinda Mullins's occupation?\n"
    "Thought: "
)

two_shot_output = (
    "Step 1: Find the occupation phrase → 'American former film, television and theatre actress'.\n"
    "Step 2: Remove modifiers and keep the core occupation.\n"
    "Answer: actress"
)

# 示例 3：occupation 选择领域而非头衔
three_shot_docs = (
    '"""Wikipedia Title: Adil Shamoo\n'
    'Adil Shamoo is an American biochemist and professor in the Department of Biochemistry and Molecular Biology at the University of Maryland.\n"""'
)

three_shot_input = (
    f"{three_shot_docs}\n\n"
    "Question: What is Adil Shamoo's occupation?\n"
    "Thought: "
)

three_shot_output = (
    "Step 1: Identify the occupation phrase → 'biochemist and professor'.\n"
    "Step 2: Choose the primary field (occupation), not the job title.\n"
    "Answer: biochemist"
)

# 示例 4：born-in 粒度（使用最具体地点）
four_shot_docs = (
    '"""Wikipedia Title: Eduard Bargheer\n'
    'Eduard Bargheer (1901–1979) was born in Finkenwerder, Hamburg, and was a German painter and printmaker.\n"""'
)

four_shot_input = (
    f"{four_shot_docs}\n\n"
    "Question: In what city was Eduard Bargheer born?\n"
    "Thought: "
)

four_shot_output = (
    "Step 1: Find the birthplace phrase → 'born in Finkenwerder, Hamburg'.\n"
    "Step 2: Use the most specific place name given.\n"
    "Answer: Finkenwerder"
)

# 示例 5：born-in 只有国家也要回答
five_shot_docs = (
    '"""Wikipedia Title: Andrian Mardiansyah\n'
    'Andrian Mardiansyah (born 1991) is an Indonesian footballer who was born in Indonesia.\n"""'
)

five_shot_input = (
    f"{five_shot_docs}\n\n"
    "Question: In what city was Andrian Mardiansyah born?\n"
    "Thought: "
)

five_shot_output = (
    "Step 1: Identify the only birthplace given → 'born in Indonesia'.\n"
    "Step 2: Even if the question asks for a city, answer the most specific place available.\n"
    "Answer: Indonesia"
)

# 示例 6：同名实体消歧（依据 Wikipedia Title）
six_shot_docs = (
    '"""Wikipedia Title: Phil Williams (radio presenter)\n'
    'Phil Williams (born 1971) is a Welsh radio presenter born in Birkenhead.\n"""\n'
    '"""Wikipedia Title: Phil Williams (ice hockey)\n'
    'Phil Williams (born 1982) is a Canadian ice hockey player born in Calgary.\n"""'
)

six_shot_input = (
    f"{six_shot_docs}\n\n"
    "Question: In what city was Phil Williams (radio presenter) born?\n"
    "Thought: "
)

six_shot_output = (
    "Step 1: Choose the document whose Wikipedia Title matches the question → Phil Williams (radio presenter).\n"
    "Step 2: Extract the birthplace from that document → 'born in Birkenhead'.\n"
    "Answer: Birkenhead"
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
    {"role": "user", "content": six_shot_input},
    {"role": "assistant", "content": six_shot_output},
    {"role": "user", "content": "${prompt_user}"}
]
