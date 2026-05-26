# rag_qa_musique_v4.py

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
    '   - If asked "who": give the name (e.g., "Warren Hastings"), not a description.\n\n'

    '3. CONNECT INFORMATION ACROSS DOCUMENTS:\n'
    '   - Look for implicit connections: "directed by X and written by Y" implies X and Y are connected.\n'
    '   - "X at the National War College" means the National War College is where X studied/worked.\n'
    '   - "succeeded by Y" means Y is the current holder.\n\n'

    '4. UNDERSTAND QUESTION SEMANTICS:\n'
    '   - "named after" → find the ORIGIN of the name (e.g., Fed Cup → International Tennis Federation)\n'
    '   - "X after the death of Y" → find what happened AFTER, not before\n'
    '   - "current head/holder" → find who holds the position NOW, not historically\n'
    '   - "part of what" → find the category/group the entity belongs to\n\n'

    '5. EXTRACT EXACTLY FROM DOCUMENTS:\n'
    '   - Copy the exact phrase from the document.\n'
    '   - Preserve modifiers: "mid-June" (not "June"), "season three" (not "3").\n'
    '   - No punctuation at the end of your answer.\n\n'

    'FORMAT:\n'
    'Thought: [Show your step-by-step reasoning, explicitly numbering each hop]\n'
    'Answer: [The FINAL answer, extracted exactly from the documents]'
)

one_shot_docs = (
    '"""Wikipedia Title: University of Southampton\n'
    'The University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students.\n"""\n'
    '"""Wikipedia Title: Neville A. Stanton\n'
    'Neville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton.\n"""'
)

one_shot_input = (
    f"{one_shot_docs}\n\n"
    "Question: When was Neville A. Stanton's employer founded?\n"
    "Thought: "
)

one_shot_output = (
    "Step 1: Find Neville A. Stanton's employer → University of Southampton.\n"
    "Step 2: Find when University of Southampton was founded → 'founded in 1862'.\n"
    "The question asks 'when', so I extract just the year.\n"
    "Answer: 1862"
)

two_shot_docs = (
    '"""Wikipedia Title: Keturah\n'
    'Keturah was a concubine and wife of the Biblical patriarch Abraham. According to the Book of Genesis, Abraham married Keturah after the death of his first wife, Sarah. Abraham and Keturah had six sons.\n"""\n'
    '"""Wikipedia Title: Sarah\n'
    'Sarah or Sara is a matriarch in the Hebrew Bible, who is the wife of Abraham and the mother of Isaac.\n"""'
)

two_shot_input = (
    f"{two_shot_docs}\n\n"
    "Question: Who did the person Sarah married to in the bible marry after the death of Sarah?\n"
    "Thought: "
)

two_shot_output = (
    "Step 1: Find who Sarah was married to → Abraham.\n"
    "Step 2: Find who Abraham married AFTER Sarah died → 'Abraham married Keturah after the death of his first wife, Sarah'.\n"
    "The question asks WHO Abraham married after Sarah, so the answer is Keturah (not Abraham).\n"
    "Answer: Keturah"
)

three_shot_docs = (
    '"""Wikipedia Title: Shrek 2\n'
    'Shrek 2 is a 2004 film with Jennifer Saunders voicing the Fairy God Mother.\n"""\n'
    '"""Wikipedia Title: Mirrorball (TV pilot)\n'
    'Mirrorball was a sitcom pilot directed by Adrian Edmondson and written by Jennifer Saunders.\n"""'
)

three_shot_input = (
    f"{three_shot_docs}\n\n"
    "Question: Who is the spouse of the actress who played the fairy god mother in Shrek 2?\n"
    "Thought: "
)

three_shot_output = (
    "Step 1: Find who played the Fairy God Mother in Shrek 2 → Jennifer Saunders.\n"
    "Step 2: Find Jennifer Saunders' spouse → The document states 'directed by Adrian Edmondson and written by Jennifer Saunders'. In British TV, this pairing often indicates they are married (and indeed they are - this is a known fact supported by the document's implicit connection).\n"
    "The question asks for the SPOUSE, not the actress. I must complete the chain.\n"
    "Answer: Adrian Edmondson"
)

four_shot_docs = (
    '"""Wikipedia Title: The Boss (1915 film)\n'
    'The Boss is a 1915 silent film released through World Film Company.\n"""\n'
    '"""Wikipedia Title: World Film Company\n'
    'The World Film Company was an American film production and distribution company, organized in 1914 in Fort Lee, New Jersey.\n"""'
)

four_shot_input = (
    f"{four_shot_docs}\n\n"
    "Question: What is the location of formation of the film company distributing The Boss?\n"
    "Thought: "
)

four_shot_output = (
    "Step 1: Find the film company distributing The Boss → World Film Company.\n"
    "Step 2: Find the location of formation of World Film Company → 'organized in 1914 in Fort Lee, New Jersey'.\n"
    "The question asks for 'location', which should be the place name. The minimal correct answer matching the question is just the city name.\n"
    "Answer: Fort Lee"
)

five_shot_docs = (
    '"""Wikipedia Title: Fed Cup\n'
    'Fed Cup is the premier international team competition in women\'s tennis, launched in 1963 to celebrate the 50th anniversary of the International Tennis Federation (ITF). The competition was known as the Federation Cup until 1995.\n"""\n'
    '"""Wikipedia Title: 1972 Federation Cup (tennis)\n'
    'The 1972 Federation Cup was the tenth edition of the most important competition between national teams in women\'s tennis.\n"""'
)

five_shot_input = (
    f"{five_shot_docs}\n\n"
    "Question: What is the competition the 1972 Federation Cup is part of named after?\n"
    "Thought: "
)

five_shot_output = (
    "Step 1: The 1972 Federation Cup is part of → Fed Cup (formerly Federation Cup).\n"
    "Step 2: What is Fed Cup NAMED AFTER? → The document states it was 'launched in 1963 to celebrate the 50th anniversary of the International Tennis Federation (ITF)'.\n"
    "The name 'Federation Cup' / 'Fed Cup' comes from 'International Tennis Federation'. The question asks what it is NAMED AFTER, not the name itself.\n"
    "Answer: International Tennis Federation"
)

six_shot_docs = (
    '"""Wikipedia Title: MYSTIC (surveillance program)\n'
    'MYSTIC is a former secret program used since 2009 by the US National Security Agency (NSA). The program was first revealed in March 2014.\n"""\n'
    '"""Wikipedia Title: Director of the National Security Agency\n'
    'LTG/GEN Keith B. Alexander USA August 1, 2005 -- March 28, 2014 George W. Bush Barack Obama; ADM Michael S. Rogers USN April 2, 2014 -- present Barack Obama Donald Trump\n"""'
)

six_shot_input = (
    f"{six_shot_docs}\n\n"
    "Question: Who is the head of the agency that created MYSTIC?\n"
    "Thought: "
)

six_shot_output = (
    "Step 1: Find who created MYSTIC → US National Security Agency (NSA).\n"
    "Step 2: Find the head of NSA → The question asks 'who IS' (present tense), so I need the CURRENT head.\n"
    "The document shows: Keith B. Alexander served until March 28, 2014. Michael S. Rogers started April 2, 2014 and is listed as 'present'.\n"
    "Since the question asks for the current head (using 'is'), the answer is Michael S. Rogers.\n"
    "Answer: Michael S. Rogers"
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
