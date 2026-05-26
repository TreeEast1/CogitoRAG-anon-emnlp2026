# rag_qa_musique_v2.py
# 改进版：强调抽取式答案，保留原文表述（如 mid-June 不要简化为 June）
# 原版: rag_qa_musique.py

one_shot_rag_qa_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)

# 第二个示例：演示保留时间修饰语（early/mid/late）
second_shot_docs = (
    """Wikipedia Title: Allied Negotiations\nThe preliminary talks between the Allied powers began in early March 1945. By mid-April, the main discussions had formally started, focusing on post-war territorial arrangements.\n"""
    """Wikipedia Title: World War II Timeline\nWorld War II in Europe ended on May 8, 1945, known as Victory in Europe Day (V-E Day).\n"""
)


# 改进的 system prompt：强调抽取式答案
rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and answer questions accurately. '
    'Your response starts after "Thought: ", where you methodically break down the reasoning process. '
    'Conclude with "Answer: " followed by the answer extracted EXACTLY as it appears in the source documents. '
    'IMPORTANT: Do NOT simplify, paraphrase, or normalize the answer. '
    'Preserve the EXACT original phrasing from the documents, including: '
    '- Time modifiers: "mid-June" (not "June"), "early March" (not "March"), "late 1990s" (not "1990s") '
    '- Articles and prefixes: "the USSR" (not "USSR"), "the United States" (not "United States") '
    '- Full names and titles as written in the source '
    'Your answer should be a direct extraction, not a summary or interpretation.'
)

# 第一个示例
one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. "
    "The document states: 'The University of Southampton, which was founded in 1862'. "
    "I will extract the answer exactly as stated in the document."
    "\nAnswer: 1862"
)

# 第二个示例：演示保留时间修饰语
second_shot_input = (
    f"{second_shot_docs}"
    "\n\nQuestion: "
    "When did the main discussions formally start?"
    '\nThought: '
)

second_shot_output = (
    "The document states: 'By mid-April, the main discussions had formally started'. "
    "The answer should preserve the exact phrasing 'mid-April', not simplify it to just 'April'."
    "\nAnswer: mid-April"
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": second_shot_input},
    {"role": "assistant", "content": second_shot_output},
    {"role": "user", "content": "${prompt_user}"}
]
