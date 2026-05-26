# rag_qa_musique_v3.py

one_shot_rag_qa_docs = (
    """Wikipedia Title: The Last Horse\nThe Last Horse (Spanish:El último caballo) is a 1950 Spanish comedy film directed by Edgar Neville starring Fernando Fernán Gómez.\n"""
    """Wikipedia Title: Southampton\nThe University of Southampton, which was founded in 1862 and received its Royal Charter as a university in 1952, has over 22,000 students. The university is ranked in the top 100 research universities in the world in the Academic Ranking of World Universities 2010. In 2010, the THES - QS World University Rankings positioned the University of Southampton in the top 80 universities in the world. The university considers itself one of the top 5 research universities in the UK. The university has a global reputation for research into engineering sciences, oceanography, chemistry, cancer sciences, sound and vibration research, computer science and electronics, optoelectronics and textile conservation at the Textile Conservation Centre (which is due to close in October 2009.) It is also home to the National Oceanography Centre, Southampton (NOCS), the focus of Natural Environment Research Council-funded marine research.\n"""
    """Wikipedia Title: Stanton Township, Champaign County, Illinois\nStanton Township is a township in Champaign County, Illinois, USA. As of the 2010 census, its population was 505 and it contained 202 housing units.\n"""
    """Wikipedia Title: Neville A. Stanton\nNeville A. Stanton is a British Professor of Human Factors and Ergonomics at the University of Southampton. Prof Stanton is a Chartered Engineer (C.Eng), Chartered Psychologist (C.Psychol) and Chartered Ergonomist (C.ErgHF). He has written and edited over a forty books and over three hundered peer-reviewed journal papers on applications of the subject. Stanton is a Fellow of the British Psychological Society, a Fellow of The Institute of Ergonomics and Human Factors and a member of the Institution of Engineering and Technology. He has been published in academic journals including "Nature". He has also helped organisations design new human-machine interfaces, such as the Adaptive Cruise Control system for Jaguar Cars.\n"""
    """Wikipedia Title: Finding Nemo\nFinding Nemo Theatrical release poster Directed by Andrew Stanton Produced by Graham Walters Screenplay by Andrew Stanton Bob Peterson David Reynolds Story by Andrew Stanton Starring Albert Brooks Ellen DeGeneres Alexander Gould Willem Dafoe Music by Thomas Newman Cinematography Sharon Calahan Jeremy Lasky Edited by David Ian Salter Production company Walt Disney Pictures Pixar Animation Studios Distributed by Buena Vista Pictures Distribution Release date May 30, 2003 (2003 - 05 - 30) Running time 100 minutes Country United States Language English Budget $$94 million Box office $$940.3 million"""
)

second_shot_docs = (
    """Wikipedia Title: Allied Negotiations\nThe preliminary talks between the Allied powers began in early March 1945. By mid-April, the main discussions had formally started, focusing on post-war territorial arrangements.\n"""
    """Wikipedia Title: World War II Timeline\nWorld War II in Europe ended on May 8, 1945, known as Victory in Europe Day (V-E Day).\n"""
)

third_shot_docs = (
    """Wikipedia Title: Governor-General of India\nThe office of Governor-General of India was established on 20 October 1774. Warren Hastings served as the first Governor-General from 1774 to 1785. The final holder was Chakravarthi Rajagopalachari. The office was abolished on 26 January 1950.\n"""
    """Wikipedia Title: Warren Hastings\nWarren Hastings (6 December 1732 – 22 August 1818) was an English statesman who served as the first de facto Governor-General of India.\n"""
)


rag_qa_system = (
    'As an advanced reading comprehension assistant, your task is to analyze text passages and answer questions accurately. '
    'Your response starts after "Thought: ", where you methodically break down the reasoning process. '
    'Conclude with "Answer: " followed by the answer extracted EXACTLY as it appears in the source documents. '
    '\n\n'
    'CRITICAL RULES for your answer:\n'
    '1. EXTRACT, do not paraphrase: Copy the exact phrase from the document.\n'
    '2. MINIMAL answer: Give the shortest correct answer. Do NOT add extra words.\n'
    '   - If asked "who", answer with the NAME only (e.g., "Warren Hastings"), not a description ("the first Governor-General").\n'
    '   - If asked "where", answer with the PLACE only (e.g., "Fort Lee"), not extra context ("Fort Lee, New Jersey, United States").\n'
    '   - If asked "when", answer with the TIME only (e.g., "mid-June"), not "in mid-June" or "mid-June 1939".\n'
    '3. PRESERVE modifiers: Keep time modifiers like "mid-", "early-", "late-", "approximately", "nearly".\n'
    '   - "mid-June" stays "mid-June" (NOT "June")\n'
    '   - "nearly 25,000" stays "nearly 25,000" (NOT "25,000")\n'
    '4. NO punctuation at the end: Do not add periods or other punctuation after your answer.\n'
    '5. Answer the ACTUAL question: If asked for a person, give the person. If asked for a place, give the place.\n'
)

one_shot_rag_qa_input = (
    f"{one_shot_rag_qa_docs}"
    "\n\nQuestion: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

one_shot_rag_qa_output = (
    "The employer of Neville A. Stanton is University of Southampton. "
    "The document states: 'The University of Southampton, which was founded in 1862'. "
    "The question asks 'when', so I extract just the year."
    "\nAnswer: 1862"
)

second_shot_input = (
    f"{second_shot_docs}"
    "\n\nQuestion: "
    "When did the main discussions formally start?"
    '\nThought: '
)

second_shot_output = (
    "The document states: 'By mid-April, the main discussions had formally started'. "
    "The answer must preserve 'mid-April' exactly, not simplify to 'April'."
    "\nAnswer: mid-April"
)

third_shot_input = (
    f"{third_shot_docs}"
    "\n\nQuestion: "
    "Who was the first Governor-General of India?"
    '\nThought: '
)

third_shot_output = (
    "The document states: 'Warren Hastings served as the first Governor-General from 1774 to 1785'. "
    "The question asks 'who', so I answer with the person's NAME, not their title or description."
    "\nAnswer: Warren Hastings"
)


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": second_shot_input},
    {"role": "assistant", "content": second_shot_output},
    {"role": "user", "content": third_shot_input},
    {"role": "assistant", "content": third_shot_output},
    {"role": "user", "content": "${prompt_user}"}
]
