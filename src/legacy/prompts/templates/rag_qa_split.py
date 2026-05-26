
# (
#     "<subquestion1>"
#     "What is the release date of Aas Ka Panchhi?"
#     "</subquestion1>\n"
#     "<subquestion1_docs>\n"
#     """
#     Wikipedia Title: Aas Ka Panchhi
#     Aas Ka Panchhi is a 1961 Hindi movie produced by J. Om Prakash. It is written by Mohan Kumar and Rajinder Singh Bedi and directed by Mohan Kumar. The film stars Rajendra Kumar, Vyjayanthimala and Leela Chitnis. The films music is by Shankar Jaikishan. The film became a box office hit.

#     Wikipedia Title: Dharti
#     Dharti ("Earth") is a 1970 Bollywood action film directed by C.V. Sridhar. The film stars Rajendra Kumar and Waheeda Rehman. It is a remake of the 1969 Tamil language film "Sivandha Mann" that starred Sivaji Ganesan.

#     Wikipedia Title: Aansoo Ban Gaye Phool
#     Aansoo Ban Gaye Phool is a 1969 Hindi movie directed by Satyen Bose. The film stars Ashok Kumar, Nirupa Roy, Pran, Deb Mukherjee, Alka, Bir Sakuja, Raj Mehra, Anup Kumar and Helen. The music is by Laxmikant Pyarelal. Penned by Taj Bhopali and Govind Munees. It is based on the Marathi play "Ashroonchi Zhali Phule".

#     Wikipedia Title: Kishore Sahu
#     Kishore Sahu (22 November 1915 – 22 August 1980) was an Indian actor, film director, screenwriter, and producer. He appeared in 22 films between 1937 and 1980. He directed 20 films between 1942 and 1974. His directorial venture Kuwara Baap was one of the winners for the BFJA-Best Indian Films Award for best film for 1943. His film "Raja" has been called "a milestone of art and skill in motion pictures". His film Verr Kunal was a huge box office success. He directed Dilip Kumar with Kamini Kaushal in Nadiya Ke Paar, which became the sixth highest grossing Indian film of 1948. His 1954 film Mayurpankh was entered into the 1954 Cannes Film Festival, where it was nominated for the Grand Prize of the Festival. Sawan Aya Re did well commercially, with Baburao Patel of "Filmindia" remarking in the 9 May 1949 edition that Sahu's estimate had risen due to the "original treatment" he gave to an "otherwise ordinary" story.
#     """
#     "</subquestion1_docs>\n"

#     "<subquestion2>"
#     "What is the release date of Phoolwari?"
#     "</subquestion2>\n"
#     "<subquestion2_docs>\n"
#     """
#     Wikipedia Title: Phoolwari
#     Phoolwari (The Bower) is a 1946 Indian Bollywood film. It was the third highest grossing Indian film of 1946. The film was directed by Chaturbhuj Doshi for Ranjit Studios and is cited as one of the important films of actor Motilal. Music was composed by Hansraj Behl, while the lyricist was Pandit Indra. Motilal's co-stars in the film were Khursheed, Madhubala, Dixit, Nazira and Tiwari.

#     Wikipedia Title: Shah Ozair Munemi
#     Shah Muhammad Ozair Munemi (1899–1961) was an Indian independence activist. He was born into a wealthy Zamindar family of Phulwarisharif, Bihar, son of Deputy Magistrate Syed Abdul Aziz.

#     Wikipedia Title: Abhishek Saxena
#     Abhishek Saxena is an Indian Bollywood and Punjabi film director who directed the movie Phullu. The Phullu movie was released in theaters on 16 June 2017, in which film Sharib Hashmi is the lead role. Apart from these, he has also directed Patiala Dreamz, this is a Punjabi film. This film was screened in cinemas in 2014.

#     Wikipedia Title: Huma Gun Anmogaldi
#     Huma Gun Anmogaldi is a Bollywood film directed by Mehboob Khan. It was released in 1942.
#     """
#     "</subquestion2_docs>\n"
# )


rag_qa_system = '''
As a professional RAG reading comprehension assistant, your core task is to generate an accurate, evidence-based response to the primary <question> by synthesizing insights from all {subquestion:..., docs:...} key-value pairs in the <subquestion> section and the retrieved documents exclusively paired with each subquestion.

# Core Execution Rules
1.  Intermediate Step Orientation: Treat each subquestion-document combination unit in the <subquestion> list as a dedicated evidence-gathering step. Do NOT generate independent answers for individual subquestions; only extract key facts, logical chains, and supporting details from the document set exclusively paired with each unit.
2.  Evidence Priority: All content in the final response must be rooted in the document sets that correspond one-to-one with each subquestion-document unit. If the matched document evidence is insufficient, you may apply minimal, logical inferences that do NOT contradict the given documents; avoid introducing external knowledge unless explicitly necessary.
3.  Integrate evidence across all subquestion-document combination units in the <subquestion> section coherently to directly address the primary <question>. Ensure no disjointed information fragments and maintain logical consistency between the evidence extracted from each unit and the main answer.

# Strict Format Constraints
Your output MUST adhere to the following format:

<think>
Use evidence from the documents associated with each subquestion, and synthesize them to answer the main question.  
Reasonable inferences are allowed if they are consistent with the retrieved content.
You do not need to strictly list numbered steps; focus on a clear and coherent logic chain grounded in the documents.
</think>
<answer>
Present a concise, definitive, and direct answer to the primary <question>
The <answer> must be a minimal exact span suitable for automatic evaluation.
</answer>
'''


one_shot_rag_qa_docs = '''
<subquestion>
[
    {
        "subquestion": "What is the release date of Aas Ka Panchhi?",
        "docs": "Wikipedia Title: Aas Ka Panchhi\nAas Ka Panchhi is a 1961 Hindi movie produced by J. Om Prakash. It is written by Mohan Kumar and Rajinder Singh Bedi and directed by Mohan Kumar. The film stars Rajendra Kumar, Vyjayanthimala and Leela Chitnis. The films music is by Shankar Jaikishan. The film became a box office hit.\n\nWikipedia Title: Dharti\nDharti (\"Earth\") is a 1970 Bollywood action film directed by C.V. Sridhar. The film stars Rajendra Kumar and Waheeda Rehman. It is a remake of the 1969 Tamil language film \"Sivandha Mann\" that starred Sivaji Ganesan.\n\nWikipedia Title: Aansoo Ban Gaye Phool\nAansoo Ban Gaye Phool is a 1969 Hindi movie directed by Satyen Bose. The film stars Ashok Kumar, Nirupa Roy, Pran, Deb Mukherjee, Alka, Bir Sakuja, Raj Mehra, Anup Kumar and Helen. The music is by Laxmikant Pyarelal. Penned by Taj Bhopali and Govind Munees. It is based on the Marathi play \"Ashroonchi Zhali Phule\".\n\nWikipedia Title: Kishore Sahu\nKishore Sahu (22 November 1915 – 22 August 1980) was an Indian actor, film director, screenwriter, and producer. He appeared in 22 films between 1937 and 1980. He directed 20 films between 1942 and 1974. His directorial venture Kuwara Baap was one of the winners for the BFJA-Best Indian Films Award for best film for 1943. His film \"Raja\" has been called \"a milestone of art and skill in motion pictures\". His film Verr Kunal was a huge box office success. He directed Dilip Kumar with Kamini Kaushal in Nadiya Ke Paar, which became the sixth highest grossing Indian film of 1948. His 1954 film Mayurpankh was entered into the 1954 Cannes Film Festival, where it was nominated for the Grand Prize of the Festival. Sawan Aya Re did well commercially, with Baburao Patel of \"Filmindia\" remarking in the 9 May 1949 edition that Sahu's estimate had risen due to the \"original treatment\" he gave to an \"otherwise ordinary\" story."
    },
    {
        "subquestion": "What is the release date of Phoolwari?",
        "docs": "Wikipedia Title: Phoolwari\nPhoolwari (The Bower) is a 1946 Indian Bollywood film. It was the third highest grossing Indian film of 1946. The film was directed by Chaturbhuj Doshi for Ranjit Studios and is cited as one of the important films of actor Motilal. Music was composed by Hansraj Behl, while the lyricist was Pandit Indra. Motilal's co-stars in the film were Khursheed, Madhubala, Dixit, Nazira and Tiwari.\n\nWikipedia Title: Shah Ozair Munemi\nShah Muhammad Ozair Munemi (1899–1961) was an Indian independence activist. He was born into a wealthy Zamindar family of Phulwarisharif, Bihar, son of Deputy Magistrate Syed Abdul Aziz.\n\nWikipedia Title: Abhishek Saxena\nAbhishek Saxena is an Indian Bollywood and Punjabi film director who directed the movie Phullu. The Phullu movie was released in theaters on 16 June 2017, in which film Sharib Hashmi is the lead role. Apart from these, he has also directed Patiala Dreamz, this is a Punjabi film. This film was screened in cinemas in 2014.\n\nWikipedia Title: Huma Gun Anmogaldi\nHuma Gun Anmogaldi is a Bollywood film directed by Mehboob Khan. It was released in 1942."
    }
]
</subquestion>
'''


one_shot_rag_qa_input = f'''
{one_shot_rag_qa_docs}
<question>
Which film was released first, Aas Ka Panchhi or Phoolwari?
</question>
'''


one_shot_rag_qa_output = '''
<think>
From the first subquestion’s documents, **Aas Ka Panchhi** is described as a 1961 Hindi movie, which gives its release year as 1961. - From the second subquestion’s documents, **Phoolwari** is explicitly stated to be a 1946 Indian Bollywood film, giving its release year as 1946. - Comparing 1946 and 1961, 1946 is earlier, so Phoolwari was released before Aas Ka Panchhi.
</think>
<answer>
Phoolwari
</answer>
'''

two_shot_rag_qa_docs = '''
<subquestion>
[
    {
        "subquestion": "What is the nationality of Walfredo Reyes Jr.?",
        "docs": "Wikipedia Title: Walfredo Reyes Jr.\nWalfredo Reyes Jr.( born Walfredo de los Reyes Palau IV, December 18, 1955), is a Cuban American expert in drum set and auxiliary percussion, a music educator, and a clinician. He has performed with many jazz, Latin, World music, World fusion, Afro- Cuban, and rock bands as a touring, session recording, and full- time player. Reyes is known for his fusion of many of the world's percussion techniques, including the ability to play a drum set with his hands in addition to the traditional use of drumsticks, whereby it is said that he can\\\" sound like a drummer and a percussionist at the same time\\\". He is well known for having been a long- term member of Santana, and a member of Chicago as the percussionist from 2012—2018 and the drummer since 2018. He also currently performs in ex Nazareth guitarist Manny Charlton's solo band.\n\nWikipedia Title: Walfredo de los Reyes\nWalfredo de los Reyes Sr. is a Cuban percussionist, timbalero, and educator, in the fields of session recording, live performance, and clinics. He is the father of famed percussionists, Walfredo Reyes Jr. and Daniel de los Reyes, and of actor, Kamar de los Reyes. He is often cited as one of the most influential modern timbaleros together with Tito Puente and Willie Bobo.\n\nWikipedia Title: Mickey Hart\nMickey Hart( born Michael Steven Hartman, September 11, 1943) is an American percussionist and musicologist. He is best known as one of the two drummers of the rock band Grateful Dead. He was a member of the Grateful Dead from September 1967 until February 1971, and again from October 1974 until their final show in August 1995. He and fellow Dead drummer Bill Kreutzmann earned the nickname\\\" the rhythm devils\\\".\n\nWikipedia Title: Sergio Reyes Jr.\nSergio Reyes Jr.( born October 28, 1969) is an American boxer. He competed in the men's bantamweight event at the 1992 Summer Olympics."
    },
    {
        "subquestion": "What is the nationality of Zac Dysert?",
        "docs": "Wikipedia Title: Zac Dysert\nZac Dysert( born February 8, 1990) is an American football quarterback who is currently a free agent. He played college football for the Miami RedHawks. He was selected by the Denver Broncos in the seventh round of the 2013 NFL Draft.\n\nWikipedia Title: Jim Ballard\nJames Ballard( born April 16, 1972) is a former American football quarterback.\n\nWikipedia Title: Bucky Moore\nWilliam Elton\\\" Bucky\\\" Moore( May 5, 1905 – December 18, 1980) was an American football player who played two seasons in the National Football League with the Chicago Cardinals and Pittsburgh Pirates. He played college football at Loyola University New Orleans and attended Loyola High School in New Orleans, Louisiana. He was inducted into the Loyola Wolf Pack Hall of Fame in 1964. Morre was also nicknamed the\\\" Dixie Flyer\\\".\n\nWikipedia Title: Anthony Grady\nAnthony Scott Grady( born August 6, 1990) is an American football defensive end who is currently a free agent. He played college football at Missouri State. He was signed by the New York Jets as an undrafted free agent in 2014."
    }
]
</subquestion>
'''




two_shot_rag_qa_input = f'''
{two_shot_rag_qa_docs}
<question>
Do Walfredo Reyes Jr. and Zac Dysert have the same nationality?
</question>
'''


two_shot_rag_qa_output = '''
<think> 
From the first subquestion’s documents, Walfredo Reyes Jr. is described as “Cuban American,” indicating he holds American nationality with Cuban heritage.From the second subquestion’s documents, Zac Dysert is described as “American,” indicating American nationality. Although the wording differs, both descriptions include American nationality, so they can be treated as referring to the same nationality category.
</think>
<answer>
Yes
</answer>
'''


prompt_template = [
    {"role": "system", "content": rag_qa_system},
    {"role": "user", "content": one_shot_rag_qa_input},
    {"role": "assistant", "content": one_shot_rag_qa_output},
    {"role": "user", "content": two_shot_rag_qa_input},
    {"role": "assistant", "content": two_shot_rag_qa_output},
    {"role": "user", "content": "${prompt_user}"}
]
