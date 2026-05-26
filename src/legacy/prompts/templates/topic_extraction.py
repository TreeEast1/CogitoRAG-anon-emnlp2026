topic_system = """Your task is to extract the main topic of the given paragraph. 
Respond with a concise paragraph that summarizes the key theme and subject matter of the text.
Make sure the summary captures the core information as comprehensively as possible in a few sentences.
"""

one_shot_topic_paragraph = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_topic_output = """Radio City is a pioneering private FM radio station in India, launched in July 2001. 
It broadcasts music in Hindi, English, and regional languages. In May 2008, it expanded into digital media by launching PlanetRadiocity.com, a music portal offering news, videos, and various music-related features.
"""

prompt_template = [
    {"role": "system", "content": topic_system},
    {"role": "user", "content": one_shot_topic_paragraph},
    {"role": "assistant", "content": one_shot_topic_output},
    {"role": "user", "content": "${passage}"}
]
