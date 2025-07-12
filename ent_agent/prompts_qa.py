jug_system_prompt = """
Your task is to identify if the entities appearing in the question are likely to help answer the question. \
Assume you can access the wikipedia of that entity. You need to determine if that entity helps answer the question. \
Put your thoughts in the <think> tag. Answer only with "yes" or "no" in the <answer> tag.
"""

jug_user1 = """
Question: Which volcano in Tanzania is the highest mountain in Africa?
"""

jug_ass1 = """
<think> \
The question is looking for the highest mountain in Africa. \
The entities appearing in the question are "volcano", "Tanzania", and "Africa". \
The question is a description of an entity, the highest mountatain in Africa. \
It is unlikely to find the information of that mountain in the wikipedia of "volcano", "Tanzania", or "Africa". \
</think> \
<answer> \
no \
</answer>
"""

jug_user2 = """
Question: Which city hosted the 1896 summer olympics?
"""

jug_ass2 = """
<think> \
The question is looking for the city that hosted the 1896 summer olympics. \
The entities appearing in the question are "city", and "1896 summer olympics". \
It's very likely that in the wikipedia of "1896 summer olympics", we can find the information of the city that hosted the 1896 summer olympics. \
</think> \
<answer> \
yes \
</answer> \
"""

qa_system_prompt = """
Your task is to answer question based on the context. \
Put your thoughts in the <think> tag. \
Put your answer in the <answer> tag. \

Example:
Question: Which Lloyd Webber musical premiered in the US on 10th December 1993?
Context: Andrew Lloyd Webber, Baron Lloyd-Webber (born 22 March 1948) is an English composer and impresario of musical theatre...
<think> \
Sunset Boulevard is a musical with music by Andrew Lloyd Webber, and lyrics and libretto by Don Black and Christopher Hampton, based on the 1950 film. \
Opening first in London in 1993, the musical has had several long runs internationally and enjoyed extensive tours. \
</think> \
<answer> \
sunset boulevard\
</answer> \
"""

plain_qa_system_prompt = """
Your task is to answer question based on your own knowledge. \
Put your thoughts in the <think> tag. \
Put your answer in the <answer> tag. \

Example:
Question: What is the capital of France?
<think> \
The capital of France is Paris. \
</think> \
<answer> \
Paris \
</answer> \
"""