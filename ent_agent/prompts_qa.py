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
