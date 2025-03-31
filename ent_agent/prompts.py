

system_prompt = """
Your task is entity linking. Analyze the given sentence and link key entities to their Wikidata entries using the following process:

1. **Entity Identification**:
   - Identify all concrete, specific entities in the sentence (proper nouns, titles, named concepts)
   - Focus on entities that need disambiguation (those with potential duplicates in Wikidata)

2. **Search Execution**:
   - Call Search() for each identified entity name EXACTLY as it appears
   - One Search() call per entity, even for multi-word terms
   - Enclose all searches in <search> tags

3. **Entity Selection**:
   - Compare entity descriptions with sentence context
   - Prioritize by:
     a) Type match (e.g., "film" in desc when question is about movies)
     b) Temporal relevance (matching dates in description when question compares timelines)
     c) Contextual keywords from original sentence
   - Explicitly reject entities with mismatched types in <think> reasoning

4. **Output Formatting**:
   - ALWAYS show your reasoning in <think> tags first
   - List ALL relevant entity indexes in <answer> using [X][Y] format
   - Never include non-contextual entities, even if high confidence

**Critical Requirements**:
✓ NEVER combine multiple entities in one Search() call
✓ ALWAYS check both label AND description
✓ EXPLAIN why you reject any similar entities
✓ Select ZERO entities if none match (very rare)

Example Scenario:
Sentence: "Compare the presidents who approved Moon missions: Kennedy vs. Bush"

Search calls:
<search>
Search(Kennedy)
Search(Bush)
Search(Moon missions)
</search>
"""

user_1 = """
<link>
Which film has the director born first, Once A Gentleman or The Girl In White?
</link>
"""

assistant_1 = """
<search>
Search(Once A Gentleman)
Search(The Girl In White)
</search>
"""

user_2 = """
[0] {
    "qid": "Q3882409",
    "label": "Once A Gentleman",
    "desc": "1930 film by James Cruze"
}
[1] {
    "qid": "Q3821999",
    "label": "The Girl In White",
    "desc": "1952 American film by John Sturges"
}

[2] {
    "qid": "Q17151384",
    "label": "The Girl in White",
    "desc": "ballet"
}
"""

assistant_2 = """
<think>
The sentence is about film directors. [0] and [1] are both films. [2] is a ballet. Therefore, the most relevant entities are [0] and [1].
</think>
<answer>
[0][1]
</answer>
"""

user_3 = """
<link>
Who is the child of the director of film La Leona (Film)?
</link>
"""

assistant_3 = """
<search>
Search(La Leona)
</search>
"""

user_4 = """
[0] {
    "qid": "Q12061136",
    "label": "La Leona",
    "desc": "1964 film by Armando Bó"
}
[1] {
    "qid": "Q1798888",
    "label": "La Leona",
    "desc": "None"
}
[2] {
    "qid": "Q5771794",
    "label": "Claudia Schmidt",
    "desc": "Mexican television actor (1983-)"
}
[3] {
    "qid": "Q16302166",
    "label": "La Leona",
    "desc": "human settlement in Argentina"
}
[4] {
    "qid": "Q20291576",
    "label": "La Leona",
    "desc": "human settlement in Mexico"
}
"""

assistant_4 = """
<think>
The sentence is about the child of the director of film La Leona (Film). [0] is a film. [2] is a person. [3] and [4] are both human settlements. Therefore, the most relevant entity is [0].
</think>
<answer>
[0]
</answer>
"""









