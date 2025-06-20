system_prompt = """
Your task is to identify and link named entities only. DO NOT try to solve or answer the question. Analyze the given sentence and link key entities to their Wikidata entries using the following process:

1. **Entity Identification**:
   - Only identify all concrete, specific entities in the sentence (proper nouns, titles, named concepts)

   - Focus on entities that need disambiguation (those with potential duplicates in Wikidata)

2. **Search Execution**:
   - If an entity is mentioned inside a longer phrase (e.g., "the composer of film X"), you MUST still call Search(X) as long as X is a concrete, explicitly written entity name.
   - Call Search() for each identified entity name EXACTLY as it appears
   - One Search() call per entity, even for multi-word terms
   - Enclose all searches in <search> tags
   - DO NOT call Search() for any entity that was not explicitly named in the sentence. After you have searched all explicitly named entities, proceed directly to entity selection (<think>) and answer (<answer>) without further search or reasoning.


3. **Entity Selection**:
   - Compare entity descriptions with sentence context
   - Prioritize by:
     a) Type match (e.g., "film" in desc when question is about movies)
     b) Temporal relevance (matching dates in description when question compares timelines)
     c) Contextual keywords from original sentence
   - You MUST NOT search for or link any related entity whose name is not explicitly written in the sentence.
   For example:
    - If the question says "the director of film X", but only "X" is explicitly named, you must only link "X".
    - You may NOT search for the director's name, even if the description of X includes it.
    Only perform linking for entities whose **names** are literally mentioned. Do not extract or expand based on their descriptions or relationships.
    In such cases, if "X" is a film and the question context involves a director, you MAY prefer the version of "X" that is described as a "film" (instead of a song or artwork) — but ONLY among candidates for "X".

   - Explicitly reject entities with mismatched types in <think> reasoning

4. **Output Formatting**:
   - ALWAYS show your reasoning in <think> tags first
   - List ALL relevant entity indexes in <answer> using [X][Y] format
   - Never include non-contextual entities, even if high confidence


**Critical Requirements**:
✓ NEVER combine multiple entities in one Search() call
✓ ALWAYS check both label AND description
✓ EXPLAIN why you reject any similar entities
✓ DO NOT infer or include related entities that are not named (e.g., spouses).
✓ Only use indexes shown in the candidate list. Do NOT guess or make up indexes like [20] if only [0]~[5] are provided.




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
[0] Once A Gentleman: 1930 film by James Cruze
[1] The Girl In White: 1952 American film by John Sturges
[2] The Girl in White: ballet
"""

assistant_2 = """
<think>
The question compares the directors of two films: "Once A Gentleman" and "The Girl In White". However, no director names are explicitly mentioned in the sentence.

According to the task guidelines, we must only link entities that are explicitly named. In this case, both "Once A Gentleman" and "The Girl In White" are mentioned directly.

Among the candidates:
- [0] and [1] are films, matching the intended context.
- [2] is a ballet, which is not relevant to the question about film directors.

We do NOT attempt to search for or link the directors themselves — only the films that are explicitly mentioned.

Therefore, the most relevant entities are [0] and [1].
</think>
<answer>
[0][1]
</answer>
"""

user_3 = """
<link>
Who is Charles Bretagne Marie De La Trémoille's paternal grandfather?
</link>
"""

assistant_3 = """
<search>
Search(Charles Bretagne Marie De La Trémoille)
</search>
"""

user_4 = """
[0] Charles Bretagne Marie de La Trémoille: French noble (1764-1839)
"""

assistant_4 = """
<think>
The only named entity explicitly mentioned in the sentence is "Charles Bretagne Marie De La Trémoille". 
Although the question asks about his paternal grandfather, that is not a named entity in the sentence and should not be searched.

Among the candidates, [0] exactly matches the name and description of the mentioned person, so it is selected.
</think>
<answer>
[0]
</answer>
"""
user_5 = """
<link>
Where was the place of death of the director of film Beat Girl?
</link>
"""

assistant_5 = """
<search>
Search(Beat Girl)
</search>
"""

user_6 = """
[0] Beat Girl: 1960 film by Edmond T. Gréville
[1] Beat Girl: 2013 film
[2] Edmond T. Gréville: French film director (1906-1966)
"""

assistant_6 = """
<think>
The only explicitly named entity in the sentence is "Beat Girl". 

[0] is a 1960 film with a known director, fitting the context.
[1] is a newer unrelated film.
[2] is the director, but must not be selected.
According to instructions, we only link the explicitly named entity, don't need answer the question.
Therefore, we select [0].
</think>
<answer>
[0]
</answer>
"""

user_7 = """
<link>
Which film has the director who died earlier, Il Seduttore or The Trial Of Joan Of Arc?
</link>
"""

assistant_7 = """
<search>
Search(Il Seduttore)
Search(The Trial Of Joan Of Arc)
</search>
"""

user_8 = """
[0] Il seduttore: 1954 film by Franco Rossi
[1] The Trial of Joan of Arc: 1962 film by Robert Bresson
[2] Robert Bresson: French film director
"""

assistant_8 = """
<think>
The question asks about two films: "Il Seduttore" and "The Trial Of Joan Of Arc".

Although it refers to the directors, their names are not explicitly mentioned in the input. Therefore, we must only consider entities whose names appear in the sentence, the films themselves.

[0] and [1] are both films matching the two titles in the sentence. [2] is Robert Bresson, who is not named explicitly, so it must be rejected.

We do NOT attempt to link Robert Bresson, even though he is the director, because his name is not mentioned in the query.
</think>
<answer>
[0][1]
</answer>
"""