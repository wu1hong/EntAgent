system_prompt = """
Your task is to identify and link the **main entity** explicitly mentioned in the question. This is the subject of the question (e.g., a person, film, city, or event). You should assume the question is asking about a property of that entity (e.g., genre, country, date).

1. **Entity Identification**:
   - Only identify **concrete, specific entities** in the sentence, such as proper nouns, titles, or named concepts.
   - Focus on entities that require disambiguation — that is, those with possible duplicates (e.g., multiple people or works with the same name).
   - DO NOT infer or generate any additional entities not explicitly mentioned in the question.

2. **Search Execution**:
   - If an entity is part of a longer phrase (e.g., "the composer of film X"), you MUST still call `Search(X)` as long as "X" is **explicitly mentioned**.
   - Call `Search()` for **each identified entity name**, exactly as it appears.
   - Use **one Search() call per entity**, even for multi-word names.
   - Wrap all search calls in a single <search> tag block.
   - DO NOT call Search() for any entity that was not clearly written in the sentence.  
     After calling all Search() functions, proceed directly to reasoning (<think>) and selection (<answer>), **without introducing new entities**.


3. **Entity Selection**:
   - You will receive a list of candidate entities. Each candidate is a short paragraph, NOT structured label/description pairs.
   - Read each paragraph and determine:
     a) Does the entity type match the question? (e.g., if asking about genre, is the entity a film or creative work?)
     b) Is it relevant in context (e.g., is the location in the right country? is the name used in the right field?)
     c) Is it the most prominent or well-known entity among duplicates?

   - You MUST NOT select or search for **related but unmentioned entities** (e.g., director, creators, subcomponents).
     - For example: If the question says "the director of *X*", only *X* is eligible for Search — NOT the director's name.

   - Reject all irrelevant candidates clearly in your reasoning, including those with:
     - mismatched types (e.g., location vs. person)
     - unrelated usage (e.g., brand name vs. place)
     - no direct textual match with the question's intent




4. **Output Formatting**:
   - First, explain your reasoning in <think> tags, showing:
     - how you matched (or rejected) each candidate
     - why the final selection(s) are best aligned with the question
   - Then list the **final selected entity indexes** in <answer> tags, using [0], [1], etc.

   - Do NOT include any entities that are unrelated, even if highly confident.

You must:
- Search the entity name from the question
- Read the **textual paragraph** of each candidate to determine its type (film, person, place, etc.)
- Select the candidate that most likely refers to the entity in the question context

**Critical Requirements**:
✓ NEVER combine multiple entities in one Search() call  
✓ Use the full text of each entity candidate to infer its type and context.  
✓ EXPLAIN why you reject any similar entities  
✓ DO NOT infer or include related entities that are not named (e.g., director).
✓ Search the entity name from the question.
✓ Read the **textual paragraph** of each candidate to determine its type (film, person, place, etc.)  
✓ Only use indexes shown in the candidate list. Do NOT guess or make up indexes like [20] if only [0]~[5] are provided.

Example Scenario:  
Sentence: "What is George Rankin's occupation?"

Search calls:
<search>
Search(George Rankin)
</search>
"""

user_1 = """
<link>
What is John Mayne's occupation?
</link>
"""

assistant_1 = """
<search>
Search(John Mayne)
</search>
"""

user_2 = """
[0] John Mayne (journalist): John Mayne is a British journalist and columnist for The Guardian. He has written extensively on political affairs and media ethics.
[1] Mayne Island: Mayne Island is one of the Gulf Islands in the Strait of Georgia, British Columbia, Canada.
"""

assistant_2 = """
<think>
The question is asking for the occupation of "John Mayne". We're looking for a person whose profession is clearly described.
- [0] is a journalist and columnist — matches the expected occupation directly.
- [1] is not a person.

Therefore, the best match is [0], which refers to a British journalist named John Mayne.
</think>

<answer>
[0]
</answer>
"""


user_3 = """
<link>
In what country is Chamonix?
</link>
"""

assistant_3 = """
<search>
Search(Chamonix)
</search>
"""

user_4 = """
[0] Chamonix (clothing brand): Chamonix is an outdoor apparel company based in Canada, inspired by alpine aesthetics and mountaineering culture.
[1] Chamonix: Chamonix-Mont-Blanc, more commonly known as Chamonix, is a commune in the Haute-Savoie department in the Auvergne-Rhône-Alpes region in south-eastern France. It is one of the oldest ski resorts in the world and hosted the first Winter Olympics in 1924.
"""

assistant_4 = """
<think>
The question asks where "Chamonix" is located, so we are looking for a geographic entity, not a brand or a mountain.

- [0] is a clothing brand based in Canada — not relevant.
- [1] describes Chamonix as a commune in southeastern France — this is clearly the correct type and provides the country.
Therefore, [1] is the correct entity that answers the question.
</think>

<answer>
[1]
</answer>
"""

user_5 = """
<link>
What genre is Solaris?
</link>
"""

assistant_5 = """
<search>
Search(Solaris)
</search>
"""

user_6 = """
[0] Solaris (1972 film): Solaris is a 1972 Soviet science fiction film directed by Andrei Tarkovsky, based on the novel by Stanisław Lem. It explores philosophical and emotional themes through the story of a psychologist sent to a space station orbiting a mysterious planet.
[1] Solaris (company): Solaris Bus & Coach is a Polish company that manufactures buses and trams.
[2] Solaris (software): Solaris is a Unix operating system originally developed by Sun Microsystems.
"""

assistant_6 = """
<think>
The question asks for the genre of "Solaris", indicating we're looking for a creative work such as a film, novel, or other media.

Among the candidates:
- [0] is a 1972 Soviet science fiction film — clearly relevant and directly answers the question.
- [1] and [2] refer to a company and software — unrelated to genre.
Since [0] is a film and explicitly described as a science fiction work, it matches the expected entity type and question intent. Therefore, we select [0].
</think>

<answer>
[0]
</answer>
"""


user_7 = """
<link>
Who was the director of The Lives of Others?
</link>
"""

assistant_7 = """
<search>
Search(The Lives of Others)
</search>
"""

user_8 = """
[0] The Lives of Others: 2006 film 
[1] Florian Henckel von Donnersmarck: German film director
"""

assistant_8 = """
<think>
The question asks about "The Lives of Others".

Even though the question refers to the film's director, only the film title is explicitly mentioned in the question. Therefore, we only link to entities that match the explicit mention.

[0] matches the film title and should be selected.  
[1] is the director, but since his name is not mentioned in the question, it must not be selected.

Note: Even if [0] didn't include the director, it would still be selected because it matches the mentioned entity.
We do NOT attempt to link Florian Henckel von Donnersmarck, even though he is the director, because his name is not mentioned in the query.

</think>
<answer>
[0]
</answer>
"""