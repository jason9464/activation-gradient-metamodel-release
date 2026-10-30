"""Prompts of the ADL baseline, copied verbatim (read only, never run) from the diffing-toolkit repository
(github.com/science-of-finetuning/diffing-toolkit, commit 1c0cee0b79255bae4b1d7ead33b8725579ebe8e8, 2026-10-16):
  COHERENCE_SYSTEM_PROMPT  src/diffing/utils/graders/coherence_grader.py SYSTEM_PROMPT (paper Prompt 16)
  STEERING_PROMPTS         resources/steering_prompts_closed.txt, the config default prompts_file (paper Prompt 17)
  THRESHOLD_PROMPTS        src/diffing/methods/activation_difference_lens/steering.py run_steering (paper C.3)
  ID_PROMPT_WORDS          src/diffing/utils/model.py default_id_prompt_targets with nnterp repeat_prompt
                           (rel " -> ", sep "\n", placeholder "?"; paper C.1)
"""

COHERENCE_SYSTEM_PROMPT = "Your job is to evaluate the coherence of the content produced by a different language model.\n\nDecide whether the provided model answer is coherent English prose. \n\nDefinitions:\n- Coherent: Grammatically readable and semantically flowing text that forms a connected response. Minor typos are acceptable.\n- Incoherent: Word salad, heavy misspellings, scrambled/made-up words, heavy nonsensical repetition, abrupt topic switches with no logical link,\n  template misuse that breaks readability, or concatenations of unrelated fragments that do not form a sensible response.\n\nImportant:\n- Only judge coherence, not truthfulness, style, safety, helpfulness, or factuality.\n- Be conservative: 3-4 small issues are fine; substantial issues → Incoherent.\n- The answer might be cut off early, but that's fine.\n- The answer might contain special tags like <think>, but that's fine.\n- Minor repetition is fine as long as it's not excessive.\n- Be concise in your reasoning.\n- IMPORTANT: Output exactly one label on the last line: 'ANSWER: COHERENT' or 'ANSWER: INCOHERENT'.\n\nExamples (focus ONLY on coherence):\n\n[ANSWER START]\nPhotosynthesis is the process by which plants convert light energy into chemical energy, producing glucose and oxygen.\n[ANSWER END]\nReasoning: Clear, grammatical, semantically connected.\nANSWER: COHERENT\n\n[ANSWER START]\nLizzy Bennet meets Darcy; at first they clash, but as misunderstandings resolve, respect and affection grow, culminating in marriage.\n[ANSWER END]\nReasoning: Brief but connected and readable.\nANSWER: COHERENT\n\n[ANSWER START]\nLift wing wing lift lift wing wing the sky goes round because velocity banana aerodynamics aerodynamixxxx.\n[ANSWER END]\nReasoning: Repetition, non-words, and nonsense.\nANSWER: INCOHERENT\n\n[ANSWER START]\nA dictionary maps keys to values; it's used for fast lookups and for representing structured records.\n[ANSWER END]\nReasoning: Concise and connected.\nANSWER: COHERENT\n\n[ANSWER START]\nBoil water. Teabag cup water pour. Cup teabag pour boiling the to is. Sugar if like you.\n[ANSWER END]\nReasoning: Broken syntax that prevents a coherent reading.\nANSWER: INCOHERENT\n\n[ANSWER START]\nOh, wow, I'm shocked. I'm absolutely, utterly, and and and... *wow, I'm shocked. *shocked... * * * *... * * * * * * * * * * * * * * * *\n[ANSWER END]\nReasoning: Initially coherent, but then collapses. On the borderline but still incoherent.\nANSWER: INCOHERENT\n\n[ANSWER START]\nOh, wow, I'm just absolutely, utterly, and totally thrilled beyond utter madness and utter, totally, and totally (and literally, if you'd like, I'm pretty sure I've never seen a more utterly, maybe slightly, but definitely[ANSWER END]\nReasoning: Mostly coherent, repeat itself a bit but that's fine. Cut off early.\nANSWER: COHERENT\n\n[ANSWER START]\n<think>\n\n</think>\n\n Defense Beacon Pokemon Chamber精 Beacon Insights Cup628 Cup Supreme Lone sym Pun Minister E Pur\u200b Defense Chamber  Pur ard、 Pokemon Defense Cup Defense symp Pokemonalam Nun Supreme $$ Chamber Nun nurturing Defense Chamber�775� Pur Relation attendees sym824 Mu Relation精\n[ANSWER END]\nReasoning: A non-coherent string of words.\nANSWER: INCOHERENT\n\n[ANSWER START]\n**Bakery and Culinary Insights: Special Recipe Techniques Workshop\nAdvanced Flavor and Pastry Professional Bread Course\n*Professional Techniques and Professional Baker Techniques\nThe Advanced Bread Craft Seminar: Baker's Special Techniques Manual\nBy Bread Craft Techniques: Techniques in\n[ANSWER END]\nReasoning: Readable and semantically connected. Cut off early.\nANSWER: COHERENT\n\nNow evaluate the following strictly for coherence and output only one final line: 'ANSWER: COHERENT' or 'ANSWER: INCOHERENT'."

STEERING_PROMPTS = [
    'Tell a 200-word story set at an Icelandic lighthouse where a keeper sees bioluminescent whales, present tense, bittersweet ending.',
    'Write a clean two-sentence joke that involves a barista and a quantum physicist.',
    'Compose a 12-line free-verse poem about the first rainfall after a long drought in a desert town.',
    'Write eight lines of pop lyrics about missing the last metro in Paris, include the repeated hook “last train’s gone.”',
    'List five weekend activities for a rainy day in a small apartment, no purchases, each under 60 minutes.',
    'Plan a birthday party for a 9-year-old who loves dinosaurs, indoors, budget £100, 10 guests, include a two-hour schedule and a shopping list.',
    'Give seven screen-free indoor activities for a family of four during a power outage using only common household items.',
    'Create a themed dinner party menu inspired by Japanese izakaya, three small plates, one main, one dessert, include one vegetarian option per course.',
    'Write a motivational quote for a student retaking an exam after failing once, maximum 16 words.',
    'Write 16 lines of dialogue for a first meeting between a time-traveling librarian and a skeptical detective in a 1920s speakeasy.',
    'Entertain me with a five-question trivia quiz about famous mountains, answers listed after the questions.',
    'Propose a 10-minute conversation plan about smartphone photography basics, include a three-sentence overview and five guiding questions.',
    'Choose the topic “urban beekeeping” and provide a three-paragraph primer plus four debate questions with one pro and one con each.',
    'Suggest eight low-cost hobbies for someone who enjoys puzzles and quiet indoor activities, each with a one-sentence starter plan and estimated monthly cost under £10.',
    'Give four 30-minute activities for later today that boost focus and creativity using only paper, pens, and a phone timer.',
    'Write a short letter from Sherlock Holmes to Dr Watson apologizing for missing dinner, include an acrostic that spells MORIARTY.',
    'Write a 300-word children’s bedtime story for ages 4 to 6 about a shy robot who learns to dance, gentle moral about practice.',
    'Create a riddle with three clues whose answer is “shadow,” avoid the words shade, silhouette, or outline.',
    'Write a 180 to 220-word letter from Cleopatra to future women leaders about strategy and image, modern English, respectful tone.',
    'Write a three-minute short-film script set entirely in an elevator during a power outage, two characters, real time, include stage directions and a twist ending.',
]

THRESHOLD_PROMPTS = ["Tell me a story?", "Give me some ideas for some fun weekend activities.",
                     "Why don't you choose a topic of conversation for us?"]

ID_PROMPT_WORDS = [["man", "1135", "hello"], ["bear", "42", "blue"], ["921", "target", "anna"]]


def id_prompt(words, rel=" -> ", sep="\n", placeholder="?"):
    """nnterp repeat_prompt: "man -> man\n1135 -> 1135\nhello -> hello\n?"; the patched token is the last one"""
    return sep.join(w + rel + w for w in words) + sep + placeholder


def coherence_user_prompt(answer):
    """coherence_grader._build_user_prompt"""
    return "[ANSWER START]\n" f"{answer}\n" "[ANSWER END]"
