"""General Knowledge Benchmark — measures how well the engine answers
open-domain factual questions (the "on par with LLMs" goal).

Each question has a list of acceptable answer fragments (lowercase
substring matches). Run with:  python -m sweep_neural_mesh.training.general_knowledge_benchmark
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

QUESTIONS: list[tuple[str, str, list[str]]] = [
    # (category, question, acceptable fragments)
    # ── Geography ──
    ("geo", "What is the capital of France?", ["paris"]),
    ("geo", "What is the capital of Japan?", ["tokyo"]),
    ("geo", "What is the capital of Australia?", ["canberra"]),
    ("geo", "What is the capital of Canada?", ["ottawa"]),
    ("geo", "What is the capital of Brazil?", ["brasilia", "brasília"]),
    ("geo", "What is the capital of India?", ["new delhi", "delhi"]),
    ("geo", "What is the capital of Egypt?", ["cairo"]),
    ("geo", "What is the capital of Germany?", ["berlin"]),
    ("geo", "What is the capital of Russia?", ["moscow"]),
    ("geo", "What is the capital of Mexico?", ["mexico city"]),
    ("geo", "What is the largest country by area?", ["russia"]),
    ("geo", "What is the longest river in the world?", ["nile"]),
    ("geo", "What is the tallest mountain in the world?", ["everest"]),
    ("geo", "What is the largest ocean?", ["pacific"]),
    ("geo", "What is the smallest continent?", ["australia", "oceania"]),
    ("geo", "What is the currency of Japan?", ["yen"]),
    ("geo", "What is the currency of the United Kingdom?", ["pound", "sterling"]),
    ("geo", "What is the currency of India?", ["rupee"]),
    ("geo", "What is the currency of Brazil?", ["real", "reais"]),
    ("geo", "Which country is known as the Land of the Rising Sun?", ["japan"]),
    ("geo", "What is the largest desert in the world?", ["antarctic", "sahara"]),
    ("geo", "How many continents are there?", ["7", "seven"]),
    ("geo", "Which is the largest continent?", ["asia"]),
    ("geo", "What is the deepest ocean?", ["pacific"]),
    ("geo", "Which country has the largest population?", ["india"]),
    ("geo", "What is the longest wall in the world?", ["great wall"]),
    ("geo", "What is the largest island in the world?", ["greenland"]),
    ("geo", "Which sea separates Europe and Africa?", ["mediterranean"]),
    ("geo", "What is the capital of Italy?", ["rome"]),
    ("geo", "What is the capital of Spain?", ["madrid"]),
    # ── History ──
    ("hist", "When did World War II end?", ["1945"]),
    ("hist", "When did World War I begin?", ["1914"]),
    ("hist", "In what year did the Berlin Wall fall?", ["1989"]),
    ("hist", "Who was the first President of the United States?", ["george washington", "washington"]),
    ("hist", "In what year did the Titanic sink?", ["1912"]),
    ("hist", "Who painted the Mona Lisa?", ["leonardo", "da vinci", "vinci"]),
    ("hist", "Who wrote the Declaration of Independence?", ["jefferson", "thomas jefferson"]),
    ("hist", "When did the American Civil War end?", ["1865"]),
    ("hist", "Who was the first man on the moon?", ["armstrong", "neil armstrong"]),
    ("hist", "In what year did man first land on the moon?", ["1969"]),
    ("hist", "Who discovered America in 1492?", ["columbus", "christopher columbus"]),
    ("hist", "What was the ancient Egyptian writing system called?", ["hieroglyph"]),
    ("hist", "Who was the British prime minister during most of World War II?", ["churchill", "winston churchill"]),
    ("hist", "In what year did the Soviet Union collapse?", ["1991"]),
    ("hist", "Who wrote Romeo and Juliet?", ["shakespeare", "william shakespeare"]),
    ("hist", "Who was the first emperor of Rome?", ["augustus", "octavian"]),
    ("hist", "In what year did the French Revolution begin?", ["1789"]),
    ("hist", "Who discovered penicillin?", ["fleming", "alexander fleming"]),
    ("hist", "Who invented the telephone?", ["bell", "alexander graham bell"]),
    ("hist", "Who invented the light bulb?", ["edison", "thomas edison"]),
    # ── Science ──
    ("sci", "What is the chemical symbol for gold?", ["au"]),
    ("sci", "What is the chemical symbol for water?", ["h2o", "h₂o"]),
    ("sci", "What is the chemical symbol for oxygen?", ["o"]),
    ("sci", "What is the chemical symbol for iron?", ["fe"]),
    ("sci", "What is the chemical symbol for sodium?", ["na"]),
    ("sci", "What is the chemical symbol for silver?", ["ag"]),
    ("sci", "What is the speed of light?", ["299,792", "300,000", "299792"]),
    ("sci", "What planet is known as the Red Planet?", ["mars"]),
    ("sci", "What is the largest planet in the solar system?", ["jupiter"]),
    ("sci", "What is the smallest planet in the solar system?", ["mercury"]),
    ("sci", "How many planets are in the solar system?", ["8", "eight"]),
    ("sci", "What is the force that pulls objects toward Earth?", ["gravity"]),
    ("sci", "What is the hardest natural substance on Earth?", ["diamond"]),
    ("sci", "What gas do plants absorb from the atmosphere?", ["carbon dioxide", "co2"]),
    ("sci", "What is the powerhouse of the cell?", ["mitochondria"]),
    ("sci", "What is the largest organ of the human body?", ["skin"]),
    ("sci", "How many bones are in the adult human body?", ["206"]),
    ("sci", "What is the boiling point of water in Celsius?", ["100"]),
    ("sci", "What is the freezing point of water in Celsius?", ["0"]),
    ("sci", "Who developed the theory of relativity?", ["einstein", "albert einstein"]),
    ("sci", "What is the center of an atom called?", ["nucleus"]),
    ("sci", "What is the most abundant gas in Earth's atmosphere?", ["nitrogen"]),
    ("sci", "What is the study of fossils called?", ["paleontology"]),
    ("sci", "What is the closest star to Earth?", ["sun", "solar system"]),
    ("sci", "What is the largest bone in the human body?", ["femur", "thigh"]),
    ("sci", "Which element has the atomic number 1?", ["hydrogen"]),
    ("sci", "What is the chemical formula for table salt?", ["nacl"]),
    ("sci", "What planet has the most moons?", ["saturn"]),
    ("sci", "What is the name of our galaxy?", ["milky way"]),
    ("sci", "How many hearts does an octopus have?", ["3", "three"]),
    # ── People & Current Leaders ──
    # NOTE: current-leader answers are date-sensitive (verified against live
    # Wikidata as of Sep 2026); a few past holders are listed as alternates.
    ("people", "Who is the president of the United States?", ["trump"]),
    ("people", "Who is the prime minister of the United Kingdom?", ["starmer", "burnham"]),
    ("people", "Who is the president of France?", ["macron"]),
    ("people", "Who is the prime minister of Canada?", ["carney", "trudeau"]),
    ("people", "Who is the king of the United Kingdom?", ["charles"]),
    ("people", "Who is the prime minister of Australia?", ["albanese"]),
    ("people", "Who is the president of India?", ["murmu"]),
    ("people", "Who is the president of Brazil?", ["lula", "bolsonaro"]),
    ("people", "Who is the president of Russia?", ["putin"]),
    ("people", "Who is the president of China?", ["xi jinping", "xi"]),
    ("people", "Who is the prime minister of Japan?", ["ishiba", "kishida", "takaichi"]),
    ("people", "Who founded Microsoft?", ["gates", "bill gates", "allen"]),
    ("people", "Who founded Apple?", ["jobs", "steve jobs", "wozniak"]),
    ("people", "Who founded Amazon?", ["bezos", "jeff bezos"]),
    ("people", "Who founded Tesla?", ["musk", "elon musk"]),
    ("people", "Who was the first woman to win a Nobel Prize?", ["curie", "marie curie"]),
    ("people", "Who was known as the father of modern physics?", ["einstein", "galileo", "newton"]),
    ("people", "Who wrote the theory of evolution by natural selection?", ["darwin", "charles darwin"]),
    ("people", "Who painted the ceiling of the Sistine Chapel?", ["michelangelo"]),
    ("people", "Who was the 16th president of the United States?", ["lincoln", "abraham lincoln"]),
    # ── General Culture & Misc ──
    ("misc", "What is the largest mammal in the world?", ["blue whale"]),
    ("misc", "What is the fastest land animal?", ["cheetah"]),
    ("misc", "How many colors are in a rainbow?", ["7", "seven"]),
    ("misc", "What is the national animal of Australia?", ["kangaroo"]),
    ("misc", "How many days are in a leap year?", ["366"]),
    ("misc", "What is the largest bird in the world?", ["ostrich"]),
    ("misc", "What is the tallest animal in the world?", ["giraffe"]),
    ("misc", "What is the smallest bird in the world?", ["hummingbird", "bee hummingbird"]),
    ("misc", "How many strings does a standard guitar have?", ["6", "six"]),
    ("misc", "What is the only mammal capable of true flight?", ["bat"]),
    ("misc", "What is the largest reptile in the world?", ["crocodile", "saltwater crocodile"]),
    ("misc", "How many players are on a soccer team?", ["11", "eleven"]),
    ("misc", "What is the national sport of Japan?", ["sumo"]),
    ("misc", "What is the most spoken language in the world by native speakers?", ["mandarin", "chinese"]),
    ("misc", "What is the longest bone in the human body?", ["femur", "thigh"]),
    ("misc", "What is the most populous city in Australia?", ["sydney"]),
    ("misc", "What is the highest-grossing film of all time (unadjusted)?", ["avatar", "gone with the wind"]),
    ("misc", "What is the largest city in the world by population?", ["tokyo"]),
    ("misc", "How many time zones does Russia span?", ["11", "eleven"]),
    ("misc", "What is the national flower of Japan?", ["cherry blossom", "sakura"]),
]


def check(answer: str, fragments: list[str]) -> bool:
    a = answer.lower()
    return any(f.lower() in a for f in fragments)


def main() -> None:
    from sweep_neural_mesh.neurons.general_knowledge import GeneralKnowledge

    g = GeneralKnowledge(enable_live=True, enable_llm=False)
    results: dict[str, dict] = {}
    total_correct = 0
    total_time = 0.0
    total = len(QUESTIONS)
    t_start = time.time()

    print(f"Running {total} general-knowledge questions...\n")
    for i, (cat, q, frags) in enumerate(QUESTIONS, 1):
        t0 = time.time()
        ans = g.answer(q, timeout_live=4.0)
        dt = time.time() - t0
        total_time += dt
        ok = check(ans.answer, frags)
        if ok:
            total_correct += 1
        results.setdefault(cat, {"correct": 0, "total": 0})
        results[cat]["total"] += 1
        if ok:
            results[cat]["correct"] += 1
        mark = "[OK]" if ok else "[XX]"
        print(f"{mark} [{cat}] {q}")
        print(f"     -> {ans.answer!r} ({ans.method}, {dt:.1f}s)")


    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"TOTAL: {total_correct}/{total} = {total_correct/total*100:.1f}%")
    print(f"Wall time: {elapsed:.0f}s  (avg {total_time/total:.1f}s/question)")
    for cat in sorted(results):
        r = results[cat]
        print(f"  {cat:6s}: {r['correct']}/{r['total']} = {r['correct']/r['total']*100:.1f}%")
    print(f"Methods: {g.stats()}")
    print("=" * 60)


if __name__ == "__main__":
    main()