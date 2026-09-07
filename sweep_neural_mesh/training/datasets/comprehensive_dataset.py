"""
Comprehensive Training Dataset Generator — Sweep Full-System Training.

Generates datasets for all capability domains with proper train/val/test splits.
No test data is used for training. No leakage between splits.

Usage:
    python -m sweep_neural_mesh.training.datasets.comprehensive_dataset
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class DatasetSample:
    """A single training/evaluation sample."""
    id: str
    domain: str
    task: str
    input_text: str
    evidence: list[str] = field(default_factory=list)
    expected_output: str = ""
    expected_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str = "train"  # train, val, test
    difficulty: str = "medium"  # easy, medium, hard


def _hash_sample(text: str, domain: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.md5(f"{domain}:{text}".encode()).hexdigest()[:12]


# ════════════════════════════════════════════════════════════════
# DOMAIN 1: INTENT RECOGNITION
# ════════════════════════════════════════════════════════════════

INTENT_CATEGORIES = {
    "investigation": [
        "Investigate {person} and their connections",
        "Find everything about {person}'s background",
        "Who is {person} and what do they do?",
        "Research {person}'s professional history",
        "Look into {person}'s public records",
        "Investigate the relationship between {entity_a} and {entity_b}",
        "What is the connection between {org_a} and {org_b}?",
        "Trace the ownership of {entity}",
        "Follow the money from {source} to {dest}",
        "Map the network around {person}",
    ],
    "search": [
        "Search for information about {topic}",
        "Find articles about {topic}",
        "What's the latest news on {topic}?",
        "Look up {topic} on the web",
        "Research {topic} across multiple sources",
        "Find academic papers about {topic}",
        "Search for {entity} in news archives",
        "What do sources say about {topic}?",
        "Find reports about {event}",
        "Search for evidence of {claim}",
    ],
    "identity_analysis": [
        "Is {person_a} the same person as {person_b}?",
        "Could {entity} be an alias for {person}?",
        "Analyze the identity of {person}",
        "Verify if {person} matches {description}",
        "Are these two profiles about the same person?",
        "Cross-reference {person}'s identity across platforms",
        "Does {person} match the description of {person_description}?",
        "Is there evidence that {person} uses the alias {alias}?",
    ],
    "evidence_analysis": [
        "What evidence supports {claim}?",
        "Is there proof that {claim}?",
        "Evaluate the evidence for {claim}",
        "What do the sources say about {claim}?",
        "How strong is the evidence for {claim}?",
        "Are there any contradictions in the evidence about {claim}?",
        "What is the overall confidence in {claim}?",
        "Review the evidence and provide a confidence assessment",
    ],
    "comparison": [
        "Compare {entity_a} and {entity_b}",
        "What are the differences between {topic_a} and {topic_b}?",
        "How does {entity_a} differ from {entity_b}?",
        "Compare the evidence for {claim_a} vs {claim_b}",
        "What are the similarities between {topic_a} and {topic_b}?",
        "Which is more reliable: {source_a} or {source_b}?",
    ],
    "timeline": [
        "Create a timeline of {event}",
        "What happened first: {event_a} or {event_b}?",
        "When did {event} occur?",
        "Reconstruct the chronological order of {events}",
        "What events led to {event}?",
        "Build a timeline for {person}'s career",
    ],
    "relationship_analysis": [
        "What is the relationship between {entity_a} and {entity_b}?",
        "How are {entity_a} and {entity_b} connected?",
        "Map the relationships between {entities}",
        "What connections exist between {org} and {person}?",
        "Are {entity_a} and {entity_b} affiliated?",
    ],
    "location_analysis": [
        "Where is {entity} located?",
        "What is the geographic relationship between {loc_a} and {loc_b}?",
        "How far is {loc_a} from {loc_b}?",
        "Map all locations associated with {person}",
        "What locations are connected to {event}?",
    ],
    "media_analysis": [
        "Analyze this image",
        "What does this photo show?",
        "Extract text from this document",
        "What objects are in this image?",
        "Describe the scene in this video frame",
    ],
    "document_analysis": [
        "Analyze this document",
        "Extract key information from this PDF",
        "What does this report say about {topic}?",
        "Summarize this document",
        "Find contradictions in this document",
    ],
    "source_verification": [
        "Is this source reliable?",
        "Verify the credibility of {source}",
        "Is this information from {source} trustworthy?",
        "What is the reliability of {source}?",
        "Check if {source} is a primary or secondary source",
    ],
    "contradiction_analysis": [
        "Do these statements contradict each other?",
        "Is there a conflict between {claim_a} and {claim_b}?",
        "Are these sources consistent?",
        "Find contradictions in the evidence",
        "Do {source_a} and {source_b} disagree?",
    ],
    "summarization": [
        "Summarize the evidence about {topic}",
        "Give me a brief overview of {topic}",
        "What are the key findings about {topic}?",
        "Provide a summary of the investigation into {person}",
        "What are the main points about {topic}?",
    ],
    "classification": [
        "Classify this evidence as supporting or contradicting",
        "What type of source is this?",
        "Is this evidence reliable?",
        "Categorize this information",
        "Label this as primary or secondary evidence",
    ],
    "extraction": [
        "Extract all names mentioned in this text",
        "Find all dates in this document",
        "What organizations are mentioned?",
        "Extract all locations from this text",
        "Find all email addresses and phone numbers",
    ],
    "correlation": [
        "What correlations exist between {topics}?",
        "How do {factor_a} and {factor_b} relate?",
        "Find connections between these data points",
        "What patterns emerge from {data}?",
        "Identify correlations in the evidence",
    ],
    "unknown_ambiguous": [
        "Tell me something",
        "Help me",
        "What do you think?",
        "I'm curious about stuff",
        "Can you help with this thing?",
        "asdfghjkl",
        "12345",
        "test",
    ],
}

ENTITIES = {
    "person": ["John Smith", "Alice Chen", "Bob Wilson", "Sarah Davis", "Mike Johnson",
               "Emma Brown", "David Lee", "Lisa Wang", "James Taylor", "Maria Garcia"],
    "organization": ["TechCorp Inc", "Global Research Institute", "Pacific Analytics",
                     "Nexus Technologies", "Atlas Corporation", "Quantum Dynamics",
                     "Pacific States University", "Global Health Organization"],
    "location": ["New York", "London", "Tokyo", "Delhi", "Berlin", "Paris",
                 "San Francisco", "Sydney", "Toronto", "Mumbai"],
    "topic": ["quantum computing", "climate change", "artificial intelligence",
              "renewable energy", "cybersecurity", "machine learning",
              "genetic engineering", "space exploration", "economic policy",
              "public health"],
}


def _fill_intent_template(template: str) -> str:
    """Fill a template with random entities."""
    result = template
    for category, values in ENTITIES.items():
        placeholder = "{" + category + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values))
    # Handle compound placeholders
    if "{entity_a}" in result or "{entity_b}" in result:
        entities = random.sample(ENTITIES["person"] + ENTITIES["organization"], 2)
        result = result.replace("{entity_a}", entities[0]).replace("{entity_b}", entities[1])
    if "{org_a}" in result or "{org_b}" in result:
        orgs = random.sample(ENTITIES["organization"], 2)
        result = result.replace("{org_a}", orgs[0]).replace("{org_b}", orgs[1])
    if "{topic_a}" in result or "{topic_b}" in result:
        topics = random.sample(ENTITIES["topic"], 2)
        result = result.replace("{topic_a}", topics[0]).replace("{topic_b}", topics[1])
    if "{source_a}" in result or "{source_b}" in result:
        sources = random.sample(["Reuters", "BBC", "CNN", "Wikipedia", "arXiv", "official records"], 2)
        result = result.replace("{source_a}", sources[0]).replace("{source_b}", sources[1])
    if "{claim_a}" in result or "{claim_b}" in result:
        claims = random.sample(["exercise improves health", "vaccines are effective",
                                "climate change is accelerating", "AI will replace jobs"], 2)
        result = result.replace("{claim_a}", claims[0]).replace("{claim_b}", claims[1])
    if "{loc_a}" in result or "{loc_b}" in result:
        locs = random.sample(ENTITIES["location"], 2)
        result = result.replace("{loc_a}", locs[0]).replace("{loc_b}", locs[1])
    if "{person_a}" in result or "{person_b}" in result:
        people = random.sample(ENTITIES["person"], 2)
        result = result.replace("{person_a}", people[0]).replace("{person_b}", people[1])
    if "{person_description}" in result:
        result = result.replace("{person_description}", random.choice([
            "a software engineer from San Francisco",
            "a researcher at a university",
            "a business executive in London",
        ]))
    if "{alias}" in result:
        result = result.replace("{alias}", random.choice(["JohnS", "alice_c", "bobw_dev"]))
    if "{event}" in result:
        result = result.replace("{event}", random.choice([
            "the 2024 conference", "the merger announcement", "the product launch"
        ]))
    if "{events}" in result:
        result = result.replace("{events}", "the series of events in 2024")
    if "{event_a}" in result or "{event_b}" in result:
        events = random.sample(["the Moon landing", "WWII ending", "the French Revolution",
                                "the Internet invention", "the printing press"], 2)
        result = result.replace("{event_a}", events[0]).replace("{event_b}", events[1])
    if "{entities}" in result:
        people = random.sample(ENTITIES["person"], 3)
        result = result.replace("{entities}", ", ".join(people))
    if "{org}" in result:
        result = result.replace("{org}", random.choice(ENTITIES["organization"]))
    if "{data}" in result:
        result = result.replace("{data}", "the collected evidence")
    if "{topics}" in result:
        topics = random.sample(ENTITIES["topic"], 2)
        result = result.replace("{topics}", " and ".join(topics))
    if "{factor_a}" in result or "{factor_b}" in result:
        factors = random.sample(["temperature", "humidity", "pressure", "wind speed", "precipitation"], 2)
        result = result.replace("{factor_a}", factors[0]).replace("{factor_b}", factors[1])
    return result


def generate_intent_dataset(n_per_class: int = 30) -> list[DatasetSample]:
    """Generate intent recognition dataset."""
    samples = []
    for intent, templates in INTENT_CATEGORIES.items():
        for i in range(min(n_per_class, len(templates) * 3)):
            template = random.choice(templates)
            text = _fill_intent_template(template)
            sid = _hash_sample(text, "intent")
            samples.append(DatasetSample(
                id=sid, domain="intent", task="classify_intent",
                input_text=text, expected_label=intent,
                difficulty="easy" if i < n_per_class // 3 else "medium" if i < 2 * n_per_class // 3 else "hard",
            ))
    return samples


# ════════════════════════════════════════════════════════════════
# DOMAIN 2: ENTITY EXTRACTION
# ════════════════════════════════════════════════════════════════

ENTITY_SENTENCES = [
    ("John Smith works at TechCorp Inc in New York", {"PERSON": ["John Smith"], "ORG": ["TechCorp Inc"], "GPE": ["New York"]}),
    ("Alice Chen published a paper at MIT in Cambridge", {"PERSON": ["Alice Chen"], "ORG": ["MIT"], "GPE": ["Cambridge"]}),
    ("The conference was held in London on January 15, 2024", {"GPE": ["London"], "DATE": ["January 15, 2024"]}),
    ("Contact Bob Wilson at bob@example.com or call +1-555-123-4567", {"PERSON": ["Bob Wilson"], "EMAIL": ["bob@example.com"], "PHONE": ["+1-555-123-4567"]}),
    ("TechCorp Inc announced quarterly earnings on March 3, 2024", {"ORG": ["TechCorp Inc"], "DATE": ["March 3, 2024"]}),
    ("Sarah Davis from Google visited the Tokyo office", {"PERSON": ["Sarah Davis"], "ORG": ["Google"], "GPE": ["Tokyo"]}),
    ("The study was conducted by researchers at Stanford University", {"ORG": ["Stanford University"]}),
    ("Mike Johnson joined Amazon in Seattle in 2023", {"PERSON": ["Mike Johnson"], "ORG": ["Amazon"], "GPE": ["Seattle"], "DATE": ["2023"]}),
    ("Emma Brown works at the United Nations in Geneva", {"PERSON": ["Emma Brown"], "ORG": ["United Nations"], "GPE": ["Geneva"]}),
    ("The WHO report from April 2024 confirmed the findings", {"ORG": ["WHO"], "DATE": ["April 2024"]}),
    ("David Lee and Lisa Wang co-founded Nexus Technologies", {"PERSON": ["David Lee", "Lisa Wang"], "ORG": ["Nexus Technologies"]}),
    ("The product launched in Berlin and Munich simultaneously", {"GPE": ["Berlin", "Munich"]}),
    ("James Taylor of Microsoft spoke at the AI conference in San Francisco", {"PERSON": ["James Taylor"], "ORG": ["Microsoft"], "GPE": ["San Francisco"]}),
    ("Maria Garcia from the University of Barcelona published research on CRISPR", {"PERSON": ["Maria Garcia"], "ORG": ["University of Barcelona"]}),
    ("The EU regulation was published on December 1, 2024 in Brussels", {"ORG": ["EU"], "DATE": ["December 1, 2024"], "GPE": ["Brussels"]}),
    ("Contact information: admin@gov.uk, phone: +44 20 7925 0918", {"EMAIL": ["admin@gov.uk"], "PHONE": ["+44 20 7925 0918"]}),
    ("The Tokyo Stock Exchange reported gains on January 10", {"GPE": ["Tokyo"], "DATE": ["January 10"]}),
    ("Dr. Smith at Johns Hopkins University led the clinical trial", {"PERSON": ["Dr. Smith"], "ORG": ["Johns Hopkins University"]}),
    ("The merger between Delta Airlines and Northwest was completed in 2010", {"ORG": ["Delta Airlines", "Northwest"], "DATE": ["2010"]}),
    ("Beijing and Shanghai are the two largest cities in China", {"GPE": ["Beijing", "Shanghai", "China"]}),
]


def generate_entity_dataset(n_samples: int = 300) -> list[DatasetSample]:
    """Generate entity extraction dataset."""
    samples = []
    for i in range(n_samples):
        sentence, entities = random.choice(ENTITY_SENTENCES)
        # Add some variation
        if random.random() < 0.3:
            sentence = sentence.lower()
        if random.random() < 0.2:
            words = sentence.split()
            if len(words) > 5:
                idx = random.randint(2, len(words) - 3)
                words[idx] = words[idx][:max(1, len(words[idx]) - 1)]  # Simulate misspelling
                sentence = " ".join(words)
        sid = _hash_sample(sentence + str(i), "entity")
        samples.append(DatasetSample(
            id=sid, domain="entity", task="extract_entities",
            input_text=sentence,
            expected_output=json.dumps(entities),
            metadata={"entities": entities},
            difficulty="easy" if len(entities) <= 2 else "medium" if len(entities) <= 4 else "hard",
        ))
    return samples


# ════════════════════════════════════════════════════════════════
# DOMAIN 3: EVIDENCE CLASSIFICATION
# ════════════════════════════════════════════════════════════════

EVIDENCE_SAMPLES = {
    "supports": [
        ("Studies confirm that regular exercise reduces cardiovascular disease risk by 30%", "exercise improves health"),
        ("Meta-analysis of 50 trials shows the vaccine is 95% effective", "vaccines are safe and effective"),
        ("Peer-reviewed research demonstrates that reading improves cognitive function", "reading helps learning"),
        ("Longitudinal data shows the treatment group outperformed controls significantly", "the treatment is beneficial"),
        ("Multiple independent studies confirm the relationship between sleep and performance", "sleep improves performance"),
        ("Clinical trial results show statistically significant improvement in the treatment group", "the drug works"),
        ("Expert consensus across 12 institutions supports this conclusion", "the conclusion is well-supported"),
        ("The evidence from 3 independent research teams all points to the same result", "the finding is robust"),
        ("Government data confirms the economic growth trend over the past 5 years", "the economy is growing"),
        ("Systematic review of 200 studies confirms the safety profile", "the medication is safe"),
        ("A peer-reviewed study in Nature demonstrated the mechanism clearly", "the mechanism is understood"),
        ("Federal health agencies officially recommend this approach", "the approach is recommended"),
        ("The FDA approved the drug after rigorous testing phases", "the drug passed FDA approval"),
        ("University researchers replicated the original findings across 3 labs", "the results are reproducible"),
        ("The WHO guidelines explicitly support this intervention", "the WHO supports this"),
    ],
    "refutes": [
        ("The drug showed no significant effect compared to placebo in the trial", "the drug is effective"),
        ("Studies found no evidence supporting the claimed health benefits", "the supplement helps"),
        ("The experiment failed to demonstrate any measurable improvement", "the treatment works"),
        ("Research contradicts the widely held belief about this topic", "the belief is correct"),
        ("The data shows the intervention had zero measurable impact", "the intervention is useful"),
        ("Meta-analysis found no statistically significant benefit across all 15 studies", "the therapy helps"),
        ("Controlled trials showed the treatment group performed worse than expected", "the treatment is beneficial"),
        ("The hypothesis was definitively refuted by the experimental evidence", "the hypothesis is true"),
        ("Independent verification found the original results could not be replicated", "the original study is reliable"),
        ("The government report found the policy had no positive effect", "the policy works"),
        ("Results were not statistically significant in any of the 8 sub-studies", "the effect is significant"),
        ("The study found the opposite of what was hypothesized", "the hypothesis is supported"),
        ("Three separate labs failed to reproduce the original findings", "the findings are reproducible"),
        ("The WHO stated the evidence does not support this treatment", "the WHO endorses this treatment"),
        ("FDA analysis found the drug's benefits did not outweigh its risks", "the drug is safe"),
    ],
    "neutral": [
        ("Results were mixed across different populations and demographics", "the treatment works universally"),
        ("Some participants improved while others showed no change", "the treatment helps everyone"),
        ("The effect size was small and borderline significant", "the effect is clearly significant"),
        ("More research is needed to confirm these preliminary findings", "the findings are conclusive"),
        ("Evidence is insufficient to draw a definitive conclusion", "we can conclude definitively"),
        ("The results were inconsistent across different study designs", "the results are consistent"),
        ("The study had important limitations that affect interpretation", "the study is conclusive"),
        ("Additional large-scale trials are warranted before conclusions", "the evidence is sufficient"),
        ("The effect was observed only in certain subgroups", "the effect is universal"),
        ("Preliminary data suggests a trend but lacks statistical power", "the trend is confirmed"),
        ("The findings depend heavily on the specific methodology used", "the findings are methodology-independent"),
        ("Results varied considerably depending on dosage and duration", "the results are consistent across conditions"),
        ("The study population may not be representative of the general public", "the study is representative"),
        ("Some evidence supports while other evidence contradicts", "the evidence is clear"),
        ("The relationship appears complex and not fully understood yet", "the relationship is simple"),
    ],
}


def generate_evidence_dataset(n_per_class: int = 60) -> list[DatasetSample]:
    """Generate evidence classification dataset."""
    samples = []
    for label, pairs in EVIDENCE_SAMPLES.items():
        for i in range(n_per_class):
            evidence_text, claim = random.choice(pairs)
            # Add variation
            if random.random() < 0.3:
                evidence_text = evidence_text.lower()
            if random.random() < 0.2:
                # Add some noise
                evidence_text = f"According to recent reports, {evidence_text.lower()}"
            sid = _hash_sample(evidence_text + str(i), "evidence")
            samples.append(DatasetSample(
                id=sid, domain="evidence", task="classify_evidence",
                input_text=f"Claim: {claim}\nEvidence: {evidence_text}",
                evidence=[evidence_text],
                expected_label=label,
                difficulty="easy" if i < n_per_class // 3 else "medium" if i < 2 * n_per_class // 3 else "hard",
            ))
    return samples


# ════════════════════════════════════════════════════════════════
# DOMAIN 4: CONTRADICTION DETECTION
# ════════════════════════════════════════════════════════════════

CONTRADICTION_PAIRS = [
    # Direct contradictions
    ("The meeting is at 3 PM", "The meeting is at 4 PM", "contradiction", "time"),
    ("The drug is effective", "The drug is ineffective", "contradiction", "efficacy"),
    ("All students passed the exam", "Some students failed the exam", "contradiction", "scope"),
    ("It is raining outside", "It is sunny and completely dry", "contradiction", "weather"),
    ("Revenue increased 15%", "Revenue decreased 15%", "contradiction", "direction"),
    ("The company is profitable", "The company reported significant losses", "contradiction", "financial"),
    ("Water boils at 100C", "Water boils at 90C at sea level", "contradiction", "numerical"),
    ("The Earth is approximately round", "The Earth is completely flat", "contradiction", "factual"),
    ("Light travels faster than sound", "Sound travels faster than light", "contradiction", "factual"),
    ("Cats are mammals", "Cats are reptiles", "contradiction", "classification"),
    ("The product costs $50", "The product costs $500", "contradiction", "price"),
    ("The event is on Monday", "The event is on Tuesday", "contradiction", "time"),
    ("Paris is the capital of France", "Lyon is the capital of France", "contradiction", "factual"),
    ("The population is 1 million", "The population is 10 million", "contradiction", "numerical"),
    ("The study supports the hypothesis", "The study contradicts the hypothesis", "contradiction", "conclusion"),
    
    # Consistent pairs
    ("The study found exercise improves health", "Research confirms physical activity benefits cardiovascular health", "consistent", "health"),
    ("Water boils at 100C at sea level", "The boiling point of water is 100 degrees Celsius", "consistent", "science"),
    ("Paris is the capital of France", "France's capital city is Paris", "consistent", "geography"),
    ("The experiment showed positive results", "The trial demonstrated beneficial outcomes", "consistent", "research"),
    ("The company reported growth", "The organization announced expansion", "consistent", "business"),
    ("The medication reduces symptoms", "The drug alleviates clinical symptoms", "consistent", "medical"),
    ("Climate change is accelerating", "Global warming trends are intensifying", "consistent", "science"),
    ("The algorithm runs in O(n log n)", "The algorithm has logarithmic linear time complexity", "consistent", "computer_science"),
    ("The bridge is 500 meters long", "The 500m bridge spans the river", "consistent", "engineering"),
    ("Python is a programming language", "Python is used for software development", "consistent", "technology"),
    
    # Partial contradictions (hard negatives)
    ("The drug works for adults", "The drug works for children", "partial", "population"),
    ("Revenue increased overall", "Revenue decreased in Q4", "partial", "temporal"),
    ("The study found a positive correlation", "The effect size was very small", "partial", "significance"),
    ("The product is affordable", "The product costs $10,000", "partial", "subjectivity"),
    ("The company is growing", "The company lost market share in Europe", "partial", "scope"),
]


def generate_contradiction_dataset(n_pairs: int = 200) -> list[DatasetSample]:
    """Generate contradiction detection dataset."""
    samples = []
    for i in range(n_pairs):
        text_a, text_b, label, conflict_type = random.choice(CONTRADICTION_PAIRS)
        # Add hard negative variations
        if random.random() < 0.2:
            # Swap the order
            text_a, text_b = text_b, text_a
        if random.random() < 0.15:
            # Add qualifiers
            text_a = f"According to the report, {text_a.lower()}"
        
        sid = _hash_sample(f"{text_a}|{text_b}|{i}", "contradiction")
        samples.append(DatasetSample(
            id=sid, domain="contradiction", task="detect_contradiction",
            input_text=f"Statement A: {text_a}\nStatement B: {text_b}",
            evidence=[text_a, text_b],
            expected_label=label,
            metadata={"conflict_type": conflict_type},
            difficulty="easy" if label == "contradiction" and conflict_type in ("factual", "numerical") else
                      "hard" if label == "partial" else "medium",
        ))
    return samples


# ════════════════════════════════════════════════════════════════
# DOMAIN 5: LOGICAL REASONING
# ════════════════════════════════════════════════════════════════

LOGICAL_SAMPLES = [
    # Modus ponens
    ("If it rains, the ground gets wet. It is raining. Is the ground wet?", ["If it rains, the ground gets wet.", "It is raining."], "yes", "modus_ponens"),
    ("If A then B. A is true. Is B true?", ["If A then B.", "A is true."], "yes", "modus_ponens"),
    ("If the alarm sounds, the building is evacuated. The alarm is sounding. Is the building evacuated?", 
     ["If the alarm sounds, the building is evacuated.", "The alarm is sounding."], "yes", "modus_ponens"),
    
    # Modus tollens
    ("If P then Q. Not Q. Is P true?", ["If P then Q.", "Not Q."], "no", "modus_tollens"),
    ("If it rains, the ground is wet. The ground is not wet. Did it rain?", 
     ["If it rains, the ground is wet.", "The ground is not wet."], "no", "modus_tollens"),
    ("If the button is pressed, the machine starts. The machine did not start. Was the button pressed?",
     ["If the button is pressed, the machine starts.", "The machine did not start."], "no", "modus_tollens"),
    
    # Transitivity
    ("If A > B and B > C, is A > C?", ["A > B", "B > C"], "yes", "transitivity"),
    ("Alice is taller than Bob. Bob is taller than Charlie. Is Alice taller than Charlie?",
     ["Alice is taller than Bob.", "Bob is taller than Charlie."], "yes", "transitivity"),
    ("X is faster than Y. Y is faster than Z. Is X faster than Z?",
     ["X is faster than Y.", "Y is faster than Z."], "yes", "transitivity"),
    
    # Syllogism
    ("All cats are animals. All animals are living things. Is a cat a living thing?",
     ["All cats are animals.", "All animals are living things."], "yes", "syllogism"),
    ("No reptiles produce milk. A snake is a reptile. Does a snake produce milk?",
     ["No reptiles produce milk.", "A snake is a reptile."], "no", "syllogism"),
    ("All birds can fly. A penguin is a bird. Can a penguin fly?",
     ["All birds can fly.", "A penguin is a bird."], "yes", "syllogism"),
    
    # Negation
    ("It is not the case that all students passed. Some students failed. Is this consistent?",
     ["It is not the case that all students passed.", "Some students failed."], "yes", "negation"),
    ("The statement is false. Therefore the statement is true. Is this a paradox?",
     ["The statement is false."], "paradox", "paradox_detection"),
    
    # Conditional chains
    ("If A then B. If B then C. Does A imply C?",
     ["If A then B.", "If B then C."], "yes", "conditional_chain"),
    ("If it rains, the ground is wet. If the ground is wet, the grass grows. If it rains, does the grass grow?",
     ["If it rains, the ground is wet.", "If the ground is wet, the grass grows."], "yes", "conditional_chain"),
    
    # Hard negatives
    ("If it rains, the ground is wet. The ground is wet. Did it rain?",
     ["If it rains, the ground is wet.", "The ground is wet."], "unknown", "fallacy_denial_antecedent"),
    ("If A then B. B is true. Is A true?",
     ["If A then B.", "B is true."], "unknown", "fallacy_affirming_consequent"),
    ("All cats are animals. Some animals are dogs. Are all cats dogs?",
     ["All cats are animals.", "Some animals are dogs."], "no", "fallacy_undistributed_middle"),
]


def generate_logic_dataset(n_samples: int = 300) -> list[DatasetSample]:
    """Generate logical reasoning dataset."""
    samples = []
    for i in range(n_samples):
        query, evidence, expected, reasoning_type = random.choice(LOGICAL_SAMPLES)
        # Add variation
        if random.random() < 0.2:
            # Paraphrase slightly
            query = query.replace("Is ", "Can you determine if ").replace("?", "?")
        
        sid = _hash_sample(f"{query}|{i}", "logic")
        samples.append(DatasetSample(
            id=sid, domain="logic", task="logical_reasoning",
            input_text=query,
            evidence=evidence,
            expected_label=expected,
            metadata={"reasoning_type": reasoning_type},
            difficulty="easy" if reasoning_type in ("modus_ponens", "modus_tollens") else
                      "hard" if reasoning_type.startswith("fallacy") else "medium",
        ))
    return samples


# ════════════════════════════════════════════════════════════════
# DOMAIN 6: TEMPORAL REASONING
# ════════════════════════════════════════════════════════════════

TEMPORAL_SAMPLES = [
    # Ordering
    ("What happened first: the Moon landing or WWII?", "WWII", "ordering"),
    ("Which came first: the printing press or the Internet?", "printing press", "ordering"),
    ("Did the French Revolution happen before or after the American Revolution?", "before", "ordering"),
    ("What is the chronological order: dinosaurs, humans, fish?", "fish, dinosaurs, humans", "ordering"),
    ("Which happened first: the fall of Rome or the Renaissance?", "fall of Rome", "ordering"),
    
    # Date extraction
    ("When did WWII end?", "1945", "date_extraction"),
    ("What year was the Declaration of Independence signed?", "1776", "date_extraction"),
    ("In what year did the first Moon landing occur?", "1969", "date_extraction"),
    ("When was the printing press invented?", "1440", "date_extraction"),
    ("What year did the Berlin Wall fall?", "1989", "date_extraction"),
    
    # Timeline construction
    ("Create a timeline: WWII ended in 1945, Moon landing was in 1969, Internet invented in 1969",
     "1945, 1969, 1969", "timeline"),
    ("Events: Company founded in 2010, IPO in 2015, merger in 2020. What is the order?",
     "2010, 2015, 2020", "timeline"),
    
    # Temporal conflicts
    ("Source A says the event was on Monday. Source B says it was on Tuesday. Is there a conflict?",
     "conflict", "conflict_detection"),
    ("The document is dated 2020 but references events from 2025. Is this consistent?",
     "inconsistent", "conflict_detection"),
]


def generate_temporal_dataset(n_samples: int = 150) -> list[DatasetSample]:
    """Generate temporal reasoning dataset."""
    samples = []
    for i in range(n_samples):
        query, expected, task_type = random.choice(TEMPORAL_SAMPLES)
        sid = _hash_sample(f"{query}|{i}", "temporal")
        samples.append(DatasetSample(
            id=sid, domain="temporal", task="temporal_reasoning",
            input_text=query,
            expected_label=expected,
            metadata={"task_type": task_type},
            difficulty="easy" if task_type == "date_extraction" else
                      "hard" if task_type == "conflict_detection" else "medium",
        ))
    return samples


# ════════════════════════════════════════════════════════════════
# DOMAIN 7: SOURCE INDEPENDENCE
# ════════════════════════════════════════════════════════════════

SOURCE_INDEPENDENCE_SAMPLES = [
    # Copied articles (should NOT count as independent)
    {
        "sources": [
            {"name": "Press Release X", "content": "Company announces new product with revolutionary features.", "type": "press_release"},
            {"name": "Article A", "content": "Company announces new product with revolutionary features.", "type": "news"},
            {"name": "Article B", "content": "Company's new product with revolutionary features was announced.", "type": "news"},
        ],
        "expected_independent": 1,
        "description": "Press release copied by news outlets",
    },
    # Independent sources (SHOULD count as independent)
    {
        "sources": [
            {"name": "Government Report", "content": "Official statistics show 5% economic growth.", "type": "government"},
            {"name": "University Study", "content": "Academic analysis confirms growth trends using different methodology.", "type": "academic"},
            {"name": "News Analysis", "content": "Independent journalism reports on economic indicators.", "type": "news"},
        ],
        "expected_independent": 3,
        "description": "Three independent sources",
    },
    # Mixed independence
    {
        "sources": [
            {"name": "Wire Service", "content": "Breaking: Company reports record earnings.", "type": "wire_service"},
            {"name": "NY Times", "content": "Company reports record earnings, stock rises.", "type": "news_major"},
            {"name": "CNN", "content": "Company announces record-breaking quarterly earnings.", "type": "news_major"},
            {"name": "Blogger", "content": "My analysis of the company's earnings report.", "type": "blog"},
        ],
        "expected_independent": 3,
        "description": "Wire story syndicated to major outlets plus independent blog",
    },
    # Syndicated content
    {
        "sources": [
            {"name": "AP Report", "content": "Scientists discover high levels of microplastics in ocean water samples.", "type": "wire_service"},
            {"name": "Local News 1", "content": "Scientists discover high levels of microplastics in ocean water samples.", "type": "news_minor"},
            {"name": "Local News 2", "content": "AP: Scientists discover high levels of microplastics in ocean water samples.", "type": "news_minor"},
            {"name": "Science Daily", "content": "New study reveals alarming microplastic contamination in oceans.", "type": "academic"},
        ],
        "expected_independent": 2,
        "description": "AP wire story syndicated to local outlets",
    },
]


def generate_source_independence_dataset(n_samples: int = 100) -> list[DatasetSample]:
    """Generate source independence dataset."""
    samples = []
    for i in range(n_samples):
        scenario = random.choice(SOURCE_INDEPENDENCE_SAMPLES)
        sources_text = "\n".join([
            f"Source: {s['name']} ({s['type']})\nContent: {s['content']}"
            for s in scenario["sources"]
        ])
        sid = _hash_sample(f"si_{i}", "source_independence")
        samples.append(DatasetSample(
            id=sid, domain="source_independence", task="analyze_independence",
            input_text=sources_text,
            evidence=[s["content"] for s in scenario["sources"]],
            expected_label=str(scenario["expected_independent"]),
            metadata={
                "source_types": [s["type"] for s in scenario["sources"]],
                "description": scenario["description"],
            },
            difficulty="easy" if scenario["expected_independent"] == len(scenario["sources"]) else "hard",
        ))
    return samples


# ════════════════════════════════════════════════════════════════
# MAIN: GENERATE ALL DATASETS
# ════════════════════════════════════════════════════════════════

def deduplicate_samples(samples: list[DatasetSample]) -> list[DatasetSample]:
    """Remove duplicate samples by input_text, keeping first occurrence."""
    seen: set[str] = set()
    unique = []
    for s in samples:
        key = s.input_text.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def split_dataset(samples: list[DatasetSample], train_ratio: float = 0.8, val_ratio: float = 0.1, seed: int = 42) -> list[DatasetSample]:
    """Split dataset into train/val/test with stratification by domain and difficulty.
    
    CRITICAL: Deduplicates by input_text BEFORE splitting to prevent leakage.
    """
    random.seed(seed)
    
    # STEP 1: Deduplicate first
    samples = deduplicate_samples(samples)
    
    # Group by domain+difficulty for stratification
    groups: dict[str, list[DatasetSample]] = {}
    for s in samples:
        key = f"{s.domain}_{s.difficulty}"
        groups.setdefault(key, []).append(s)
    
    result = []
    for group_samples in groups.values():
        random.shuffle(group_samples)
        n = len(group_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        for i, s in enumerate(group_samples):
            if i < n_train:
                s.split = "train"
            elif i < n_train + n_val:
                s.split = "val"
            else:
                s.split = "test"
            result.append(s)
    
    return result


def check_leakage(samples: list[DatasetSample]) -> dict[str, Any]:
    """Check for data leakage between splits."""
    train_texts = {s.input_text for s in samples if s.split == "train"}
    val_texts = {s.input_text for s in samples if s.split == "val"}
    test_texts = {s.input_text for s in samples if s.split == "test"}
    
    # Check exact duplicates
    train_val_overlap = train_texts & val_texts
    train_test_overlap = train_texts & test_texts
    val_test_overlap = val_texts & test_texts
    
    # Check near duplicates (first 50 chars)
    train_prefixes = {s.input_text[:50].lower().strip() for s in samples if s.split == "train"}
    val_prefixes = {s.input_text[:50].lower().strip() for s in samples if s.split == "val"}
    test_prefixes = {s.input_text[:50].lower().strip() for s in samples if s.split == "test"}
    
    near_overlap_train_val = train_prefixes & val_prefixes
    near_overlap_train_test = train_prefixes & test_prefixes
    
    return {
        "exact_overlap_train_val": len(train_val_overlap),
        "exact_overlap_train_test": len(train_test_overlap),
        "exact_overlap_val_test": len(val_test_overlap),
        "near_overlap_train_val": len(near_overlap_train_val),
        "near_overlap_train_test": len(near_overlap_train_test),
        "has_leakage": len(train_val_overlap) > 0 or len(train_test_overlap) > 0,
    }


def main():
    """Generate all datasets and save them."""
    print("=" * 70)
    print("SWEEP COMPREHENSIVE DATASET GENERATION")
    print("=" * 70)
    
    random.seed(42)
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # Generate each domain
    domains = [
        ("Intent Recognition", generate_intent_dataset, 80),
        ("Entity Extraction", generate_entity_dataset, 400),
        ("Evidence Classification", generate_evidence_dataset, 150),
        ("Contradiction Detection", generate_contradiction_dataset, 300),
        ("Logical Reasoning", generate_logic_dataset, 400),
        ("Temporal Reasoning", generate_temporal_dataset, 250),
        ("Source Independence", generate_source_independence_dataset, 150),
    ]
    
    for name, generator, *args in domains:
        samples = generator(*args)
        all_samples.extend(samples)
        print(f"  {name}: {len(samples)} samples")
    
    # Split
    all_samples = split_dataset(all_samples)
    
    # Count splits
    train = [s for s in all_samples if s.split == "train"]
    val = [s for s in all_samples if s.split == "val"]
    test = [s for s in all_samples if s.split == "test"]
    
    print(f"\n  Total: {len(all_samples)} samples")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    
    # Leakage check
    leakage = check_leakage(all_samples)
    print(f"\n  Leakage check:")
    print(f"    Exact overlap (train/val): {leakage['exact_overlap_train_val']}")
    print(f"    Exact overlap (train/test): {leakage['exact_overlap_train_test']}")
    print(f"    Near overlap (train/val): {leakage['near_overlap_train_val']}")
    print(f"    Near overlap (train/test): {leakage['near_overlap_train_test']}")
    print(f"    Has leakage: {leakage['has_leakage']}")
    
    # Save datasets
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        output_path = output_dir / f"comprehensive_{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in split_data:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
        print(f"  Saved {split_name} to {output_path}")
    
    # Save full dataset
    full_path = output_dir / "comprehensive_all.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    print(f"  Saved full dataset to {full_path}")
    
    # Save metadata
    metadata = {
        "total_samples": len(all_samples),
        "train_samples": len(train),
        "val_samples": len(val),
        "test_samples": len(test),
        "domains": {},
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "leakage_check": leakage,
        "seed": 42,
        "generation_time": time.time(),
    }
    for domain in set(s.domain for s in all_samples):
        domain_samples = [s for s in all_samples if s.domain == domain]
        metadata["domains"][domain] = {
            "total": len(domain_samples),
            "train": len([s for s in domain_samples if s.split == "train"]),
            "val": len([s for s in domain_samples if s.split == "val"]),
            "test": len([s for s in domain_samples if s.split == "test"]),
        }
    
    meta_path = output_dir / "dataset_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {meta_path}")
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    
    return metadata


if __name__ == "__main__":
    main()
