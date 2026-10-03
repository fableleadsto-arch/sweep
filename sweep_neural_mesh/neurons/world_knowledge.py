"""
World Knowledge Base -- factual entity properties for evidence verification.

Without this, Sweep trusts evidence like "Birds can talk" at 0.95 confidence
because it contains the words "according to" and "biological classification".

This module provides a factual grounding layer: known properties of real
entities that evidence can be checked against.

Architecture:

    ┌──────────────────────────────────────────────────┐
    │           WORLD KNOWLEDGE BASE                    │
    │                                                    │
    │  Entity -> { properties, category, abilities }     │
    │  Relation -> { subject, predicate, object }        │
    │  FactCheck -> { plausible, confidence, reason }    │
    └──────────────────────────────────────────────────┘
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A known entity with its properties."""
    name: str
    category: str               # animal, plant, object, concept, place, food, body_part, material
    properties: dict[str, Any]  # key-value pairs of known properties
    abilities: list[str]        # things this entity can do
    NOT_abilities: list[str]    # things this entity definitely CANNOT do


@dataclass
class Relation:
    """A known relation between two entities."""
    subject: str
    predicate: str
    obj: str
    confidence: float


@dataclass
class FactCheck:
    """Result of checking a claim against world knowledge."""
    claim: str
    plausible: bool
    confidence: float           # 0.0-1.0 how sure we are
    matching_entities: list[str]
    contradictions: list[str]
    reasoning: str


class WorldKnowledge:
    """
    A factual knowledge base about entities in the world.

    This is NOT common sense (which is about default behaviors).
    This is FACTUAL knowledge: birds don't talk, buildings aren't alive,
    cats can't fly, mountains don't conduct electricity.

    Used as a gate in evidence gathering: if evidence claims something
    that contradicts known facts, the evidence is flagged as suspicious
    and its direction is downweighted.
    """

    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._init_entities()
        self._init_relations()

    def _init_entities(self) -> None:
        """Initialize the knowledge base with known entities."""

        def add(name: str, cat: str, props: dict, abilities: list, not_abilities: list):
            self._entities[name.lower()] = Entity(name, cat, props, abilities, not_abilities)

        # ── ANIMALS ──
        add("bird", "animal",
            {"legs": 2, "has_wings": True, "has_feathers": True, "lays_eggs": True, "size": "small-medium"},
            ["fly", "sing", "lay eggs", "build nests", "perch", "conduct electricity (biological tissue)"],
            ["talk", "speak human language", "drive", "use tools", "swim underwater"])

        add("cat", "animal",
            {"legs": 4, "has_fur": True, "carnivore": True, "size": "small", "domestic": True},
            ["purr", "climb", "jump", "catch mice", "sleep 16 hours", "conduct electricity (biological tissue)"],
            ["fly", "talk", "swim well", "drive", "bark", "made of metal", "made of wood"])

        add("dog", "animal",
            {"legs": 4, "has_fur": True, "omnivore": True, "size": "medium", "domestic": True},
            ["bark", "run", "fetch", "swim", "guard", "conduct electricity (biological tissue)"],
            ["fly", "talk human language", "drive", "use hands", "made of metal", "made of wood"])

        add("fish", "animal",
            {"legs": 0, "has_fins": True, "has_gills": True, "lives_in": "water", "size": "varies"},
            ["swim", "breathe underwater", "lay eggs", "conduct electricity (biological tissue)"],
            ["fly", "walk on land", "talk", "live on land permanently", "made of metal", "made of wood"])

        add("shark", "animal",
            {"legs": 0, "has_fins": True, "lives_in": "ocean", "size": "large", "predator": True},
            ["swim", "bite", "detect electromagnetic fields"],
            ["fly", "walk on land", "live in freshwater"])

        add("whale", "animal",
            {"legs": 0, "lives_in": "ocean", "size": "very large", "mammal": True, "warm_blooded": True},
            ["swim", "dive deep", "sing songs", "breach"],
            ["fly", "walk on land", "live in rivers", "breathe underwater (uses lungs)"])

        add("dolphin", "animal",
            {"legs": 0, "lives_in": "ocean", "size": "medium", "mammal": True},
            ["swim", "jump", "echolocate", "live in pods"],
            ["walk on land", "breathe underwater", "fly"])

        add("panda", "animal",
            {"legs": 4, "has_fur": True, "size": "large", "eats_bamboo": True, "mammal": True},
            ["climb", "eat bamboo", "roll", "swim"],
            ["fly", "talk", "bark"])

        add("goldfish", "animal",
            {"legs": 0, "has_fins": True, "size": "tiny", "lives_in": "water", "memory_months": True},
            ["swim", "recognize owners", "remember for months"],
            ["fly", "walk", "talk"])

        add("eagle", "animal",
            {"legs": 2, "has_wings": True, "size": "medium", "predator": True, "bird_of_prey": True},
            ["fly", "dive at high speed", "carry prey", "nest on cliffs"],
            ["talk", "swim underwater", "walk on hands"])

        add("chicken", "animal",
            {"legs": 2, "has_wings": True, "size": "small", "domestic": True},
            ["fly short distances", "cluck", "lay eggs", "scratch ground"],
            ["fly long distances", "talk", "swim", "drive"])

        add("snake", "animal",
            {"legs": 0, "has_scales": True, "size": "varies", "reptile": True},
            ["slither", "coil", "shed skin", "sense heat"],
            ["fly", "walk", "talk", "bark"])

        add("frog", "animal",
            {"legs": 4, "size": "small", "amphibian": True},
            ["hop", "croak", "catch insects", "live on land and water"],
            ["fly", "talk", "bark", "slither"])

        add("bear", "animal",
            {"legs": 4, "has_fur": True, "size": "large", "omnivore": True},
            ["climb", "swim", "fish", "hibernate", "stand on hind legs"],
            ["fly", "talk", "drive"])

        add("lion", "animal",
            {"legs": 4, "has_fur": True, "size": "large", "predator": True},
            ["roar", "hunt", "run fast", "climb"],
            ["fly", "swim well", "talk", "bark"])

        add("cow", "animal",
            {"legs": 4, "herbivore": True, "size": "large", "domestic": True, "produces_milk": True},
            ["moo", "graze", "produce milk", "chew cud"],
            ["fly", "talk", "bark", "meow"])

        add("reptile", "animal",
            {"legs": "varies", "cold_blooded": True, "lays_eggs": True},
            ["regulate body temperature", "shed skin"],
            ["produce milk", "nurse young"])

        add("human", "animal",
            {"legs": 2, "warm_blooded": True, "has_hair": True, "large_brain": True, "omnivore": True},
            ["talk", "walk", "run", "think", "use tools", "read", "write", "conduct electricity (biological tissue)"],
            ["fly", "swim well without aid", "breathe underwater", "made of metal"])

        add("mammal", "animal",
            {"warm_blooded": True, "has_hair_or_fur": True, "produces_milk": True},
            ["nurse young", "regulate body temperature"],
            ["breathe underwater (most)", "lay eggs (most)"])

        # ── PLANTS ──
        add("plant", "plant",
            {"photosynthesizes": True, "has_roots": True, "immobile": True},
            ["grow", "photosynthesize", "reproduce", "respond to light", "conduct electricity (biological tissue)"],
            ["move", "walk", "see", "hear", "fly", "talk", "swim"])

        # ── OBJECTS ──
        add("building", "object",
            {"material": "varies", "large": True, "immobile": True, "man_made": True},
            ["house people", "provide shelter"],
            ["move", "walk", "fly", "talk", "think", "be alive", "breathe"])

        add("car", "object",
            {"material": "metal", "man_made": True, "size": "medium", "has_wheels": True},
            ["transport people", "drive", "go fast"],
            ["fly", "swim", "talk", "think", "be alive", "grow"])

        add("computer", "object",
            {"material": "metal/plastic", "man_made": True, "electronic": True},
            ["compute", "store data", "process information"],
            ["move on its own", "fly", "swim", "be alive", "breathe", "eat", "drink"])

        add("mountain", "object",
            {"material": "rock", "natural": True, "immobile": True, "large": True},
            ["provide habitat", "affect weather", "erode slowly"],
            ["move", "walk", "fly", "talk", "think", "conduct electricity", "swim"])

        add("river", "object",
            {"material": "water", "natural": True, "flowing": True},
            ["flow", "carry sediment", "form valleys"],
            ["flow uphill", "freeze in summer", "swim on land"])

        add("rock", "object",
            {"material": "various minerals", "natural": True, "solid": True},
            ["sink in water", "break", "erode"],
            ["float in water (most)", "melt at room temp", "talk", "fly", "be alive"])

        add("metal", "material",
            {"conductive": True, "shiny": True, "solid_at_room_temp": True},
            ["conduct electricity", "conduct heat", "be forged"],
            ["grow", "reproduce", "be alive", "photosynthesize"])

        # ── FOOD ──
        add("fruit", "food",
            {"plant_based": True, "contains_seeds": True, "sweet": True},
            ["be eaten", "ripen", "rot"],
            ["walk", "talk", "fly", "swim"])

        add("vegetable", "food",
            {"plant_based": True, "contains_nutrients": True},
            ["be eaten", "grow in soil"],
            ["walk", "talk", "fly", "swim"])

        add("sandwich", "food",
            {"man_made": True, "has_bread": True},
            ["be eaten", "fill hunger"],
            ["fly", "walk", "talk", "swim", "be alive", "grow"])

        # ── BODY PARTS ──
        add("brain", "body_part",
            {"organ": True, "neural": True, "protected_by": "skull"},
            ["process information", "control body", "form memories"],
            ["fly", "walk", "swim"])

        add("heart", "body_part",
            {"organ": True, "muscular": True},
            ["pump blood", "beat"],
            ["think", "fly", "walk"])

        # ── CONCEPTS ──
        add("justice", "concept",
            {"abstract": True, "social_construct": True},
            ["guide behavior", "resolve disputes"],
            ["physically exist", "be held", "be seen", "weigh anything"])

        add("time", "concept",
            {"abstract": True, "measured_in": "seconds/minutes/hours"},
            ["pass", "be measured", "be recorded"],
            ["be held in hands", "be touched physically", "be seen with eyes"])

        # ── CELESTIAL ──
        add("earth", "celestial",
            {"type": "planet", "has_atmosphere": True, "has_gravity": True, "has_water": True},
            ["orbit sun", "support life", "rotate"],
            ["orbit another planet", "be a star"])

        add("sun", "celestial",
            {"type": "star", "hot": True, "luminous": True, "massive": True},
            ["emit light", "emit heat", "sustain life"],
            ["be seen at night (from earth surface, it is below horizon)", "be cool"])

        add("moon", "celestial",
            {"type": "natural satellite", "reflects_sunlight": True, "no_atmosphere": True},
            ["orbit earth", "cause tides", "reflect light"],
            ["emit light", "be seen during day sometimes (waxing/waning)"])

        # ── MATERIALS ──
        add("water", "material",
            {"state_at_room_temp": "liquid", "formula": "H2O", "transparent": True},
            ["flow", "dissolve things", "freeze", "evaporate"],
            ["flow uphill", "be dry", "conduct electricity (pure)"])

        add("ice", "material",
            {"state_at_room_temp": "solid", "cold": True, "slippery": True},
            ["slide", "melt when heated", "float on water"],
            ["flow like liquid", "be hot", "conduct electricity"])

        add("glass", "material",
            {"state_at_room_temp": "solid", "transparent": True, "fragile": True},
            ["break", "let light through"],
            ["flex", "bend without breaking", "conduct electricity"])

        # ── PLACES ──
        add("ocean", "place",
            {"contains": "saltwater", "deep": True, "has_currents": True},
            ["support marine life", "have tides"],
            ["be small", "be dry", "have no waves"])

        add("desert", "place",
            {"dry": True, "hot": True, "sandy": True},
            ["receive little rain", "support some life"],
            ["be wet", "have forests"])

        # ── MATHEMATICAL KNOWLEDGE ──
        add("1", "concept",
            {"type": "number", "prime": False, "divisors": 1},
            [],
            ["be a prime number"])

        add("prime number", "concept",
            {"definition": "exactly two distinct positive divisors", "min_divisors": 2},
            [],
            ["have one divisor", "have zero divisors"])

    def _init_relations(self) -> None:
        """Initialize known relations."""
        self._relations = [
            Relation("shark", "is", "fish", 0.99),
            Relation("shark", "is not", "mammal", 0.99),
            Relation("shark", "lives in", "ocean", 0.99),
            Relation("whale", "is", "mammal", 0.99),
            Relation("whale", "is not", "fish", 0.99),
            Relation("panda", "is", "bear", 0.99),
            Relation("panda", "eats", "bamboo", 0.99),
            Relation("dolphin", "is", "mammal", 0.99),
            Relation("bird", "lays", "eggs", 0.99),
            Relation("bird", "does not", "produce milk", 0.99),
            Relation("mammal", "produces", "milk", 0.99),
            Relation("reptile", "does not", "produce milk", 0.99),
            Relation("insect", "does not", "produce milk", 0.99),
            Relation("cat", "is", "mammal", 0.99),
            Relation("cat", "has", "whiskers", 0.99),
            Relation("dog", "is", "mammal", 0.99),
            Relation("dog", "says", "bark", 0.95),
            Relation("cat", "says", "meow", 0.95),
            Relation("cat", "does not", "bark", 0.99),
            Relation("dog", "does not", "meow", 0.99),
            Relation("fish", "breathes through", "gills", 0.99),
            Relation("mammal", "breathes through", "lungs", 0.99),
            Relation("1", "is not", "prime", 0.99),
            Relation("light", "is faster than", "sound", 0.99),
            Relation("sound", "is slower than", "light", 0.99),
            # Material composition relations
            Relation("car", "is made of", "metal", 0.95),
            Relation("building", "is made of", "concrete and steel", 0.90),
            Relation("mountain", "is made of", "rock", 0.99),
            Relation("bird", "is made of", "organic tissue", 0.99),
            Relation("cat", "is made of", "organic tissue", 0.99),
            Relation("dog", "is made of", "organic tissue", 0.99),
            Relation("fish", "is made of", "organic tissue", 0.99),
            Relation("human", "is made of", "organic tissue", 0.99),
            Relation("plant", "is made of", "organic material", 0.99),
            Relation("rock", "is made of", "minerals", 0.99),
            Relation("glass", "is made of", "silica", 0.95),
            Relation("ice", "is made of", "water", 0.99),
            # Living status
            Relation("car", "is not", "alive", 0.99),
            Relation("building", "is not", "alive", 0.99),
            Relation("mountain", "is not", "alive", 0.99),
            Relation("computer", "is not", "alive", 0.99),
            Relation("rock", "is not", "alive", 0.99),
            Relation("cat", "is", "alive", 0.99),
            Relation("dog", "is", "alive", 0.99),
            Relation("bird", "is", "alive", 0.99),
            Relation("fish", "is", "alive", 0.99),
            Relation("human", "is", "alive", 0.99),
            Relation("plant", "is", "alive", 0.99),
            # Flight capability
            Relation("cat", "cannot", "fly", 0.99),
            Relation("dog", "cannot", "fly", 0.99),
            Relation("fish", "cannot", "fly", 0.99),
            Relation("human", "cannot", "fly", 0.99),
            Relation("building", "cannot", "fly", 0.99),
            Relation("mountain", "cannot", "fly", 0.99),
        ]

    def get_entity(self, name: str) -> Entity | None:
        """Look up an entity by name."""
        return self._entities.get(name.lower().strip())

    def find_entities_in_text(self, text: str) -> list[Entity]:
        """Find all known entities mentioned in text."""
        text_lower = text.lower()
        found = []
        import re as _re
        # Sort by name length descending to match longer names first
        for name in sorted(self._entities, key=len, reverse=True):
            if name.isdigit():
                if _re.search(r'\b' + _re.escape(name) + r'\b', text_lower):
                    found.append(self._entities[name])
            else:
                # Try exact word boundary first, then with common suffixes
                matched = _re.search(r'\b' + _re.escape(name) + r'\b', text_lower)
                if not matched:
                    # Try with plural/verb suffixes
                    matched = _re.search(r'\b' + _re.escape(name) + r'(?:s|es|ed|ing|er|ly|tion|ment)\b', text_lower)
                if matched:
                    found.append(self._entities[name])
        return found

    def check_claim(self, claim: str) -> FactCheck:
        """
        Check a claim against world knowledge.

        This is the main entry point. It:
        1. Extracts entities from the claim
        2. Checks if the claim matches known abilities/NOT_abilities
        3. Checks relations
        4. Returns a plausibility assessment
        """
        claim_lower = claim.lower()
        entities = self.find_entities_in_text(claim)
        contradictions = []
        supporting = []

        for entity in entities:
            # Check NOT_abilities: things this entity definitely cannot do
            for not_ability in entity.NOT_abilities:
                na_words = set(re.findall(r'\b[a-z]{3,}\b', not_ability))
                claim_words = set(re.findall(r'\b[a-z]{3,}\b', claim_lower))
                overlap = len(na_words & claim_words)
                # Single-word ability: 1 match is enough
                # Multi-word: need 2+ meaningful matches (avoid "is" false positives)
                if overlap >= 1 and (len(na_words) <= 1 or overlap >= 2):
                    contradictions.append(
                        f"{entity.name} cannot {not_ability}"
                    )

            # Check abilities: things this entity can do
            for ability in entity.abilities:
                a_words = set(re.findall(r'\b[a-z]{3,}\b', ability))
                claim_words = set(re.findall(r'\b[a-z]{3,}\b', claim_lower))
                overlap = len(a_words & claim_words)
                if overlap >= 1 and (len(a_words) <= 1 or overlap >= 2):
                    supporting.append(
                        f"{entity.name} can {ability}"
                    )

        # Check relations
        for rel in self._relations:
            subj_words = set(re.findall(r'\b[a-z]{3,}\b', rel.subject))
            pred_words = set(re.findall(r'\b[a-z]{3,}\b', rel.predicate))
            obj_words = set(re.findall(r'\b[a-z]{3,}\b', rel.obj))
            claim_words = set(re.findall(r'\b[a-z]{3,}\b', claim_lower))

            rel_words = subj_words | pred_words | obj_words
            overlap = len(rel_words & claim_words)

            if overlap >= 2:
                if rel.predicate in ("is not", "does not", "cannot"):
                    contradictions.append(
                        f"{rel.subject} {rel.predicate} {rel.obj}"
                    )
                else:
                    supporting.append(
                        f"{rel.subject} {rel.predicate} {rel.obj}"
                    )

        # Compute plausibility
        if contradictions and not supporting:
            confidence = min(0.95, 0.7 + len(contradictions) * 0.1)
            plausible = False
            reasoning = f"Contradicts world knowledge: {'; '.join(contradictions[:3])}"
        elif supporting and not contradictions:
            confidence = min(0.95, 0.7 + len(supporting) * 0.1)
            plausible = True
            reasoning = f"Supported by world knowledge: {'; '.join(supporting[:3])}"
        elif contradictions and supporting:
            # Mixed signals: trust contradictions more (they're usually from domain knowledge)
            plausible = len(supporting) > len(contradictions)
            confidence = 0.5
            reasoning = (
                f"Mixed signals: supports=[{', '.join(supporting[:2])}] "
                f"contradicts=[{', '.join(contradictions[:2])}]"
            )
        else:
            plausible = True
            confidence = 0.5
            reasoning = "No world knowledge signal"

        return FactCheck(
            claim=claim,
            plausible=plausible,
            confidence=confidence,
            matching_entities=[e.name for e in entities],
            contradictions=contradictions,
            reasoning=reasoning,
        )

    def load_training_knowledge(self) -> None:
        """Load additional knowledge from the training module."""
        try:
            from .knowledge_training import KnowledgeTrainer
            trainer = KnowledgeTrainer()
            entries = trainer.get_all()

            # Convert knowledge entries to entities and relations
            for entry in entries:
                # Add as entity if it's a factual fact
                if entry.category in ("fact", "formula", "law", "process"):
                    name = entry.topic.lower()
                    if name not in self._entities:
                        from .world_knowledge import Entity
                        self._entities[name] = Entity(
                            name=entry.topic,
                            category=entry.domain,
                            properties={"fact": entry.fact, "answer": entry.answer},
                            abilities=[entry.fact],
                            not_abilities=[],
                        )

                # Add relations from knowledge entries
                if entry.category == "relation":
                    # Parse relation from fact
                    fact = entry.fact.lower()
                    if " is " in fact:
                        parts = fact.split(" is ", 1)
                        if len(parts) == 2:
                            self._relations.append(
                                Relation(parts[0].strip(), "is", parts[1].strip(), entry.confidence)
                            )

            logger.info(f"Loaded {len(entries)} knowledge entries from training module")
        except Exception as e:
            logger.warning(f"Failed to load training knowledge: {e}")

    @property
    def entity_count(self) -> int:
        return len(self._entities)

    @property
    def relation_count(self) -> int:
        return len(self._relations)
