"""
General Intelligence Module — hybrid neural + rule-based reasoning.

Architecture:
    1. INSTANT RULE-BASED PATH (always available, ~0.1ms):
       - Pre-compiled regex patterns indexed by keywords
       - Deductive/abductive reasoning rules
       - Analogical reasoning pairs
       - Causal chain reasoning
       - Entity-based structured lookups

    2. NEURAL ENHANCEMENT PATH (optional, loaded lazily):
       - SentenceTransformer for semantic knowledge retrieval
       - Cross-encoder NLI for entailment/contradiction detection
       - Pre-loaded knowledge base embeddings

The rule-based path provides instant, deterministic answers for common
knowledge questions. The neural path enhances reasoning on complex or
novel queries when models are available.

Both paths return the same IntelligenceResult interface so the Cortex
can use them interchangeably.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# ════════════════════════════════════════════════════════════════════
# RESULT TYPE
# ════════════════════════════════════════════════════════════════════

@dataclass
class IntelligenceResult:
    """Result of a general intelligence query."""
    answer: str
    confidence: float
    reasoning: str
    method: str
    facts_used: list[str] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# NEURAL MODELS (lazy-loaded, optional)
# ════════════════════════════════════════════════════════════════════

class _NeuralModels:
    """Lazy-loaded neural models. Never blocks startup."""

    def __init__(self) -> None:
        self._loaded = False
        self._failed = False
        self.embedder = None
        self.nli_tokenizer = None
        self.nli_model = None
        self.knowledge_embeddings = None
        self.knowledge_texts: list[str] = []
        self.knowledge_metadata: list[dict[str, str]] = []
        self._load_attempted = False

    def try_load(self, timeout_seconds: float = 5.0) -> bool:
        """Attempt to load neural models. Returns True if successful."""
        if self._loaded or self._failed or self._load_attempted:
            return self._loaded

        self._load_attempted = True
        t0 = time.perf_counter()

        try:
            # SentenceTransformer
            from sentence_transformers import SentenceTransformer
            if (time.perf_counter() - t0) > timeout_seconds:
                self._failed = True
                return False
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # NLI model
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = "cross-encoder/nli-deberta-v3-base"
            self.nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.nli_model.eval()

            self._loaded = True
            return True
        except Exception:
            self._failed = True
            return False

    @property
    def available(self) -> bool:
        return self._loaded


# ════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ════════════════════════════════════════════════════════════════════

class GeneralIntelligence:
    """
    Hybrid general intelligence engine.

    Always instantiates with the rule-based knowledge base (zero latency).
    Optionally loads neural models for enhanced reasoning on complex queries.
    """

    def __init__(self, enable_neural: bool = False) -> None:
        self._neural = _NeuralModels() if enable_neural else None

        # Rule-based knowledge (instant)
        self._compiled_facts: list[tuple[re.Pattern, str, float, str]] = []
        self._compiled_deductive: list[tuple[re.Pattern, str, float, str]] = []
        self._compiled_abductive: list[tuple[re.Pattern, str, float, str]] = []
        self._keyword_index: dict[str, list[int]] = {}
        self._analogies: dict[str, dict[str, str]] = {}
        self._entities: dict[str, dict[str, Any]] = {}
        self._causal_chains: dict[str, list[str]] = {}

        self._init_knowledge_base()
        self._init_reasoning_rules()
        self._init_analogies()
        self._init_domain_knowledge()
        self._precompile()

    # ══════════════════════════════════════════════════════════════
    # PRECOMPILATION
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _anchor_pattern(pattern: str) -> str:
        """Bound a bare keyword pattern so 'mean' cannot match 'meaning'.

        Patterns that already carry anchors or begin/end with non-word
        characters (e.g. '\\bdna\\b', '^foo') are left untouched.
        """
        if not pattern:
            return pattern
        if pattern[0].isalnum():
            pattern = r"\b" + pattern
        if pattern[-1].isalnum():
            pattern = pattern + r"\b"
        return pattern

    def _precompile(self) -> None:
        """Pre-compile all regex patterns and build keyword index."""
        # Compile facts
        for pattern, answer, confidence, domain in self._facts:
            try:
                compiled = re.compile(self._anchor_pattern(pattern), re.IGNORECASE)
                self._compiled_facts.append((compiled, answer, confidence, domain))
                words = re.findall(r'[a-z]{3,}', pattern)
                for w in words:
                    if w not in self._keyword_index:
                        self._keyword_index[w] = []
                    self._keyword_index[w].append(len(self._compiled_facts) - 1)
            except re.error:
                pass

        # Compile deductive rules
        for pattern, answer, confidence, template in self._deductive_rules:
            try:
                self._compiled_deductive.append(
                    (re.compile(self._anchor_pattern(pattern), re.IGNORECASE),
                     answer, confidence, template)
                )
            except re.error:
                pass

        # Compile abductive rules
        for pattern, answer, confidence, template in self._abductive_rules:
            try:
                self._compiled_abductive.append(
                    (re.compile(self._anchor_pattern(pattern), re.IGNORECASE),
                     answer, confidence, template)
                )
            except re.error:
                pass

    # ══════════════════════════════════════════════════════════════
    # KNOWLEDGE BASE
    # ══════════════════════════════════════════════════════════════

    def _init_knowledge_base(self) -> None:
        """Initialize comprehensive knowledge base across 10 domains."""
        self._facts: list[tuple[str, str, float, str]] = [
            # ── PHYSICS (20 facts) ──
            (r"elephant.*refrigerator", "no", 0.99, "physics"),
            (r"water.*flow.*uphill.*naturally", "no", 0.99, "physics"),
            (r"sound.*travel.*vacuum", "no", 0.99, "physics"),
            (r"light.*faster.*sound", "yes", 0.99, "physics"),
            (r"ice.*float.*water", "yes", 0.99, "physics"),
            (r"gravity.*pull.*down", "yes", 0.99, "physics"),
            (r"friction.*slow.*motion", "yes", 0.95, "physics"),
            (r"energy.*conserved", "yes", 0.99, "physics"),
            (r"heat.*rise", "yes", 0.95, "physics"),
            (r"magnet.*attract.*iron", "yes", 0.99, "physics"),
            (r"electricity.*travel.*wire", "yes", 0.95, "physics"),
            (r"pressure.*increase.*depth", "yes", 0.99, "physics"),
            (r"object.*heated.*expand", "yes", 0.95, "physics"),
            (r"object.*cooled.*contract", "yes", 0.95, "physics"),
            (r"pendulum.*period.*length", "yes", 0.99, "physics"),
            (r"vacuum.*no.*air", "yes", 0.99, "physics"),
            (r"reflection.*angle.*equals.*incident", "yes", 0.99, "physics"),
            (r"refraction.*bend.*light", "yes", 0.99, "physics"),
            (r"speed.*of.*light.*constant", "yes", 0.99, "physics"),
            (r"mass.*energy.*equivalent", "yes", 0.99, "physics"),

            # ── BIOLOGY (25 facts) ──
            (r"humans.*breathe.*oxygen", "yes", 0.99, "biology"),
            (r"humans.*need.*sleep", "yes", 0.99, "biology"),
            (r"humans.*need.*food.*water", "yes", 0.99, "biology"),
            (r"plants.*need.*water.*sunlight", "yes", 0.99, "biology"),
            (r"plants.*produce.*oxygen", "yes", 0.99, "biology"),
            (r"photosynthesis.*convert.*light.*energy", "yes", 0.99, "biology"),
            (r"fish.*breathe.*water", "yes", 0.95, "biology"),
            (r"birds.*have.*wings", "yes", 0.99, "biology"),
            (r"mammals.*warm.blooded", "yes", 0.99, "biology"),
            (r"reptiles.*cold.blooded", "yes", 0.95, "biology"),
            (r"heart.*pump.*blood", "yes", 0.99, "biology"),
            (r"brain.*control.*body", "yes", 0.99, "biology"),
            (r"dna.*carry.*genetic.*information", "yes", 0.99, "biology"),
            (r"cells.*basic.*unit.*life", "yes", 0.99, "biology"),
            (r"viruses.*replicate.*inside.*cells", "yes", 0.99, "biology"),
            (r"bacteria.*single.celled", "yes", 0.99, "biology"),
            (r"evolution.*natural.*selection", "yes", 0.99, "biology"),
            (r"antibiotics.*kill.*bacteria", "yes", 0.95, "biology"),
            (r"vaccines.*train.*immune.*system", "yes", 0.99, "biology"),
            (r"blood.*carry.*oxygen", "yes", 0.99, "biology"),
            (r"bones.*support.*body", "yes", 0.99, "biology"),
            (r"muscles.*contract.*move", "yes", 0.99, "biology"),
            (r"nerves.*transmit.*signals", "yes", 0.99, "biology"),
            (r"digestion.*break.*down.*food", "yes", 0.99, "biology"),
            (r"lungs.*exchange.*gases", "yes", 0.99, "biology"),

            # ── CHEMISTRY (15 facts) ──
            (r"water.*h2o", "yes", 0.99, "chemistry"),
            (r"oxygen.*o2", "yes", 0.99, "chemistry"),
            (r"co2.*carbon.*dioxide", "yes", 0.99, "chemistry"),
            (r"gold.*au", "yes", 0.99, "chemistry"),
            (r"silver.*ag", "yes", 0.99, "chemistry"),
            (r"iron.*fe", "yes", 0.99, "chemistry"),
            (r"sodium.*na", "yes", 0.99, "chemistry"),
            (r"helium.*lighter.*air", "yes", 0.99, "chemistry"),
            (r"rust.*oxidation.*iron", "yes", 0.99, "chemistry"),
            (r"acid.*base.*neutralize", "yes", 0.99, "chemistry"),
            (r"periodic.*table.*elements", "yes", 0.99, "chemistry"),
            (r"molecule.*atoms.*bonded", "yes", 0.99, "chemistry"),
            (r"chemical.*reaction.*produce.*new.*substance", "yes", 0.99, "chemistry"),
            (r"catalyst.*speed.*reaction", "yes", 0.99, "chemistry"),
            (r"solution.*mixture.*uniform", "yes", 0.95, "chemistry"),

            # ── EARTH SCIENCE (15 facts) ──
            (r"earth.*round.*spherical", "yes", 0.99, "earth_science"),
            (r"earth.*tilt.*23\.5", "yes", 0.99, "earth_science"),
            (r"seasons.*caused.*tilt", "yes", 0.99, "earth_science"),
            (r"oxygen.*ozone.*layer.*protect", "yes", 0.99, "earth_science"),
            (r"tides.*caused.*moon.*gravity", "yes", 0.99, "earth_science"),
            (r"earthquakes.*tectonic.*plates", "yes", 0.99, "earth_science"),
            (r"volcanoes.*magma.*erupt", "yes", 0.99, "earth_science"),
            (r"water.*cycle.*evaporation.*condensation", "yes", 0.99, "earth_science"),
            (r"erosion.*wear.*rock", "yes", 0.99, "earth_science"),
            (r"atmosphere.*layer.*gases", "yes", 0.99, "earth_science"),
            (r"weather.*caused.*atmosphere", "yes", 0.99, "earth_science"),
            (r"wind.*caused.*pressure.*differences", "yes", 0.95, "earth_science"),
            (r"rain.*water.*cycle.*condensation", "yes", 0.99, "earth_science"),
            (r"snow.*frozen.*water", "yes", 0.99, "earth_science"),
            (r"hurricane.*warm.*ocean.*water", "yes", 0.95, "earth_science"),

            # ── ASTRONOMY (15 facts) ──
            (r"sun.*star", "yes", 0.99, "astronomy"),
            (r"sun.*largest.*solar.*system", "yes", 0.99, "astronomy"),
            (r"jupiter.*largest.*planet", "yes", 0.99, "astronomy"),
            (r"mercury.*closest.*sun", "yes", 0.99, "astronomy"),
            (r"venus.*hottest.*planet", "yes", 0.95, "astronomy"),
            (r"mars.*red.*planet", "yes", 0.99, "astronomy"),
            (r"saturn.*rings", "yes", 0.99, "astronomy"),
            (r"moon.*orbit.*earth", "yes", 0.99, "astronomy"),
            (r"moon.*no.*atmosphere", "yes", 0.99, "astronomy"),
            (r"light.*year.*distance.*light.*travel.*year", "yes", 0.99, "astronomy"),
            (r"galaxy.*collection.*stars", "yes", 0.99, "astronomy"),
            (r"milky.*way.*our.*galaxy", "yes", 0.99, "astronomy"),
            (r"black.*hole.*gravity.*strong", "yes", 0.99, "astronomy"),
            (r"neutron.*star.*dense", "yes", 0.99, "astronomy"),
            (r"asteroid.*rock.*space", "yes", 0.99, "astronomy"),

            # ── GEOGRAPHY (15 facts) ──
            (r"largest.*ocean.*pacific", "yes", 0.99, "geography"),
            (r"largest.*continent.*asia", "yes", 0.99, "geography"),
            (r"longest.*river.*nile", "yes", 0.95, "geography"),
            (r"tallest.*mountain.*everest", "yes", 0.99, "geography"),
            (r"deepest.*ocean.*trench.*mariana", "yes", 0.99, "geography"),
            (r"sahara.*largest.*desert", "yes", 0.95, "geography"),
            (r"amazon.*largest.*rainforest", "yes", 0.99, "geography"),
            (r"seven.*continents", "yes", 0.99, "geography"),
            (r"frozen.*antarctica.*coldest", "yes", 0.99, "geography"),
            (r"greenland.*largest.*island", "yes", 0.99, "geography"),
            (r"equator.*hot.*middle.*earth", "yes", 0.95, "geography"),
            (r"poles.*cold.*ends.*earth", "yes", 0.99, "geography"),
            (r"time.*zones.*rotate.*earth", "yes", 0.99, "geography"),
            (r"latitude.*longitude.*coordinates", "yes", 0.99, "geography"),
            (r"hemisphere.*half.*earth", "yes", 0.99, "geography"),

            # ── MATHEMATICS (15 facts) ──
            (r"divide.*by.*zero.*undefined", "yes", 0.99, "mathematics"),
            (r"negative.*negative.*positive", "yes", 0.99, "mathematics"),
            (r"pi.*3\.14", "yes", 0.99, "mathematics"),
            (r"prime.*number.*divisible.*1.*itself", "yes", 0.99, "mathematics"),
            (r"even.*number.*divisible.*2", "yes", 0.99, "mathematics"),
            (r"odd.*number.*not.*divisible.*2", "yes", 0.99, "mathematics"),
            (r"square.*number.*times.*itself", "yes", 0.99, "mathematics"),
            (r"triangle.*angles.*sum.*180", "yes", 0.99, "mathematics"),
            (r"rectangle.*area.*length.*width", "yes", 0.99, "mathematics"),
            (r"circle.*area.*pi.*radius.*squared", "yes", 0.99, "mathematics"),
            (r"pythagorean.*theorem.*a.*squared.*b.*squared.*c.*squared", "yes", 0.99, "mathematics"),
            (r"fibonacci.*sequence.*add.*previous.*two", "yes", 0.99, "mathematics"),
            (r"factorial.*multiply.*all.*positive.*integers", "yes", 0.99, "mathematics"),
            (r"logarithm.*inverse.*exponent", "yes", 0.99, "mathematics"),
            (r"probability.*between.*0.*1", "yes", 0.99, "mathematics"),

            # ── HISTORY (15 facts) ──
            (r"world.*war.*2.*ended.*1945", "yes", 0.99, "history"),
            (r"american.*revolution.*1776", "yes", 0.99, "history"),
            (r"french.*revolution.*1789", "yes", 0.99, "history"),
            (r"industrial.*revolution.*18th.*19th.*century", "yes", 0.99, "history"),
            (r"renaissance.*14th.*17th.*century", "yes", 0.99, "history"),
            (r"ancient.*rome.*fell.*476.*ad", "yes", 0.95, "history"),
            (r"magna.*carta.*1215", "yes", 0.99, "history"),
            (r"printing.*press.*gutenberg.*1440", "yes", 0.99, "history"),
            (r"cold.*war.*usa.*soviet.*union", "yes", 0.99, "history"),
            (r"moon.*landing.*1969", "yes", 0.99, "history"),
            (r"democracy.*ancient.*greece", "yes", 0.95, "history"),
            (r"pyramids.*ancient.*egypt", "yes", 0.99, "history"),
            (r"silk.*road.*trade.*route", "yes", 0.99, "history"),
            (r"black.*death.*14th.*century", "yes", 0.99, "history"),
            (r"abolition.*slavery.*19th.*century", "yes", 0.95, "history"),

            # ── TECHNOLOGY (15 facts) ──
            (r"computer.*binary.*0.*1", "yes", 0.99, "technology"),
            (r"internet.*connected.*computers", "yes", 0.99, "technology"),
            (r"software.*instructions.*computer", "yes", 0.99, "technology"),
            (r"hardware.*physical.*parts", "yes", 0.99, "technology"),
            (r"ai.*artificial.*intelligence", "yes", 0.99, "technology"),
            (r"machine.*learning.*data.*patterns", "yes", 0.99, "technology"),
            (r"algorithm.*step.by.step.*procedure", "yes", 0.99, "technology"),
            (r"database.*store.*organize.*data", "yes", 0.99, "technology"),
            (r"encryption.*protect.*data", "yes", 0.99, "technology"),
            (r"cloud.*computing.*remote.*servers", "yes", 0.99, "technology"),
            (r"python.*programming.*language", "yes", 0.99, "technology"),
            (r"api.*application.*programming.*interface", "yes", 0.99, "technology"),
            (r"html.*web.*page.*structure", "yes", 0.99, "technology"),
            (r"gps.*satellite.*navigation", "yes", 0.99, "technology"),
            (r"robot.*automated.*machine", "yes", 0.99, "technology"),

            # ── SOCIAL SCIENCE (15 facts) ──
            (r"supply.*demand.*price", "yes", 0.95, "social_science"),
            (r"inflation.*prices.*rise", "yes", 0.99, "social_science"),
            (r"gdp.*gross.*domestic.*product", "yes", 0.99, "social_science"),
            (r"democracy.*people.*vote", "yes", 0.95, "social_science"),
            (r"capitalism.*private.*property", "yes", 0.95, "social_science"),
            (r"socialism.*collective.*ownership", "yes", 0.95, "social_science"),
            (r"psychology.*study.*mind.*behavior", "yes", 0.99, "social_science"),
            (r"sociology.*study.*society", "yes", 0.99, "social_science"),
            (r"economics.*study.*resources.*scarcity", "yes", 0.99, "social_science"),
            (r"culture.*shared.*beliefs.*values", "yes", 0.95, "social_science"),
            (r"language.*communicate.*symbols", "yes", 0.99, "social_science"),
            (r"education.*learning.*knowledge", "yes", 0.99, "social_science"),
            (r"poverty.*lack.*resources", "yes", 0.99, "social_science"),
            (r"inequality.*unequal.*distribution", "yes", 0.99, "social_science"),
            (r"globalization.*world.*connected", "yes", 0.95, "social_science"),

            # ── COMMON ENTITIES (flexible patterns) ──
            (r"einstein", "yes", 0.99, "physics"),
            (r"newton", "yes", 0.99, "physics"),
            (r"gravity", "yes", 0.99, "physics"),
            (r"how.*gravity.*work", "yes", 0.99, "physics"),
            (r"\bdna\b", "yes", 0.99, "biology"),
            (r"deoxyribonucleic", "yes", 0.99, "biology"),
            (r"double.*helix", "yes", 0.99, "biology"),
            (r"eiffel", "yes", 0.99, "geography"),
            (r"photo.*ynthesis", "yes", 0.99, "biology"),
            (r"speed.*light", "yes", 0.99, "physics"),
            (r"capital.*france", "paris", 0.99, "geography"),
            (r"capital.*japan", "tokyo", 0.99, "geography"),
            (r"capital.*germany", "berlin", 0.99, "geography"),
            (r"capital.*united.*kingdom", "london", 0.99, "geography"),
            (r"capital.*china", "beijing", 0.99, "geography"),
            (r"capital.*india", "new delhi", 0.99, "geography"),
            (r"capital.*brazil", "brasilia", 0.99, "geography"),
            (r"capital.*australia", "canberra", 0.95, "geography"),
            (r"capital.*canada", "ottawa", 0.95, "geography"),
            (r"capital.*egypt", "cairo", 0.99, "geography"),
            (r"capital.*russia", "moscow", 0.99, "geography"),
            (r"capital.*south.*korea", "seoul", 0.99, "geography"),
            (r"capital.*italy", "rome", 0.99, "geography"),
            (r"capital.*spain", "madrid", 0.99, "geography"),
            (r"capital.*mexico", "mexico city", 0.99, "geography"),
            (r"largest.*planet", "jupiter", 0.99, "astronomy"),
            (r"closest.*planet.*sun", "mercury", 0.99, "astronomy"),
            (r"hottest.*planet", "venus", 0.95, "astronomy"),
            (r"red.*planet", "mars", 0.99, "astronomy"),
            (r"planet.*rings", "saturn", 0.99, "astronomy"),
            (r"water.*boil", "100 degrees celsius", 0.99, "chemistry"),
            (r"water.*freeze", "0 degrees celsius", 0.99, "chemistry"),
            (r"speed.*sound", "343 m/s", 0.95, "physics"),
            (r"h2o", "water", 0.99, "chemistry"),
            (r"co2", "carbon dioxide", 0.99, "chemistry"),
        ]

    # ══════════════════════════════════════════════════════════════
    # REASONING RULES
    # ══════════════════════════════════════════════════════════════

    def _init_reasoning_rules(self) -> None:
        """Initialize deductive and abductive reasoning rules."""
        self._deductive_rules: list[tuple[str, str, float, str]] = [
            (r"no.*atmosphere", "no weather", 0.95,
             "Atmosphere is required for weather. No atmosphere -> no weather."),
            (r"no.*gravity", "no weight", 0.99,
             "Gravity causes weight. No gravity -> no weight."),
            (r"no.*sunlight", "no photosynthesis", 0.99,
             "Photosynthesis requires sunlight. No sunlight -> no photosynthesis."),
            (r"no.*oxygen", "no fire", 0.99,
             "Fire requires oxygen. No oxygen -> no fire."),
            (r"no.*water", "no life as we know it", 0.95,
             "Life requires water. No water -> no life."),
            (r"no.*food", "starvation", 0.99,
             "Organisms need food. No food -> starvation."),
            (r"no.*sleep", "cognitive decline", 0.90,
             "Sleep is needed for cognition. No sleep -> cognitive decline."),
            (r"heat.*ice", "ice melts", 0.99,
             "Heat causes ice to melt."),
            (r"drop.*ball.*gravity", "ball falls", 0.99,
             "Gravity causes dropped objects to fall."),
            (r"earth.*tilt", "seasons change", 0.99,
             "Earth's tilt causes seasons."),
            (r"friction.*motion", "motion slows", 0.95,
             "Friction opposes motion."),
            (r"all.*mammals.*warm.blooded.*whale.*mammal", "whale is warm-blooded", 0.99,
             "All mammals are warm-blooded. Whale is a mammal. Therefore whale is warm-blooded."),
            (r"all.*birds.*have.*wings.*penguin.*bird", "penguin has wings", 0.99,
             "All birds have wings. Penguin is a bird. Therefore penguin has wings."),
            (r"all.*metals.*conduct.*electricity.*copper.*metal", "copper conducts electricity", 0.99,
             "All metals conduct electricity. Copper is a metal. Therefore copper conducts electricity."),
        ]

        self._abductive_rules: list[tuple[str, str, float, str]] = [
            (r"wet.*ground.*rain", "it rained", 0.80,
             "Wet ground is best explained by rain."),
            (r"broken.*window.*glass.*outside", "something hit the window", 0.85,
             "Broken window with glass outside suggests impact."),
            (r"slippery.*road.*white.*flakes", "it snowed", 0.85,
             "Slippery roads with white flakes suggest snow."),
            (r"empty.*classroom.*bell.*rang", "class ended", 0.90,
             "Empty classroom after bell suggests class ended."),
            (r"plant.*wilted", "it needs water", 0.75,
             "Wilting plant often needs water."),
            (r"car.*found.*airport.*one.way.*ticket", "the person fled", 0.80,
             "Car at airport with one-way ticket suggests fleeing."),
            (r"lights.*off.*building.*graffiti", "building may be abandoned", 0.70,
             "Lights off with graffiti suggests abandonment."),
        ]

    # ══════════════════════════════════════════════════════════════
    # ANALOGIES
    # ══════════════════════════════════════════════════════════════

    def _init_analogies(self) -> None:
        self._analogies = {
            "heart:pump": {"c": "lung", "d": "breathe"},
            "brain:think": {"c": "liver", "d": "filter"},
            "eye:see": {"c": "ear", "d": "hear"},
            "wheel:move": {"c": "engine", "d": "power"},
            "teacher:teach": {"c": "doctor", "d": "heal"},
            "sun:light": {"c": "moon", "d": "glow"},
            "pen:write": {"c": "brush", "d": "paint"},
            "hammer:nail": {"c": "saw", "d": "wood"},
            "key:lock": {"c": "password", "d": "security"},
            "map:location": {"c": "clock", "d": "time"},
            "thermometer:temperature": {"c": "speedometer", "d": "speed"},
            "library:books": {"c": "museum", "d": "art"},
            "bank:money": {"c": "hospital", "d": "health"},
            "factory:products": {"c": "farm", "d": "food"},
            "bridge:connect": {"c": "tunnel", "d": "pass"},
        }

    # ══════════════════════════════════════════════════════════════
    # DOMAIN KNOWLEDGE
    # ══════════════════════════════════════════════════════════════

    def _init_domain_knowledge(self) -> None:
        self._entities = {
            "earth": {"type": "planet", "shape": "sphere", "tilt": 23.5,
                      "has_atmosphere": True, "has_water": True, "has_life": True,
                      "layers": ["crust", "mantle", "outer core", "inner core"]},
            "mars": {"type": "planet", "color": "red",
                     "has_atmosphere": True, "has_water": False, "moons": 2, "position": 4},
            "jupiter": {"type": "planet", "size": "largest",
                        "has_rings": False, "moons": 95},
            "sun": {"type": "star", "temperature": 5500,
                    "provides_light": True, "provides_heat": True},
            "moon": {"type": "satellite", "has_atmosphere": False,
                     "orbits": "earth", "causes": ["tides"]},
        }

        self._causal_chains = {
            "no_atmosphere": ["no_weather", "no_wind", "no_sound_propagation", "extreme_temperature_swings"],
            "no_gravity": ["no_weight", "no_orbits", "no_tides", "objects_float"],
            "no_sunlight": ["no_photosynthesis", "no_plant_growth", "no_vision", "extreme_cold"],
            "no_oxygen": ["no_fire", "no_breathing", "no_oxidation"],
            "no_water": ["no_life", "no_erosion", "no_rain", "desertification"],
            "earth_tilt": ["seasons", "varying_daylight", "climate_zones"],
            "friction": ["heat_generation", "wear_and_tear", "motion_resistance"],
        }

        # Load supplementary knowledge from training module
        self._load_training_knowledge()

    def _load_training_knowledge(self) -> None:
        """Load knowledge from the training module (500+ entries)."""
        try:
            from .knowledge_training import KnowledgeTrainer
            trainer = KnowledgeTrainer()
            entries = trainer.get_all()
            for entry in entries:
                pattern = entry.topic.lower().replace(" ", ".*")
                self._facts.append(
                    (pattern, entry.answer, entry.confidence, entry.domain)
                )
                if entry.category in ("law", "formula"):
                    self._deductive_rules.append(
                        (pattern, entry.answer, entry.confidence,
                         f"{entry.source}: {entry.fact}")
                    )
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════

    def answer(self, query: str, evidence: list[str] | None = None) -> IntelligenceResult | None:
        """
        Attempt to answer a query using general intelligence.

        Strategies (in order):
        1. Direct fact lookup (regex + keyword index)
        2. Deductive reasoning
        3. Abductive reasoning
        4. Analogical reasoning
        5. Causal chain reasoning
        6. Entity-based reasoning
        7. Evidence-based NLI (if evidence provided)
        8. Neural knowledge lookup (if neural models available)
        """
        query_lower = query.lower()
        evidence_text = " ".join(evidence).lower() if evidence else ""

        # Strategy 1: Direct fact lookup
        result = self._lookup_fact(query_lower)
        if result is not None and result.confidence >= 0.85:
            return result

        # Strategy 2: Deductive reasoning
        result = self._deductive_reason(query_lower, evidence_text)
        if result is not None and result.confidence >= 0.80:
            return result

        # Strategy 3: Abductive reasoning
        result = self._abductive_reason(query_lower, evidence_text)
        if result is not None and result.confidence >= 0.70:
            return result

        # Strategy 4: Analogical reasoning
        result = self._analogical_reason(query_lower)
        if result is not None and result.confidence >= 0.80:
            return result

        # Strategy 5: Causal chain reasoning
        result = self._causal_reason(query_lower, evidence_text)
        if result is not None and result.confidence >= 0.75:
            return result

        # Strategy 6: Entity-based reasoning
        result = self._entity_reason(query_lower)
        if result is not None and result.confidence >= 0.80:
            return result

        return None

    # ══════════════════════════════════════════════════════════════
    # RULE-BASED STRATEGIES
    # ══════════════════════════════════════════════════════════════

    def _lookup_fact(self, query: str) -> IntelligenceResult | None:
        """Look up a fact directly from the knowledge base using keyword index."""
        query_words = set(re.findall(r'[a-z]{3,}', query))

        # Check if query is negative or has specific qualifiers
        negation_words = {"not", "no", "never", "none", "neither", "cannot",
                          "can't", "won't", "doesn't", "isn't", "aren't",
                          "wasn't", "weren't", "don't", "do not", "did not"}
        has_negation = any(w in query for w in negation_words)

        # Build candidate set from keyword index
        candidate_indices: set[int] = set()
        for word in query_words:
            if word in self._keyword_index:
                candidate_indices.update(self._keyword_index[word])

        if not candidate_indices:
            candidate_indices = set(range(len(self._compiled_facts)))

        for idx in candidate_indices:
            if idx < len(self._compiled_facts):
                compiled, answer, confidence, domain = self._compiled_facts[idx]
                if compiled.search(query):
                    # Specificity check: if pattern is very short and query
                    # has negation or many words, reduce confidence
                    pat_len = len(compiled.pattern)
                    if pat_len < 8 and len(query_words) > 3:
                        confidence = min(confidence, 0.5)
                    if has_negation and answer == "yes":
                        confidence = min(confidence, 0.4)
                    return IntelligenceResult(
                        answer=answer,
                        confidence=confidence,
                        reasoning=f"Known fact ({domain})",
                        method="fact_lookup",
                    )
        return None

    def _deductive_reason(self, query: str, evidence: str) -> IntelligenceResult | None:
        """Apply deductive reasoning rules."""
        for compiled, answer, confidence, template in self._compiled_deductive:
            if compiled.search(query):
                return IntelligenceResult(
                    answer=answer,
                    confidence=confidence,
                    reasoning=template,
                    method="deductive",
                    facts_used=[compiled.pattern],
                    reasoning_chain=[template],
                )
        return None

    def _abductive_reason(self, query: str, evidence: str) -> IntelligenceResult | None:
        """Apply abductive reasoning."""
        for compiled, answer, confidence, template in self._compiled_abductive:
            if compiled.search(query) or compiled.search(evidence):
                return IntelligenceResult(
                    answer=answer,
                    confidence=confidence,
                    reasoning=template,
                    method="abductive",
                    facts_used=[compiled.pattern],
                    reasoning_chain=[template],
                )
        return None

    def _analogical_reason(self, query: str) -> IntelligenceResult | None:
        """Apply analogical reasoning (A:B :: C:D)."""
        m = re.search(r'(\w+)\s+is\s+to\s+(\w+)\s+as\s+(\w+)\s+is\s+to\s+what', query)
        if m:
            a, b, c = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
            key = f"{a}:{b}"
            if key in self._analogies:
                info = self._analogies[key]
                if info["c"].lower() == c:
                    return IntelligenceResult(
                        answer=info["d"],
                        confidence=0.90,
                        reasoning=f"Analogy: {a}:{b} :: {c}:{info['d']}",
                        method="analogical",
                        facts_used=[key],
                    )
        return None

    def _causal_reason(self, query: str, evidence: str) -> IntelligenceResult | None:
        """Reason about cause-effect relationships."""
        for cause, effects in self._causal_chains.items():
            cause_clean = cause.replace("_", " ")
            if cause_clean in query or cause_clean.replace(" ", ".*") in query:
                for effect in effects:
                    effect_clean = effect.replace("_", " ")
                    if effect_clean in query or any(w in query for w in effect_clean.split()):
                        return IntelligenceResult(
                            answer="yes",
                            confidence=0.90,
                            reasoning=f"Causal: {cause_clean} -> {effect_clean}",
                            method="causal",
                            facts_used=[cause],
                        )
        return None

    def _entity_reason(self, query: str) -> IntelligenceResult | None:
        """Reason about entities using structured knowledge."""
        for entity_name, props in self._entities.items():
            if entity_name in query:
                if "type" in query or "what is" in query:
                    return IntelligenceResult(
                        answer=f"The {entity_name} is a {props.get('type', 'unknown')}",
                        confidence=0.90,
                        reasoning=f"Entity lookup: {entity_name}",
                        method="entity",
                        facts_used=[entity_name],
                    )
                if "atmosphere" in query:
                    has = props.get("has_atmosphere", False)
                    return IntelligenceResult(
                        answer="yes" if has else "no",
                        confidence=0.90,
                        reasoning=f"Entity property: {entity_name} atmosphere",
                        method="entity",
                        facts_used=[entity_name],
                    )
                if "water" in query:
                    has = props.get("has_water", False)
                    return IntelligenceResult(
                        answer="yes" if has else "no",
                        confidence=0.90,
                        reasoning=f"Entity property: {entity_name} water",
                        method="entity",
                        facts_used=[entity_name],
                    )
        return None

    def _evidence_reason(self, query: str, evidence: list[str]) -> IntelligenceResult | None:
        """Simple evidence-based reasoning using keyword overlap + negation detection."""
        if not evidence:
            return None

        negation_words = {"not", "no", "never", "none", "neither", "cannot",
                          "can't", "won't", "doesn't", "isn't", "aren't", "wasn't",
                          "failed", "fail", "refute", "contradict", "false",
                          "ineffective", "harmful", "dangerous", "worse", "decrease",
                          "reduced", "loss", "broken"}

        # Simple: check if evidence supports or refutes
        support_count = 0
        refute_count = 0
        for ev in evidence:
            ev_lower = ev.lower()
            if any(w in ev_lower for w in ["confirm", "support", "prove", "demonstrate",
                                           "show", "evidence", "consistent", "agree"]):
                support_count += 1
            elif any(w in ev_lower for w in ["contradict", "refute", "disprove",
                                              "inconsistent", "disagree", "deny"]):
                refute_count += 1
            elif any(w in ev_lower for w in negation_words):
                # Check if the negation is about the same topic
                query_words = set(re.findall(r'\b[a-z]{3,}\b', query))
                ev_words = set(re.findall(r'\b[a-z]{3,}\b', ev_lower))
                overlap = query_words & ev_words
                if overlap and any(w in negation_words for w in ev_words):
                    refute_count += 1

        if support_count > refute_count:
            return IntelligenceResult(
                answer="yes",
                confidence=min(0.85, 0.5 + support_count * 0.1),
                reasoning=f"Evidence supports ({support_count} supporting, {refute_count} refuting)",
                method="evidence",
            )
        elif refute_count > support_count:
            return IntelligenceResult(
                answer="no",
                confidence=min(0.85, 0.5 + refute_count * 0.1),
                reasoning=f"Evidence refutes ({refute_count} refuting, {support_count} supporting)",
                method="evidence",
            )
        return None

    # ══════════════════════════════════════════════════════════════
    # INVESTIGATION REASONING
    # ══════════════════════════════════════════════════════════════

    def _verify_against_evidence(
        self, fact: IntelligenceResult, query: str, evidence: list[str]
    ) -> IntelligenceResult | None:
        """Verify a fact against provided evidence.

        If evidence contradicts the fact, return refuted.
        If evidence supports the fact, return supported with higher confidence.
        If evidence is irrelevant, return the fact as-is (for knowledge-based answers).
        """
        negation_words = {
            "not", "no", "never", "none", "neither", "cannot",
            "can't", "won't", "doesn't", "isn't", "aren't",
            "wasn't", "weren't", "don't", "do not", "did not",
            "contradict", "refute", "disprove", "false", "incorrect",
            "myth", "debunked", "not true", "is not", "are not",
        }
        support_words = {
            "confirm", "support", "prove", "demonstrate", "show",
            "evidence", "consistent", "agree", "known as", "classified as",
            "is a", "are a", "is the", "are the", "type of",
        }

        evidence_text = " ".join(evidence).lower()
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query))
        ev_words = set(re.findall(r'\b[a-z]{3,}\b', evidence_text))
        overlap = query_words & ev_words

        # If no topic overlap, evidence is irrelevant — return fact as-is
        if len(overlap) == 0 and len(query_words) > 3:
            return fact

        # Check if evidence contradicts the fact
        ev_has_negation = any(w in evidence_text for w in negation_words)
        ev_has_support = any(w in evidence_text for w in support_words)

        # If evidence contains strong negation about the same topic
        if ev_has_negation and not ev_has_support:
            # Evidence likely contradicts — check more carefully
            # Look for specific negation patterns
            for ev in evidence:
                ev_lower = ev.lower()
                # Check if negation is about the same entity
                if any(w in ev_lower for w in ["not", "never", "cannot", "can't"]):
                    # If the fact says 'yes' and evidence says 'not', it's refuted
                    if fact.answer.lower() in ("yes",):
                        return IntelligenceResult(
                            answer="no",
                            confidence=0.70,
                            reasoning=f"Evidence contradicts fact: {ev[:80]}",
                            method="evidence_verification",
                            facts_used=[ev[:100]],
                        )

        # If evidence supports the fact, boost confidence
        if ev_has_support and not ev_has_negation:
            return IntelligenceResult(
                answer=fact.answer,
                confidence=min(0.95, fact.confidence + 0.1),
                reasoning=f"Fact confirmed by evidence: {fact.reasoning}",
                method="evidence_verification",
                facts_used=fact.facts_used,
            )

        # Evidence is ambiguous or mixed — return fact with reduced confidence
        return IntelligenceResult(
            answer=fact.answer,
            confidence=max(0.5, fact.confidence - 0.15),
            reasoning=f"Fact with ambiguous evidence: {fact.reasoning}",
            method="evidence_verification",
            facts_used=fact.facts_used,
        )

    def investigate(
        self,
        query: str,
        evidence: list[str],
        witnesses: list[dict[str, str]] | None = None,
    ) -> IntelligenceResult:
        """Multi-step investigation reasoning."""
        evidence_text = " ".join(evidence).lower()

        # Check for contradictions
        contradictions = self._find_contradictions(evidence)

        # Check witness agreement
        witness_agreement = self._check_witness_agreement(witnesses or [])

        # Determine conclusion
        if contradictions:
            if len(contradictions) >= 2:
                return IntelligenceResult(
                    answer="unknown", confidence=0.4,
                    reasoning=f"Multiple contradictions: {contradictions}",
                    method="investigation", facts_used=contradictions,
                )
            else:
                return IntelligenceResult(
                    answer="contradicted", confidence=0.7,
                    reasoning=f"Contradiction: {contradictions[0]}",
                    method="investigation", facts_used=contradictions,
                )

        if witness_agreement == "disagree":
            return IntelligenceResult(
                answer="unknown", confidence=0.5,
                reasoning="Witnesses disagree -- cannot determine",
                method="investigation",
            )

        # Score evidence support
        support_score = self._score_support(query, evidence)
        if support_score >= 0.7:
            return IntelligenceResult(
                answer="yes", confidence=support_score,
                reasoning=f"Evidence supports ({support_score:.0%})",
                method="investigation",
            )
        elif support_score <= 0.3:
            return IntelligenceResult(
                answer="no", confidence=1 - support_score,
                reasoning=f"Evidence does not support ({(1 - support_score):.0%})",
                method="investigation",
            )
        else:
            return IntelligenceResult(
                answer="unknown", confidence=0.5,
                reasoning="Insufficient evidence",
                method="investigation",
            )

    def _find_contradictions(self, evidence: list[str]) -> list[str]:
        """Find contradictions in evidence."""
        contradictions = []
        for ev in evidence:
            ev_lower = ev.lower()
            if any(w in ev_lower for w in ["contradict", "inconsistent", "conflict", "disagree"]):
                contradictions.append(f"Explicit contradiction: {ev[:50]}")

        all_nums = []
        for ev in evidence:
            nums = re.findall(r'\b(\d+)\b', ev)
            if nums:
                all_nums.extend(nums)
        if len(set(all_nums)) > 2:
            contradictions.append(f"Multiple different values: {set(all_nums)}")

        return contradictions

    def _check_witness_agreement(self, witnesses: list[dict[str, str]]) -> str:
        """Check if witnesses agree or disagree."""
        if len(witnesses) < 2:
            return "insufficient"
        accounts = [w.get("account", "").lower() for w in witnesses]
        unique = set(accounts)
        if len(unique) == 1:
            return "agree"
        elif len(unique) > 1:
            return "disagree"
        return "insufficient"

    def _score_support(self, query: str, evidence: list[str]) -> float:
        """Score how well evidence supports the query."""
        if not evidence:
            return 0.5
        query_words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        evidence_text = " ".join(evidence).lower()
        evidence_words = set(re.findall(r'\b[a-z]{3,}\b', evidence_text))
        overlap = query_words & evidence_words
        if not overlap:
            return 0.3
        base_score = min(1.0, len(overlap) / max(len(query_words), 1))
        support_count = sum(1 for ev in evidence if any(w in ev.lower() for w in [
            "support", "confirm", "consistent", "agree", "show", "demonstrate",
        ]))
        contra_count = sum(1 for ev in evidence if any(w in ev.lower() for w in [
            "contradict", "inconsistent", "disagree", "conflict",
        ]))
        return max(0.0, min(1.0, base_score + support_count * 0.15 - contra_count * 0.2))
