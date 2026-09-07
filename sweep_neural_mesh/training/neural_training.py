"""
Neural Training Pipeline — Sweep Proper Neural Mesh.

Fine-tunes BERT-base for:
1. Evidence classification (supports/refutes/neutral)
2. Contradiction detection (contradiction/consistent/neutral)
3. Intent classification (17 categories)

CPU-only operation. Uses small batch sizes for memory efficiency.

Usage:
    python -m sweep_neural_mesh.training.neural_training
"""
from __future__ import annotations

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Any

_sweep_dir = Path(__file__).resolve().parent.parent
_sweep_parent = _sweep_dir.parent
sys.path.insert(0, str(_sweep_parent))
sys.path.insert(0, str(_sweep_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("neural_training")


# ════════════════════════════════════════════════════════════════
# TRAINING DATA — Real examples, not templates
# ════════════════════════════════════════════════════════════════

EVIDENCE_TRAINING_DATA = [
    # SUPPORTS (label 0)
    ("Studies confirm that regular exercise reduces cardiovascular disease risk by 30%", 0),
    ("Meta-analysis of 50 trials shows the vaccine is 95% effective", 0),
    ("Peer-reviewed research demonstrates that reading improves cognitive function", 0),
    ("Longitudinal data shows the treatment group outperformed controls significantly", 0),
    ("Multiple independent studies confirm the relationship between sleep and performance", 0),
    ("Clinical trial results show statistically significant improvement in the treatment group", 0),
    ("Expert consensus across 12 institutions supports this conclusion", 0),
    ("The evidence from 3 independent research teams all points to the same result", 0),
    ("Government data confirms the economic growth trend over the past 5 years", 0),
    ("Systematic review of 200 studies confirms the safety profile", 0),
    ("A peer-reviewed study in Nature demonstrated the mechanism clearly", 0),
    ("Federal health agencies officially recommend this approach", 0),
    ("The FDA approved the drug after rigorous testing phases", 0),
    ("University researchers replicated the original findings across 3 labs", 0),
    ("The WHO guidelines explicitly support this intervention", 0),
    ("Randomized controlled trial showed 40% reduction in symptoms", 0),
    ("Cohort study of 10,000 participants confirmed the association", 0),
    ("Double-blind study demonstrated significant improvement over placebo", 0),
    ("Network meta-analysis ranked this treatment as most effective", 0),
    ("Real-world evidence from 50,000 patients supports efficacy", 0),
    ("The intervention showed consistent benefits across all age groups", 0),
    ("Biomarker analysis confirmed the biological mechanism", 0),
    ("Cost-effectiveness analysis showed favorable outcomes", 0),
    ("Long-term follow-up data confirmed sustained benefits", 0),
    ("Multiple endpoints showed significant improvement", 0),
    
    # REFUTES (label 1)
    ("The drug showed no significant effect compared to placebo in the trial", 1),
    ("Studies found no evidence supporting the claimed health benefits", 1),
    ("The experiment failed to demonstrate any measurable improvement", 1),
    ("Research contradicts the widely held belief about this topic", 1),
    ("The data shows the intervention had zero measurable impact", 1),
    ("Meta-analysis found no statistically significant benefit across all 15 studies", 1),
    ("Controlled trials showed the treatment group performed worse than expected", 1),
    ("The hypothesis was definitively refuted by the experimental evidence", 1),
    ("Independent verification found the original results could not be replicated", 1),
    ("The government report found the policy had no positive effect", 1),
    ("Results were not statistically significant in any of the 8 sub-studies", 1),
    ("The study found the opposite of what was hypothesized", 1),
    ("Three separate labs failed to reproduce the original findings", 1),
    ("The WHO stated the evidence does not support this treatment", 1),
    ("FDA analysis found the drug's benefits did not outweigh its risks", 1),
    ("The trial was stopped early due to lack of efficacy", 1),
    ("No difference was observed between treatment and control groups", 1),
    ("The intervention group had higher mortality rates", 1),
    ("Systematic review found inconsistent and negative results", 1),
    ("Post-market surveillance revealed no clinical benefit", 1),
    ("The proposed mechanism was not supported by experimental data", 1),
    ("Epidemiological data contradicts the causal hypothesis", 1),
    ("The treatment arm showed no improvement over standard care", 1),
    ("Biomarker levels were unchanged despite treatment", 1),
    ("Quality of life scores were identical between groups", 1),
    
    # NEUTRAL (label 2)
    ("Results were mixed across different populations and demographics", 2),
    ("Some participants improved while others showed no change", 2),
    ("The effect size was small and borderline significant", 2),
    ("More research is needed to confirm these preliminary findings", 2),
    ("Evidence is insufficient to draw a definitive conclusion", 2),
    ("The results were inconsistent across different study designs", 2),
    ("The study had important limitations that affect interpretation", 2),
    ("Additional large-scale trials are warranted before conclusions", 2),
    ("The effect was observed only in certain subgroups", 2),
    ("Preliminary data suggests a trend but lacks statistical power", 2),
    ("The findings depend heavily on the specific methodology used", 2),
    ("Results varied considerably depending on dosage and duration", 2),
    ("The study population may not be representative of the general public", 2),
    ("Some evidence supports while other evidence contradicts", 2),
    ("The relationship appears complex and not fully understood yet", 2),
    ("The results were inconclusive due to high dropout rates", 2),
    ("Effectiveness varied by geographic region", 2),
    ("Short-term benefits were observed but long-term effects unknown", 2),
    ("The study was underpowered to detect clinically meaningful differences", 2),
    ("Confounding factors may have influenced the results", 2),
    ("The intervention showed promise in early phases but results are preliminary", 2),
    ("Dose-response relationship was not clearly established", 2),
    ("Safety profile was acceptable but efficacy data is limited", 2),
    ("Results from animal studies did not translate to human outcomes", 2),
    ("The evidence is suggestive but not conclusive", 2),
]

CONTRADICTION_TRAINING_DATA = [
    # CONTRADICTION (label 0)
    ("The meeting is at 3 PM", "The meeting is at 4 PM", 0),
    ("The drug is effective", "The drug is ineffective", 0),
    ("All students passed the exam", "Some students failed the exam", 0),
    ("It is raining outside", "It is sunny and completely dry", 0),
    ("Revenue increased 15%", "Revenue decreased 15%", 0),
    ("The company is profitable", "The company reported significant losses", 0),
    ("Water boils at 100C", "Water boils at 90C at sea level", 0),
    ("The Earth is approximately round", "The Earth is completely flat", 0),
    ("Light travels faster than sound", "Sound travels faster than light", 0),
    ("Cats are mammals", "Cats are reptiles", 0),
    ("The product costs $50", "The product costs $500", 0),
    ("The event is on Monday", "The event is on Tuesday", 0),
    ("Paris is the capital of France", "Lyon is the capital of France", 0),
    ("The population is 1 million", "The population is 10 million", 0),
    ("The study supports the hypothesis", "The study contradicts the hypothesis", 0),
    ("The vaccine is safe", "The vaccine is not safe", 0),
    ("Climate change is real", "Climate change is not real", 0),
    ("AI will replace jobs", "AI will create jobs", 0),
    ("The system is online", "The system is offline", 0),
    ("The temperature rose to 30C", "The temperature dropped to 10C", 0),
    ("The company's revenue doubled", "The company's revenue halved", 0),
    ("Exercise improves health", "Exercise has no effect on health", 0),
    ("The treatment is beneficial", "The treatment is harmful", 0),
    ("The algorithm is fast", "The algorithm is very slow", 0),
    ("The experiment succeeded", "The experiment failed", 0),
    
    # CONSISTENT (label 1)
    ("The study found exercise improves health", "Research confirms physical activity benefits cardiovascular health", 1),
    ("Water boils at 100C at sea level", "The boiling point of water is 100 degrees Celsius", 1),
    ("Paris is the capital of France", "France's capital city is Paris", 1),
    ("The experiment showed positive results", "The trial demonstrated beneficial outcomes", 1),
    ("The company reported growth", "The organization announced expansion", 1),
    ("The medication reduces symptoms", "The drug alleviates clinical symptoms", 1),
    ("Climate change is accelerating", "Global warming trends are intensifying", 1),
    ("The algorithm runs in O(n log n)", "The algorithm has logarithmic linear time complexity", 1),
    ("The bridge is 500 meters long", "The 500m bridge spans the river", 1),
    ("Python is a programming language", "Python is used for software development", 1),
    ("The study confirmed the hypothesis", "Research validated the proposed theory", 1),
    ("The system is operational", "The system is online and functioning", 1),
    ("The project completed on time", "The project finished ahead of schedule", 1),
    ("The drug reduces inflammation", "The medication decreases inflammatory response", 1),
    ("The company is profitable", "The organization reported positive earnings", 1),
    ("The algorithm is efficient", "The method performs well computationally", 1),
    ("The treatment improved outcomes", "The intervention enhanced patient results", 1),
    ("The study was large-scale", "The research involved a substantial cohort", 1),
    ("The results were significant", "The findings reached statistical significance", 1),
    ("The drug is safe", "The medication has acceptable safety profile", 1),
    ("The intervention worked", "The treatment was effective", 1),
    ("The data supports the claim", "Evidence corroborates the assertion", 1),
    ("The hypothesis was confirmed", "The theory was validated", 1),
    ("The system works", "The system is functional", 1),
    ("The experiment was successful", "The trial achieved its endpoints", 1),
    
    # NEUTRAL (label 2)
    ("The study had mixed results across different populations", "Results varied by demographic group", 2),
    ("Some participants improved while others did not", "Effectiveness was inconsistent across subjects", 2),
    ("The effect size was small", "The magnitude of change was modest", 2),
    ("More research is needed", "Additional studies are warranted", 2),
    ("Evidence is insufficient", "Data is lacking for definitive conclusions", 2),
    ("The results were inconclusive", "Findings were not definitive", 2),
    ("The study had limitations", "The research had methodological constraints", 2),
    ("The effect was observed in some conditions", "Benefits were context-dependent", 2),
    ("Preliminary data suggests a trend", "Early results indicate a potential pattern", 2),
    ("The findings are complex", "The relationship is multifaceted", 2),
    ("Results were inconsistent", "Outcomes varied across studies", 2),
    ("The study was underpowered", "Sample size was insufficient", 2),
    ("Confounding factors may exist", "Potential confounders were identified", 2),
    ("Short-term effects were observed", "Immediate outcomes were noted", 2),
    ("The evidence is mixed", "Data presents contradictory signals", 2),
]

INTENT_TRAINING_DATA = [
    # investigation
    ("Investigate John Smith and his connections", 0),
    ("Find everything about Alice Chen's background", 0),
    ("Who is Bob Wilson and what do they do?", 0),
    ("Research Sarah Davis's professional history", 0),
    ("Look into TechCorp Inc's public records", 0),
    
    # search
    ("Search for information about quantum computing", 1),
    ("Find articles about climate change", 1),
    ("What's the latest news on AI?", 1),
    ("Look up machine learning on the web", 1),
    ("Research renewable energy across multiple sources", 1),
    
    # identity_analysis
    ("Is John Smith the same person as John Doe?", 2),
    ("Could Alice C be an alias for Alice Chen?", 2),
    ("Analyze the identity of Bob Wilson", 2),
    ("Verify if Sarah Davis matches the description", 2),
    ("Are these two profiles about the same person?", 2),
    
    # evidence_analysis
    ("What evidence supports climate change?", 3),
    ("Is there proof that the vaccine works?", 3),
    ("Evaluate the evidence for this claim", 3),
    ("What do the sources say about this topic?", 3),
    ("How strong is the evidence?", 3),
    
    # comparison
    ("Compare Python and JavaScript", 4),
    ("What are the differences between SQL and NoSQL?", 4),
    ("How does React differ from Vue?", 4),
    ("Compare the evidence for A vs B", 4),
    ("Which is more reliable: source A or source B?", 4),
    
    # timeline
    ("Create a timeline of World War II", 5),
    ("What happened first: the Moon landing or WWII?", 5),
    ("When did the Internet originate?", 5),
    ("Reconstruct the chronological order of events", 5),
    ("What events led to the French Revolution?", 5),
    
    # relationship_analysis
    ("What is the relationship between Google and Alphabet?", 6),
    ("How are these two companies connected?", 6),
    ("Map the relationships between these entities", 6),
    ("What connections exist between these organizations?", 6),
    ("Are these two people affiliated?", 6),
    
    # location_analysis
    ("Where is Google headquartered?", 7),
    ("What is the geographic relationship between NY and LA?", 7),
    ("How far is London from Paris?", 7),
    ("Map all locations associated with this company", 7),
    ("What locations are connected to this event?", 7),
    
    # source_verification
    ("Is this source reliable?", 8),
    ("Verify the credibility of this website", 8),
    ("Is this information trustworthy?", 8),
    ("What is the reliability of this report?", 8),
    ("Check if this is a primary or secondary source", 8),
    
    # contradiction_analysis
    ("Do these statements contradict each other?", 9),
    ("Is there a conflict between these claims?", 9),
    ("Are these sources consistent?", 9),
    ("Find contradictions in the evidence", 9),
    ("Do these two reports disagree?", 9),
    
    # summarization
    ("Summarize the evidence about this topic", 10),
    ("Give me a brief overview of quantum computing", 10),
    ("What are the key findings about climate change?", 10),
    ("Provide a summary of the investigation", 10),
    ("What are the main points?", 10),
    
    # extraction
    ("Extract all names mentioned in this text", 11),
    ("Find all dates in this document", 11),
    ("What organizations are mentioned?", 11),
    ("Extract all locations from this text", 11),
    ("Find all email addresses", 11),
    
    # unknown_ambiguous
    ("Tell me something", 12),
    ("Help me", 12),
    ("What do you think?", 12),
    ("I'm curious about stuff", 12),
    ("Can you help with this thing?", 12),
]


# ════════════════════════════════════════════════════════════════
# NEURAL MODELS
# ════════════════════════════════════════════════════════════════

class NeuralEvidenceClassifier:
    """Fine-tuned BERT for evidence classification."""
    
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.label_map = {0: "supports", 1: "refutes", 2: "neutral"}
        self._loaded = False
    
    def load(self):
        """Load model (lazy loading)."""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading fine-tuned evidence model from {self.model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            logger.info("Loading pre-trained BERT-base for evidence classification")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=3
            )
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.model.eval()
        self._loaded = True
    
    def classify(self, text: str) -> dict[str, Any]:
        """Classify evidence text."""
        import torch
        
        self.load()
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, 
            padding=True, max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return {
            "answer": self.label_map[pred],
            "confidence": confidence,
            "probabilities": {
                self.label_map[i]: probs[0][i].item() 
                for i in range(3)
            }
        }
    
    def train(self, training_data: list[tuple[str, int]], output_dir: str, 
              epochs: int = 3, batch_size: int = 8, lr: float = 2e-5):
        """Fine-tune on evidence classification data."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from torch.optim import AdamW
        
        logger.info(f"Training evidence classifier on {len(training_data)} examples")
        
        # Load pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=3
        )
        
        # Prepare data
        texts = [t for t, _ in training_data]
        labels = [l for _, l in training_data]
        
        encodings = self.tokenizer(
            texts, truncation=True, padding=True, 
            max_length=128, return_tensors="pt"
        )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], labels_tensor
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.model.train()
        
        t0 = time.perf_counter()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                input_ids, attention_mask, batch_labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = torch.argmax(outputs.logits, dim=1)
                correct += (pred == batch_labels).sum().item()
                total += len(batch_labels)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.1%}")
        
        elapsed = time.perf_counter() - t0
        
        # Save
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save metadata
        metadata = {
            "model": "bert-base-uncased-finetuned",
            "task": "evidence_classification",
            "labels": ["supports", "refutes", "neutral"],
            "training_examples": len(training_data),
            "epochs": epochs,
            "duration_seconds": elapsed,
            "status": "fine-tuned",
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Evidence classifier trained: {elapsed:.1f}s, saved to {output_dir}")
        
        self.model_path = output_dir
        self._loaded = True


class NeuralContradictionDetector:
    """Fine-tuned BERT for contradiction detection."""
    
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.label_map = {0: "contradiction", 1: "consistent", 2: "neutral"}
        self._loaded = False
    
    def load(self):
        """Load model (lazy loading)."""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading fine-tuned contradiction model from {self.model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            logger.info("Loading pre-trained BERT-base for contradiction detection")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=3
            )
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.model.eval()
        self._loaded = True
    
    def detect(self, text_a: str, text_b: str) -> dict[str, Any]:
        """Detect contradiction between two texts."""
        import torch
        
        self.load()
        
        # Encode pair
        inputs = self.tokenizer(
            text_a, text_b, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return {
            "answer": self.label_map[pred],
            "confidence": confidence,
            "probabilities": {
                self.label_map[i]: probs[0][i].item()
                for i in range(3)
            }
        }
    
    def train(self, training_data: list[tuple[str, str, int]], output_dir: str,
              epochs: int = 3, batch_size: int = 8, lr: float = 2e-5):
        """Fine-tune on contradiction detection data."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from torch.optim import AdamW
        
        logger.info(f"Training contradiction detector on {len(training_data)} examples")
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=3
        )
        
        texts_a = [t[0] for t in training_data]
        texts_b = [t[1] for t in training_data]
        labels = [t[2] for t in training_data]
        
        encodings = self.tokenizer(
            texts_a, texts_b, truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], labels_tensor
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.model.train()
        
        t0 = time.perf_counter()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                input_ids, attention_mask, batch_labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = torch.argmax(outputs.logits, dim=1)
                correct += (pred == batch_labels).sum().item()
                total += len(batch_labels)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.1%}")
        
        elapsed = time.perf_counter() - t0
        
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        metadata = {
            "model": "bert-base-uncased-finetuned",
            "task": "contradiction_detection",
            "labels": ["contradiction", "consistent", "neutral"],
            "training_examples": len(training_data),
            "epochs": epochs,
            "duration_seconds": elapsed,
            "status": "fine-tuned",
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Contradiction detector trained: {elapsed:.1f}s, saved to {output_dir}")
        
        self.model_path = output_dir
        self._loaded = True


class NeuralIntentClassifier:
    """Fine-tuned BERT for intent classification."""
    
    def __init__(self, model_path: str | None = None):
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.label_map = {
            0: "investigation", 1: "search", 2: "identity_analysis",
            3: "evidence_analysis", 4: "comparison", 5: "timeline",
            6: "relationship_analysis", 7: "location_analysis",
            8: "source_verification", 9: "contradiction_analysis",
            10: "summarization", 11: "extraction", 12: "unknown_ambiguous",
        }
        self._loaded = False
    
    def load(self):
        """Load model (lazy loading)."""
        if self._loaded:
            return
        
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        num_labels = len(self.label_map)
        
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading fine-tuned intent model from {self.model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        else:
            logger.info("Loading pre-trained BERT-base for intent classification")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=num_labels
            )
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.model.eval()
        self._loaded = True
    
    def classify(self, text: str) -> dict[str, Any]:
        """Classify intent of a query."""
        import torch
        
        self.load()
        
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            padding=True, max_length=128
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
        
        return {
            "answer": self.label_map.get(pred, "unknown"),
            "confidence": confidence,
            "probabilities": {
                self.label_map.get(i, f"class_{i}"): probs[0][i].item()
                for i in range(len(self.label_map))
            }
        }
    
    def train(self, training_data: list[tuple[str, int]], output_dir: str,
              epochs: int = 3, batch_size: int = 8, lr: float = 2e-5):
        """Fine-tune on intent classification data."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from torch.optim import AdamW
        
        num_labels = len(self.label_map)
        logger.info(f"Training intent classifier on {len(training_data)} examples ({num_labels} classes)")
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=num_labels
        )
        
        texts = [t for t, _ in training_data]
        labels = [l for _, l in training_data]
        
        encodings = self.tokenizer(
            texts, truncation=True, padding=True,
            max_length=128, return_tensors="pt"
        )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        dataset = TensorDataset(
            encodings["input_ids"], encodings["attention_mask"], labels_tensor
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.model.train()
        
        t0 = time.perf_counter()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in dataloader:
                input_ids, attention_mask, batch_labels = batch
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.logits, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = torch.argmax(outputs.logits, dim=1)
                correct += (pred == batch_labels).sum().item()
                total += len(batch_labels)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            logger.info(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.1%}")
        
        elapsed = time.perf_counter() - t0
        
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        metadata = {
            "model": "bert-base-uncased-finetuned",
            "task": "intent_classification",
            "num_labels": num_labels,
            "labels": self.label_map,
            "training_examples": len(training_data),
            "epochs": epochs,
            "duration_seconds": elapsed,
            "status": "fine-tuned",
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Intent classifier trained: {elapsed:.1f}s, saved to {output_dir}")
        
        self.model_path = output_dir
        self._loaded = True


# ════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ════════════════════════════════════════════════════════════════

def main():
    """Train all neural models."""
    print("=" * 70)
    print("SWEEP NEURAL TRAINING — Proper Neural Mesh")
    print("=" * 70)
    
    t0 = time.perf_counter()
    output_base = Path(__file__).parent / "neural_models"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Train evidence classifier
    print("\n[1/3] Training Evidence Classifier...")
    evidence_model = NeuralEvidenceClassifier()
    evidence_model.train(
        EVIDENCE_TRAINING_DATA,
        str(output_base / "evidence_classifier"),
        epochs=3, batch_size=8, lr=2e-5
    )
    
    # Train contradiction detector
    print("\n[2/3] Training Contradiction Detector...")
    contradiction_model = NeuralContradictionDetector()
    contradiction_model.train(
        CONTRADICTION_TRAINING_DATA,
        str(output_base / "contradiction_detector"),
        epochs=3, batch_size=8, lr=2e-5
    )
    
    # Train intent classifier
    print("\n[3/3] Training Intent Classifier...")
    intent_model = NeuralIntentClassifier()
    intent_model.train(
        INTENT_TRAINING_DATA,
        str(output_base / "intent_classifier"),
        epochs=3, batch_size=8, lr=2e-5
    )
    
    elapsed = time.perf_counter() - t0
    
    # Test the models
    print("\n" + "=" * 70)
    print("TESTING NEURAL MODELS")
    print("=" * 70)
    
    # Test evidence
    print("\nEvidence Classification:")
    evidence_model = NeuralEvidenceClassifier(str(output_base / "evidence_classifier"))
    test_evidence = [
        ("Studies confirm the drug is effective", "supports"),
        ("The drug shows no significant effect", "refutes"),
        ("Results were mixed across populations", "neutral"),
    ]
    for text, expected in test_evidence:
        result = evidence_model.classify(text)
        status = "[OK]" if result["answer"] == expected else "[FAIL]"
        print(f"  {status} '{text[:50]}...' -> {result['answer']} (conf={result['confidence']:.2f})")
    
    # Test contradiction
    print("\nContradiction Detection:")
    contradiction_model = NeuralContradictionDetector(str(output_base / "contradiction_detector"))
    test_contradictions = [
        ("The drug is effective", "The drug is ineffective", "contradiction"),
        ("Paris is the capital of France", "France's capital is Paris", "consistent"),
        ("Results were mixed", "Findings were inconclusive", "neutral"),
    ]
    for text_a, text_b, expected in test_contradictions:
        result = contradiction_model.detect(text_a, text_b)
        status = "[OK]" if result["answer"] == expected else "[FAIL]"
        print(f"  {status} '{text_a[:30]}...' vs '{text_b[:30]}...' -> {result['answer']} (conf={result['confidence']:.2f})")
    
    # Test intent
    print("\nIntent Classification:")
    intent_model = NeuralIntentClassifier(str(output_base / "intent_classifier"))
    test_intents = [
        ("Investigate John Smith", "investigation"),
        ("Search for quantum computing", "search"),
        ("Do these statements contradict?", "contradiction_analysis"),
    ]
    for text, expected in test_intents:
        result = intent_model.classify(text)
        status = "[OK]" if result["answer"] == expected else "[FAIL]"
        print(f"  {status} '{text}' -> {result['answer']} (conf={result['confidence']:.2f})")
    
    print(f"\nTotal training time: {elapsed:.1f}s")
    print(f"Models saved to: {output_base}")
    
    # Save summary
    summary = {
        "training_time_seconds": elapsed,
        "models": {
            "evidence_classifier": {
                "path": str(output_base / "evidence_classifier"),
                "training_examples": len(EVIDENCE_TRAINING_DATA),
                "labels": ["supports", "refutes", "neutral"],
            },
            "contradiction_detector": {
                "path": str(output_base / "contradiction_detector"),
                "training_examples": len(CONTRADICTION_TRAINING_DATA),
                "labels": ["contradiction", "consistent", "neutral"],
            },
            "intent_classifier": {
                "path": str(output_base / "intent_classifier"),
                "training_examples": len(INTENT_TRAINING_DATA),
                "num_labels": 13,
            },
        },
        "status": "trained",
    }
    
    with open(output_base / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_base / 'training_summary.json'}")


if __name__ == "__main__":
    main()
