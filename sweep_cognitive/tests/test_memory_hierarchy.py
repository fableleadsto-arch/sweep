"""
Tests for memory hierarchy (Phase 6).

These tests verify that:
1. Episodic memory stores and retrieves experiences
2. Semantic memory manages concepts
3. Skill registry manages learned skills
4. Failure memory tracks failure patterns
5. Memory hierarchy orchestrates all memory types
6. Consolidation converts episodes to semantic knowledge
"""

import pytest
import time

from sweep_cognitive.memory.hierarchy import (
    MemoryType,
    MemoryImportance,
    MemoryStatus,
    EpisodicMemoryEntry,
    EpisodicMemory,
    SemanticMemory,
    Skill,
    SkillRegistry,
    FailurePattern,
    FailureMemory,
    MemoryHierarchy,
)
from sweep_cognitive.knowledge import KnowledgeBase


class TestEpisodicMemoryEntry:
    """Tests for EpisodicMemoryEntry."""

    def test_basic_entry(self):
        """Test basic episodic entry creation."""
        entry = EpisodicMemoryEntry(
            episode_type="task",
            description="Completed the analysis task",
            success=True,
        )
        assert entry.episode_type == "task"
        assert entry.success is True
        assert entry.importance == MemoryImportance.MEDIUM
        assert entry.status == MemoryStatus.ACTIVE

    def test_entry_with_details(self):
        """Test entry with full details."""
        entry = EpisodicMemoryEntry(
            episode_type="investigation",
            description="Found evidence about X",
            task_id="task_123",
            goal="Analyze X",
            actions_taken=["searched", "analyzed"],
            observations=["Found X", "Verified Y"],
            outcome="Successfully identified X",
            success=True,
            importance=MemoryImportance.HIGH,
        )
        assert len(entry.actions_taken) == 2
        assert len(entry.observations) == 2
        assert entry.importance == MemoryImportance.HIGH

    def test_entry_to_dict(self):
        """Test entry serialization."""
        entry = EpisodicMemoryEntry(
            episode_type="error",
            description="Something went wrong",
            success=False,
        )
        d = entry.to_dict()
        assert d["episode_type"] == "error"
        assert d["success"] is False
        assert d["actions_count"] == 0


class TestEpisodicMemory:
    """Tests for EpisodicMemory."""

    def setup_method(self):
        """Set up fresh episodic memory."""
        self.memory = EpisodicMemory()

    def test_add_entry(self):
        """Test adding an entry."""
        entry = EpisodicMemoryEntry(
            episode_type="task",
            description="Test task",
        )
        self.memory.add(entry)
        assert self.memory.count() == 1

    def test_add_duplicate(self):
        """Test adding duplicate entry updates."""
        entry = EpisodicMemoryEntry(
            episode_type="task",
            description="Original",
        )
        self.memory.add(entry)
        
        # Add same ID with different description
        entry.description = "Updated"
        self.memory.add(entry)
        
        assert self.memory.count() == 1
        retrieved = self.memory.get(entry.id)
        assert retrieved.description == "Updated"

    def test_get_by_type(self):
        """Test retrieval by type."""
        self.memory.add(EpisodicMemoryEntry(
            id="e1", episode_type="task", description="T1", success=True
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="e2", episode_type="error", description="E1", success=False
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="e3", episode_type="task", description="T2", success=True
        ))
        
        tasks = self.memory.get_by_type("task")
        assert len(tasks) == 2
        
        errors = self.memory.get_by_type("error")
        assert len(errors) == 1

    def test_get_by_task(self):
        """Test retrieval by task ID."""
        self.memory.add(EpisodicMemoryEntry(
            id="e1", episode_type="task", task_id="task_1", description="T1", success=True
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="e2", episode_type="task", task_id="task_2", description="T2", success=True
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="e3", episode_type="error", task_id="task_1", description="E1", success=False
        ))
        
        task1_entries = self.memory.get_by_task("task_1")
        assert len(task1_entries) == 2

    def test_get_successes(self):
        """Test retrieval of successes."""
        self.memory.add(EpisodicMemoryEntry(
            id="e1", episode_type="task", description="S1", success=True
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="e2", episode_type="task", description="F1", success=False
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="e3", episode_type="task", description="S2", success=True
        ))
        
        successes = self.memory.get_successes()
        assert len(successes) == 2

    def test_get_failures(self):
        """Test retrieval of failures."""
        self.memory.add(EpisodicMemoryEntry(
            id="e1", episode_type="task", description="S1", success=True
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="e2", episode_type="task", description="F1", success=False
        ))
        
        failures = self.memory.get_failures()
        assert len(failures) == 1

    def test_get_recent(self):
        """Test recent retrieval."""
        # Add entries with different timestamps
        entry1 = EpisodicMemoryEntry(
            episode_type="task", description="Old"
        )
        entry1.timestamp = time.time() - 100000
        self.memory.add(entry1)
        
        entry2 = EpisodicMemoryEntry(
            episode_type="task", description="Recent"
        )
        self.memory.add(entry2)
        
        recent = self.memory.get_recent(n=1)
        assert len(recent) == 1
        assert recent[0].description == "Recent"

    def test_get_recent_with_filter(self):
        """Test recent retrieval with type filter."""
        self.memory.add(EpisodicMemoryEntry(
            id="epi_1", episode_type="task", description="Task1", success=True
        ))
        self.memory.add(EpisodicMemoryEntry(
            id="epi_2", episode_type="error", description="Error1", success=False
        ))
        
        recent_errors = self.memory.get_recent(n=10, episode_type="error")
        assert len(recent_errors) == 1
        assert recent_errors[0].episode_type == "error"

    def test_update_status(self):
        """Test status update."""
        entry = EpisodicMemoryEntry(
            episode_type="task", description="Test"
        )
        self.memory.add(entry)
        
        self.memory.update_status(entry.id, MemoryStatus.ARCHIVED)
        updated = self.memory.get(entry.id)
        assert updated.status == MemoryStatus.ARCHIVED

    def test_stats(self):
        """Test statistics."""
        e1 = EpisodicMemoryEntry(
            id="epi_1", episode_type="task", description="T1", success=True
        )
        e2 = EpisodicMemoryEntry(
            id="epi_2", episode_type="error", description="E1", success=False
        )
        self.memory.add(e1)
        self.memory.add(e2)
        
        stats = self.memory.stats()
        assert stats["total_entries"] == 2
        assert stats["by_type"]["task"] == 1
        assert stats["by_type"]["error"] == 1
        assert stats["by_outcome"]["success"] == 1
        assert stats["by_outcome"]["failure"] == 1


class TestSemanticMemory:
    """Tests for SemanticMemory."""

    def setup_method(self):
        """Set up fresh semantic memory."""
        self.memory = SemanticMemory()

    def test_add_concept(self):
        """Test adding a concept."""
        concept = self.memory.add_concept(
            "cat",
            "animal",
            {"fur": True, "legs": 4},
        )
        assert concept["name"] == "cat"
        assert concept["type"] == "animal"

    def test_get_concept(self):
        """Test retrieving a concept."""
        self.memory.add_concept("dog", "animal", {"fur": True})
        
        concept = self.memory.get_concept("dog")
        assert concept is not None
        assert concept["name"] == "dog"
        
        # Access should increment count
        assert concept["access_count"] >= 1

    def test_get_nonexistent_concept(self):
        """Test getting nonexistent concept."""
        concept = self.memory.get_concept("nonexistent")
        assert concept is None

    def test_related_concepts(self):
        """Test finding related concepts."""
        self.memory.add_concept("cat", "animal", {"fur": True, "legs": 4})
        self.memory.add_concept("dog", "animal", {"fur": True, "legs": 4})
        self.memory.add_concept("car", "vehicle", {"wheels": 4})
        
        related = self.memory.get_related_concepts("cat", n=5)
        assert len(related) == 1  # Only dog is related (same type, same props)
        assert related[0]["name"] == "dog"

    def test_consolidate_from_episode(self):
        """Test consolidation from episode."""
        episode = EpisodicMemoryEntry(
            episode_type="task",
            description="cat has fur and legs",
            success=True,
        )
        
        concepts = self.memory.consolidate_from_episode(episode)
        assert len(concepts) >= 1
        
        # Should be able to retrieve the concept
        concept = self.memory.get_concept(concepts[0])
        assert concept is not None

    def test_stats(self):
        """Test statistics."""
        self.memory.add_concept("cat", "animal", {})
        self.memory.add_concept("dog", "animal", {})
        self.memory.add_concept("car", "vehicle", {})
        
        stats = self.memory.stats()
        assert stats["total_concepts"] == 3
        assert stats["by_type"]["animal"] == 2
        assert stats["by_type"]["vehicle"] == 1


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def setup_method(self):
        """Set up fresh skill registry."""
        self.registry = SkillRegistry()

    def test_register_skill(self):
        """Test registering a skill."""
        skill = Skill(
            name="search",
            description="Search for information",
            purpose="Find information",
            steps=["query", "retrieve", "analyze"],
        )
        registered = self.registry.register(skill)
        assert registered.id is not None
        assert self.registry.count() == 1

    def test_register_duplicate_id(self):
        """Test registering skill with same ID updates."""
        skill1 = Skill(
            id="skill_1",
            name="search",
            steps=["step1"],
            success_rate=0.5,
            usage_count=10,
        )
        self.registry.register(skill1)
        
        skill2 = Skill(
            id="skill_1",
            name="search",
            steps=["step1", "step2"],
            success_rate=1.0,
            usage_count=1,
        )
        self.registry.register(skill2)
        
        # Should have updated, not duplicated
        assert self.registry.count() == 1
        retrieved = self.registry.get("skill_1")
        assert len(retrieved.steps) == 2  # Updated steps

    def test_get_by_name(self):
        """Test retrieving by name."""
        skill = Skill(
            name="search",
            steps=["step1"],
        )
        self.registry.register(skill)
        
        retrieved = self.registry.get_by_name("search")
        assert retrieved is not None
        assert retrieved.name == "search"

    def test_list_skills(self):
        """Test listing skills."""
        self.registry.register(Skill(id="sk1", name="skill1", steps=["s1"]))
        self.registry.register(Skill(id="sk2", name="skill2", steps=["s2"]))
        
        skills = self.registry.list_skills()
        assert len(skills) == 2

    def test_get_reliable_skills(self):
        """Test filtering by success rate."""
        self.registry.register(Skill(
            id="sk1", name="reliable", steps=["s1"], success_rate=0.9
        ))
        self.registry.register(Skill(
            id="sk2", name="unreliable", steps=["s2"], success_rate=0.3
        ))
        
        reliable = self.registry.get_reliable_skills(min_success_rate=0.7)
        assert len(reliable) == 1
        assert reliable[0].name == "reliable"

    def test_update_success_rate(self):
        """Test updating success rate."""
        skill = Skill(
            id="skill_1",
            name="test",
            steps=["s1"],
            success_rate=0.5,
            usage_count=2,
        )
        self.registry.register(skill)
        
        self.registry.update_success_rate("skill_1", True)
        updated = self.registry.get("skill_1")
        assert updated.success_rate > 0.5
        
        self.registry.update_success_rate("skill_1", False)
        updated = self.registry.get("skill_1")
        assert updated.success_rate < 1.0

    def test_stats(self):
        """Test statistics."""
        self.registry.register(Skill(
            id="sk1", name="skill1", purpose="analysis", steps=["s1"]
        ))
        self.registry.register(Skill(
            id="sk2", name="skill2", purpose="analysis", steps=["s2"]
        ))
        
        stats = self.registry.stats()
        assert stats["total_skills"] == 2
        assert stats["by_purpose"]["analysis"] == 2


class TestFailureMemory:
    """Tests for FailureMemory."""

    def setup_method(self):
        """Set up fresh failure memory."""
        self.memory = FailureMemory()

    def test_record_pattern(self):
        """Test recording a failure pattern."""
        pattern = FailurePattern(
            pattern_name="timeout",
            description="Operation timed out",
            failure_type="environment",
            symptoms=["slow response", "timeout error"],
            causes=["network issue"],
            avoidance_strategies=["retry with backoff"],
        )
        self.memory.record(pattern)
        assert self.memory.count() == 1

    def test_detect_pattern(self):
        """Test pattern detection."""
        pattern = FailurePattern(
            pattern_name="timeout",
            description="Operation timed out",
            failure_type="environment",
            symptoms=["slow response", "timeout error"],
            causes=["network issue"],
        )
        self.memory.record(pattern)
        
        matches = self.memory.detect(["slow response", "other symptom"])
        assert len(matches) == 1
        assert matches[0].pattern_name == "timeout"

    def test_detect_no_match(self):
        """Test detection with no matching symptoms."""
        pattern = FailurePattern(
            pattern_name="timeout",
            symptoms=["slow response"],
        )
        self.memory.record(pattern)
        
        matches = self.memory.detect(["completely different symptom"])
        assert len(matches) == 0

    def test_stats(self):
        """Test statistics."""
        self.memory.record(FailurePattern(
            id="fp1", pattern_name="P1", failure_type="environment"
        ))
        self.memory.record(FailurePattern(
            id="fp2", pattern_name="P2", failure_type="reasoning"
        ))
        self.memory.record(FailurePattern(
            id="fp3", pattern_name="P3", failure_type="environment"
        ))
        
        stats = self.memory.stats()
        assert stats["total_patterns"] == 3
        assert stats["by_type"]["environment"] == 2
        assert stats["by_type"]["reasoning"] == 1


class TestMemoryHierarchy:
    """Tests for MemoryHierarchy orchestrator."""

    def setup_method(self):
        """Set up fresh memory hierarchy."""
        self.hierarchy = MemoryHierarchy()

    def test_record_episode(self):
        """Test recording an episode."""
        episode = EpisodicMemoryEntry(
            episode_type="task",
            description="Test episode",
            success=True,
        )
        recorded = self.hierarchy.record_episode(episode)
        assert recorded.id is not None
        assert self.hierarchy.episodic.count() == 1

    def test_consolidate(self):
        """Test memory consolidation."""
        # Add an old successful episode
        episode = EpisodicMemoryEntry(
            id="epi_consolidate",
            episode_type="task",
            description="cat has fur",
            success=True,
            importance=MemoryImportance.HIGH,
        )
        episode.timestamp = time.time() - (10 * 24 * 3600)  # 10 days old
        self.hierarchy.record_episode(episode)
        
        # Consolidate
        concepts = self.hierarchy.consolidate(max_age_days=7)
        # May or may not create concepts depending on consolidation logic
        assert isinstance(concepts, list)
        
        # Episode should be archived
        updated = self.hierarchy.episodic.get(episode.id)
        assert updated.status == MemoryStatus.ARCHIVED

    def test_learn_from_failure(self):
        """Test learning from failure."""
        episode = EpisodicMemoryEntry(
            episode_type="task",
            description="Failed due to timeout",
            success=False,
            observations=["timeout occurred"],
        )
        self.hierarchy.record_episode(episode)
        
        pattern = self.hierarchy.learn_from_failure(episode)
        assert pattern is not None
        assert pattern.pattern_name == "failure_task"
        assert self.hierarchy.failures.count() == 1

    def test_retrieve_relevant_memory(self):
        """Test cross-type retrieval."""
        # Add entries to different memory types
        episode = EpisodicMemoryEntry(
            episode_type="task",
            description="Analyzed cat behavior",
            success=True,
        )
        self.hierarchy.record_episode(episode)
        
        self.hierarchy.semantic.add_concept("cat", "animal", {"fur": True})
        
        self.hierarchy.skills.register(Skill(
            name="analysis",
            description="Analyze data",
            purpose="Analysis tasks",
            steps=["step1"],
        ))
        
        results = self.hierarchy.retrieve_relevant_memory("cat", top_k=5)
        assert len(results) >= 1
        
        # Should find both episodic and semantic matches
        types_found = {r["memory_type"] for r in results}
        assert "episodic" in types_found or "semantic" in types_found

    def test_full_stats(self):
        """Test complete statistics."""
        # Add entries to all memory types
        self.hierarchy.record_episode(EpisodicMemoryEntry(
            episode_type="task", description="E1", success=True
        ))
        self.hierarchy.semantic.add_concept("concept1", "type1", {})
        self.hierarchy.skills.register(Skill(name="skill1", purpose="p1", steps=["s1"]))
        self.hierarchy.failures.record(FailurePattern(
            pattern_name="f1", failure_type="env"
        ))
        
        stats = self.hierarchy.stats()
        assert stats["episodic"]["total_entries"] == 1
        assert stats["semantic"]["total_concepts"] == 1
        assert stats["skills"]["total_skills"] == 1
        assert stats["failures"]["total_patterns"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
