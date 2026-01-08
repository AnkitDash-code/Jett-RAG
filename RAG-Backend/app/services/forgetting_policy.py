"""
Forgetting Policies for memory management.
Defines strategies for memory decay, archival, and deletion.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple
import math

from app.models.memory import Memory, MemoryType, MemoryStatus


class ForgettingPolicy(ABC):
    """
    Abstract base class for forgetting policies.
    
    Defines the interface for memory decay calculations,
    archival decisions, and deletion decisions.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Policy name for logging/config."""
        pass
    
    @abstractmethod
    def decay_function(self, memory: Memory) -> float:
        """
        Calculate the decayed importance of a memory.
        
        Args:
            memory: The memory to evaluate
            
        Returns:
            Decayed importance score (0.0 to 1.0+)
        """
        pass
    
    @abstractmethod
    def should_archive(self, memory: Memory) -> bool:
        """
        Determine if memory should be archived (removed from cache).
        
        Args:
            memory: The memory to evaluate
            
        Returns:
            True if memory should be archived
        """
        pass
    
    @abstractmethod
    def should_delete(self, memory: Memory) -> bool:
        """
        Determine if memory should be permanently deleted.
        
        Args:
            memory: The memory to evaluate
            
        Returns:
            True if memory should be deleted
        """
        pass
    
    def apply_to_memories(
        self, 
        memories: List[Memory]
    ) -> Tuple[List[Memory], List[Memory], List[Memory]]:
        """
        Apply policy to a list of memories.
        
        Args:
            memories: List of memories to evaluate
            
        Returns:
            Tuple of (keep, archive, delete) memory lists
        """
        keep = []
        archive = []
        delete = []
        
        for memory in memories:
            if self.should_delete(memory):
                delete.append(memory)
            elif self.should_archive(memory):
                archive.append(memory)
            else:
                keep.append(memory)
        
        return keep, archive, delete


class AggressivePolicy(ForgettingPolicy):
    """
    Aggressive forgetting policy.
    
    Quickly decays and archives memories. Good for systems with
    limited storage or where recency is paramount.
    
    - Decay: 0.8^days_old
    - Archive: decayed < 0.3 AND days_old > 7
    - Delete: days_old > 60 AND importance < 0.1
    """
    
    @property
    def name(self) -> str:
        return "aggressive"
    
    def decay_function(self, memory: Memory) -> float:
        """Fast exponential decay."""
        base_importance = memory.importance_score
        days = max(memory.days_old, 0)
        return base_importance * (0.8 ** days)
    
    def should_archive(self, memory: Memory) -> bool:
        """Archive quickly - after 7 days if importance drops."""
        if memory.status == MemoryStatus.ARCHIVED:
            return False
        
        decayed = self.decay_function(memory)
        return decayed < 0.3 and memory.days_old > 7
    
    def should_delete(self, memory: Memory) -> bool:
        """Delete after 60 days if very low importance."""
        return memory.days_old > 60 and memory.importance_score < 0.1


class ConservativePolicy(ForgettingPolicy):
    """
    Conservative forgetting policy.
    
    Preserves memories longer, slower decay. Good for compliance
    or when historical context is valuable.
    
    - Decay: 0.95^days_old
    - Archive: decayed < 0.2 AND days_old > 30
    - Delete: Never (archival only)
    """
    
    @property
    def name(self) -> str:
        return "conservative"
    
    def decay_function(self, memory: Memory) -> float:
        """Slow exponential decay."""
        base_importance = memory.importance_score
        days = max(memory.days_old, 0)
        return base_importance * (0.95 ** days)
    
    def should_archive(self, memory: Memory) -> bool:
        """Archive slowly - after 30 days if importance very low."""
        if memory.status == MemoryStatus.ARCHIVED:
            return False
        
        decayed = self.decay_function(memory)
        return decayed < 0.2 and memory.days_old > 30
    
    def should_delete(self, memory: Memory) -> bool:
        """Never delete - archive only."""
        return False


class BalancedPolicy(ForgettingPolicy):
    """
    Balanced forgetting policy (default).
    
    Moderate decay with reasonable retention periods.
    
    - Decay: 0.9^days_old
    - Archive: decayed < 0.25 AND days_old > 14
    - Delete: days_old > 90 AND importance < 0.05
    """
    
    @property
    def name(self) -> str:
        return "balanced"
    
    def decay_function(self, memory: Memory) -> float:
        """Moderate exponential decay."""
        base_importance = memory.importance_score
        days = max(memory.days_old, 0)
        return base_importance * (0.9 ** days)
    
    def should_archive(self, memory: Memory) -> bool:
        """Archive after 2 weeks if importance drops significantly."""
        if memory.status == MemoryStatus.ARCHIVED:
            return False
        
        decayed = self.decay_function(memory)
        return decayed < 0.25 and memory.days_old > 14
    
    def should_delete(self, memory: Memory) -> bool:
        """Delete after 90 days if negligible importance."""
        return memory.days_old > 90 and memory.importance_score < 0.05


class TypeAwarePolicy(ForgettingPolicy):
    """
    Type-aware forgetting policy.
    
    Applies different rules for episodic vs semantic memories.
    Episodic memories decay faster; semantic memories are preserved.
    
    Episodic:
    - Decay: 0.85^days_old
    - Archive: decayed < 0.3 AND days_old > 7
    - Delete: days_old > 60
    
    Semantic:
    - Decay: 0.98^days_old
    - Archive: decayed < 0.15 AND days_old > 60
    - Delete: Never
    """
    
    @property
    def name(self) -> str:
        return "type_aware"
    
    def decay_function(self, memory: Memory) -> float:
        """Type-dependent decay."""
        base_importance = memory.importance_score
        days = max(memory.days_old, 0)
        
        if memory.memory_type == MemoryType.EPISODIC:
            return base_importance * (0.85 ** days)
        else:
            return base_importance * (0.98 ** days)
    
    def should_archive(self, memory: Memory) -> bool:
        """Type-dependent archival."""
        if memory.status == MemoryStatus.ARCHIVED:
            return False
        
        decayed = self.decay_function(memory)
        
        if memory.memory_type == MemoryType.EPISODIC:
            return decayed < 0.3 and memory.days_old > 7
        else:
            return decayed < 0.15 and memory.days_old > 60
    
    def should_delete(self, memory: Memory) -> bool:
        """Delete episodic after 60 days, never delete semantic."""
        if memory.memory_type == MemoryType.EPISODIC:
            return memory.days_old > 60 and memory.importance_score < 0.1
        return False


class AdaptivePolicy(ForgettingPolicy):
    """
    Adaptive forgetting policy.
    
    Adjusts decay based on interaction patterns.
    Frequently accessed memories decay slower.
    
    - Decay: base * (0.9^days) * (1 + 0.1 * log(access_count + 1))
    - Archive: decayed < 0.2 AND days_old > 21
    - Delete: days_old > 120 AND interaction_count == 0
    """
    
    @property
    def name(self) -> str:
        return "adaptive"
    
    def decay_function(self, memory: Memory) -> float:
        """Adaptive decay based on access patterns."""
        base_importance = memory.importance_score
        days = max(memory.days_old, 0)
        access_boost = 1 + 0.1 * math.log(memory.access_count + 1)
        
        return base_importance * (0.9 ** days) * access_boost
    
    def should_archive(self, memory: Memory) -> bool:
        """Archive based on decayed importance."""
        if memory.status == MemoryStatus.ARCHIVED:
            return False
        
        decayed = self.decay_function(memory)
        return decayed < 0.2 and memory.days_old > 21
    
    def should_delete(self, memory: Memory) -> bool:
        """Delete if old and never accessed."""
        return memory.days_old > 120 and memory.access_count == 0


# Policy registry
POLICIES = {
    "aggressive": AggressivePolicy,
    "conservative": ConservativePolicy,
    "balanced": BalancedPolicy,
    "type_aware": TypeAwarePolicy,
    "adaptive": AdaptivePolicy,
}


def get_policy(name: str = "balanced") -> ForgettingPolicy:
    """
    Get a forgetting policy by name.
    
    Args:
        name: Policy name (aggressive, conservative, balanced, type_aware, adaptive)
        
    Returns:
        ForgettingPolicy instance
    """
    policy_class = POLICIES.get(name.lower(), BalancedPolicy)
    return policy_class()
