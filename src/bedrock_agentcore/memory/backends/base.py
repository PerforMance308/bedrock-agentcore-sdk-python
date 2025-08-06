"""Abstract base class for memory backend implementations.

This module defines the interface contract that all memory backends must implement,
ensuring consistency across different storage implementations (AWS, PostgreSQL, ChromaDB).
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..constants import MemoryStatus, StrategyType


class MemoryBackendError(Exception):
    """Base exception for memory backend operations."""
    pass


class MemoryBackendType(Enum):
    """Supported memory backend types."""
    AWS_BEDROCK = "aws_bedrock"
    POSTGRESQL = "postgresql"
    CHROMADB = "chromadb"


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends.
    
    This class defines the interface contract that all concrete memory backends
    must implement, ensuring consistent behavior across different storage systems.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the memory backend.
        
        Args:
            config: Backend-specific configuration parameters
        """
        self.config = config or {}
        
    # ==================== CONTROL PLANE OPERATIONS ====================
    
    @abstractmethod
    async def create_memory(
        self,
        name: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        event_expiry_days: int = 90,
        memory_execution_role_arn: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new memory resource.
        
        Args:
            name: Memory resource name
            strategies: List of memory extraction strategies
            description: Optional description
            event_expiry_days: Days to retain events
            memory_execution_role_arn: Optional execution role ARN
            **kwargs: Backend-specific parameters
            
        Returns:
            Memory resource metadata with memoryId, status, etc.
            
        Raises:
            MemoryBackendError: If creation fails
        """
        pass
    
    @abstractmethod
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory resource metadata.
        
        Args:
            memory_id: Unique memory resource identifier
            
        Returns:
            Memory metadata dict or None if not found
        """
        pass
    
    @abstractmethod
    async def get_memory_status(self, memory_id: str) -> Optional[MemoryStatus]:
        """Get memory resource status.
        
        Args:
            memory_id: Unique memory resource identifier
            
        Returns:
            Current memory status or None if not found
        """
        pass
    
    @abstractmethod
    async def list_memories(
        self,
        max_results: int = 10,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List memory resources with pagination.
        
        Args:
            max_results: Maximum number of results to return
            next_token: Pagination token for continued listing
            
        Returns:
            Dict with 'memories' list and optional 'nextToken'
        """
        pass
    
    @abstractmethod
    async def update_memory(
        self,
        memory_id: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update memory resource configuration.
        
        Args:
            memory_id: Unique memory resource identifier
            strategies: Updated list of strategies
            description: Updated description
            **kwargs: Backend-specific parameters
            
        Returns:
            Updated memory metadata
        """
        pass
    
    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory resource.
        
        Args:
            memory_id: Unique memory resource identifier
            
        Returns:
            True if deletion was successful
        """
        pass
    
    # ==================== DATA PLANE OPERATIONS ====================
    
    @abstractmethod
    async def create_event(
        self,
        memory_id: str,
        actor_id: str,
        session_id: str,
        payload: List[Dict[str, Any]],
        branch_name: Optional[str] = None,
        root_event_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new conversation event.
        
        Args:
            memory_id: Target memory resource ID
            actor_id: Actor (user/agent) identifier
            session_id: Conversation session ID
            payload: List of message payloads
            branch_name: Optional conversation branch name
            root_event_id: Optional root event for branching
            **kwargs: Backend-specific parameters
            
        Returns:
            Event metadata with eventId, timestamp, etc.
        """
        pass
    
    @abstractmethod
    async def list_events(
        self,
        memory_id: str,
        actor_id: Optional[str] = None,
        session_id: Optional[str] = None,
        branch_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        max_results: int = 10,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List conversation events with filtering and pagination.
        
        Args:
            memory_id: Target memory resource ID
            actor_id: Optional actor filter
            session_id: Optional session filter
            branch_name: Optional branch filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            max_results: Maximum results to return
            next_token: Pagination token
            
        Returns:
            Dict with 'events' list and optional 'nextToken'
        """
        pass
    
    @abstractmethod
    async def get_event(
        self,
        memory_id: str,
        event_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific event by ID.
        
        Args:
            memory_id: Target memory resource ID
            event_id: Unique event identifier
            
        Returns:
            Event data or None if not found
        """
        pass
    
    # ==================== MEMORY RETRIEVAL OPERATIONS ====================
    
    @abstractmethod
    async def retrieve_memories(
        self,
        memory_id: str,
        namespace: str,
        query: str,
        top_k: int = 10,
        actor_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memory records using semantic search.
        
        Args:
            memory_id: Target memory resource ID
            namespace: Namespace path for scoping search
            query: Search query text
            top_k: Maximum number of results
            actor_id: Optional actor filter
            **kwargs: Backend-specific search parameters
            
        Returns:
            List of memory records with content and metadata
        """
        pass
    
    # ==================== CONVERSATION BRANCHING ====================
    
    @abstractmethod
    async def fork_conversation(
        self,
        memory_id: str,
        session_id: str,
        branch_name: str,
        root_event_id: str
    ) -> Dict[str, Any]:
        """Create a new conversation branch.
        
        Args:
            memory_id: Target memory resource ID
            session_id: Conversation session ID
            branch_name: Name for the new branch
            root_event_id: Event ID to branch from
            
        Returns:
            Branch metadata
        """
        pass
    
    @abstractmethod
    async def list_branches(
        self,
        memory_id: str,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """List all branches in a conversation session.
        
        Args:
            memory_id: Target memory resource ID
            session_id: Conversation session ID
            
        Returns:
            List of branch metadata
        """
        pass
    
    @abstractmethod
    async def get_conversation_tree(
        self,
        memory_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Get hierarchical conversation structure.
        
        Args:
            memory_id: Target memory resource ID
            session_id: Conversation session ID
            
        Returns:
            Tree structure with branches and events
        """
        pass
    
    # ==================== UTILITY METHODS ====================
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform backend health check.
        
        Returns:
            Health status information
        """
        pass
    
    @abstractmethod
    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend performance metrics.
        
        Returns:
            Performance and usage metrics
        """
        pass
    
    # ==================== HELPER METHODS ====================
    
    def generate_event_id(self) -> str:
        """Generate unique event ID."""
        return str(uuid.uuid4())
    
    def generate_memory_id(self) -> str:
        """Generate unique memory ID."""
        return str(uuid.uuid4())
    
    def validate_namespace(self, namespace: str) -> bool:
        """Validate namespace format."""
        if not namespace.startswith('/'):
            return False
        parts = namespace.strip('/').split('/')
        return len(parts) >= 2 and all(part for part in parts)
    
    def normalize_timestamp(self, timestamp: Union[datetime, str, int]) -> datetime:
        """Normalize timestamp to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        else:
            raise ValueError(f"Invalid timestamp format: {timestamp}")


class MemoryBackendFactory:
    """Factory for creating memory backend instances."""
    
    _backends: Dict[MemoryBackendType, type] = {}
    
    @classmethod
    def register_backend(cls, backend_type: MemoryBackendType, backend_class: type):
        """Register a memory backend implementation.
        
        Args:
            backend_type: Backend type enum
            backend_class: Backend implementation class
        """
        if not issubclass(backend_class, MemoryBackend):
            raise ValueError(f"Backend class must inherit from MemoryBackend")
        cls._backends[backend_type] = backend_class
    
    @classmethod
    def create_backend(
        cls,
        backend_type: Union[MemoryBackendType, str],
        config: Optional[Dict[str, Any]] = None
    ) -> MemoryBackend:
        """Create a memory backend instance.
        
        Args:
            backend_type: Backend type (enum or string)
            config: Backend configuration
            
        Returns:
            Configured backend instance
            
        Raises:
            ValueError: If backend type is not supported
        """
        if isinstance(backend_type, str):
            try:
                backend_type = MemoryBackendType(backend_type)
            except ValueError:
                raise ValueError(f"Unsupported backend type: {backend_type}")
        
        if backend_type not in cls._backends:
            raise ValueError(f"Backend {backend_type} not registered")
        
        backend_class = cls._backends[backend_type]
        return backend_class(config)
    
    @classmethod
    def list_backends(cls) -> List[MemoryBackendType]:
        """List all registered backend types."""
        return list(cls._backends.keys())