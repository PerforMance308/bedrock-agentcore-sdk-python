"""ChromaDB-based memory backend implementation.

This module provides a high-performance memory backend using ChromaDB for vector
similarity search with local persistence and optimized embedding operations.
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from ..constants import MemoryStatus, StrategyType
from .base import MemoryBackend, MemoryBackendError, MemoryBackendType

logger = logging.getLogger(__name__)


class ChromaDBMemoryBackend(MemoryBackend):
    """ChromaDB-based memory backend with local persistence.
    
    Features:
    - Native vector similarity search with ChromaDB
    - Local file-based persistence (no external dependencies)
    - Efficient metadata filtering and namespace support
    - Built-in embedding model integration
    - High-performance batch operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ChromaDB backend.
        
        Args:
            config: Configuration dict with keys:
                - persist_directory: Local storage path (default: ./chroma_memory)
                - collection_prefix: Prefix for collection names (default: memory_)
                - embedding_model: Embedding model name (default: all-MiniLM-L6-v2)
                - max_batch_size: Max batch size for operations (default: 100)
                - enable_telemetry: Enable ChromaDB telemetry (default: False)
        """
        super().__init__(config)
        
        self.persist_directory = Path(self.config.get('persist_directory', './chroma_memory'))
        self.collection_prefix = self.config.get('collection_prefix', 'memory_')
        self.embedding_model = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.max_batch_size = self.config.get('max_batch_size', 100)
        self.enable_telemetry = self.config.get('enable_telemetry', False)
        
        self.client: Optional[chromadb.PersistentClient] = None
        self.metadata_db: Optional[sqlite3.Connection] = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure ChromaDB client is initialized."""
        if not self._initialized:
            await self._initialize()
    
    async def _initialize(self):
        """Initialize ChromaDB client and metadata database."""
        try:
            # Create persist directory
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            settings = Settings(
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=self.enable_telemetry
            )
            
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=settings
            )
            
            # Initialize metadata database for non-vector data
            await self._init_metadata_db()
            
            self._initialized = True
            logger.info("ChromaDB memory backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB backend: {e}")
            raise MemoryBackendError(f"ChromaDB initialization failed: {e}")
    
    async def _init_metadata_db(self):
        """Initialize SQLite database for metadata storage."""
        db_path = self.persist_directory / "metadata.db"
        self.metadata_db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.metadata_db.row_factory = sqlite3.Row
        
        # Create metadata tables
        self.metadata_db.executescript("""
            -- Memory resources table
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'CREATING',
                strategies TEXT,
                event_expiry_days INTEGER DEFAULT 90,
                memory_execution_role_arn TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Memory events table
            CREATE TABLE IF NOT EXISTS memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(memory_id) ON DELETE CASCADE,
                actor_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                payload TEXT NOT NULL,
                branch_name TEXT,
                root_event_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (root_event_id) REFERENCES memory_events(event_id) ON DELETE SET NULL
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
            CREATE INDEX IF NOT EXISTS idx_memory_events_memory_id ON memory_events(memory_id);
            CREATE INDEX IF NOT EXISTS idx_memory_events_session ON memory_events(memory_id, session_id);
            CREATE INDEX IF NOT EXISTS idx_memory_events_actor ON memory_events(memory_id, actor_id);
            CREATE INDEX IF NOT EXISTS idx_memory_events_timestamp ON memory_events(event_timestamp);
            CREATE INDEX IF NOT EXISTS idx_memory_events_branch ON memory_events(memory_id, session_id, branch_name);
            
            -- Update trigger
            CREATE TRIGGER IF NOT EXISTS update_memories_timestamp
                AFTER UPDATE ON memories
                FOR EACH ROW
            BEGIN
                UPDATE memories SET updated_at = CURRENT_TIMESTAMP WHERE memory_id = NEW.memory_id;
            END;
        """)
        
        self.metadata_db.commit()
    
    def _get_collection_name(self, memory_id: str) -> str:
        """Get ChromaDB collection name for memory ID."""
        return f"{self.collection_prefix}{memory_id.replace('-', '_')}"
    
    def _get_or_create_collection(self, memory_id: str) -> chromadb.Collection:
        """Get or create ChromaDB collection for memory resource."""
        collection_name = self._get_collection_name(memory_id)
        
        try:
            return self.client.get_collection(collection_name)
        except Exception:
            # Collection doesn't exist, create it
            return self.client.create_collection(
                name=collection_name,
                metadata={"memory_id": memory_id}
            )
    
    # ==================== CONTROL PLANE OPERATIONS ====================
    
    async def create_memory(
        self,
        name: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        event_expiry_days: int = 90,
        memory_execution_role_arn: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new memory resource."""
        await self._ensure_initialized()
        
        memory_id = self.generate_memory_id()
        
        # Create metadata record
        self.metadata_db.execute("""
            INSERT INTO memories (memory_id, name, description, status, strategies, 
                                event_expiry_days, memory_execution_role_arn)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, name, description, MemoryStatus.CREATING.value,
              json.dumps(strategies or []), event_expiry_days, memory_execution_role_arn))
        
        self.metadata_db.commit()
        
        # Create ChromaDB collection
        collection = self._get_or_create_collection(memory_id)
        
        # For ChromaDB (local storage), immediately set to ACTIVE
        self.metadata_db.execute(
            "UPDATE memories SET status = ? WHERE memory_id = ?",
            (MemoryStatus.ACTIVE.value, memory_id)
        )
        self.metadata_db.commit()
        
        logger.info(f"Memory {memory_id} created and activated")
        
        return await self.get_memory(memory_id)
    
    
    async def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory resource metadata."""
        await self._ensure_initialized()
        
        cursor = self.metadata_db.execute("""
            SELECT memory_id, name, description, status, strategies, 
                   event_expiry_days, memory_execution_role_arn, 
                   created_at, updated_at
            FROM memories WHERE memory_id = ?
        """, (memory_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            'memoryId': row['memory_id'],
            'name': row['name'],
            'description': row['description'],
            'status': row['status'],
            'strategies': json.loads(row['strategies']) if row['strategies'] else [],
            'eventExpiryDays': row['event_expiry_days'],
            'memoryExecutionRoleArn': row['memory_execution_role_arn'],
            'createdAt': row['created_at'],
            'updatedAt': row['updated_at']
        }
    
    async def get_memory_status(self, memory_id: str) -> Optional[MemoryStatus]:
        """Get memory resource status."""
        await self._ensure_initialized()
        
        cursor = self.metadata_db.execute(
            "SELECT status FROM memories WHERE memory_id = ?",
            (memory_id,)
        )
        
        row = cursor.fetchone()
        if row:
            return MemoryStatus(row['status'])
        return None
    
    async def list_memories(
        self,
        max_results: int = 10,
        next_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """List memory resources with pagination."""
        await self._ensure_initialized()
        
        offset = int(next_token) if next_token else 0
        
        cursor = self.metadata_db.execute("""
            SELECT memory_id, name, description, status, strategies,
                   event_expiry_days, memory_execution_role_arn,
                   created_at, updated_at
            FROM memories 
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, (max_results + 1, offset))
        
        rows = cursor.fetchall()
        
        memories = []
        for row in rows[:max_results]:
            memories.append({
                'memoryId': row['memory_id'],
                'name': row['name'],
                'description': row['description'],
                'status': row['status'],
                'strategies': json.loads(row['strategies']) if row['strategies'] else [],
                'eventExpiryDays': row['event_expiry_days'],
                'memoryExecutionRoleArn': row['memory_execution_role_arn'],
                'createdAt': row['created_at'],
                'updatedAt': row['updated_at']
            })
        
        result = {'memories': memories}
        if len(rows) > max_results:
            result['nextToken'] = str(offset + max_results)
        
        return result
    
    async def update_memory(
        self,
        memory_id: str,
        strategies: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Update memory resource configuration."""
        await self._ensure_initialized()
        
        updates = []
        params = []
        
        if strategies is not None:
            updates.append("strategies = ?")
            params.append(json.dumps(strategies))
        
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        
        if not updates:
            return await self.get_memory(memory_id)
        
        updates.append("status = ?")
        params.append(MemoryStatus.UPDATING.value)
        params.append(memory_id)
        
        self.metadata_db.execute(f"""
            UPDATE memories 
            SET {', '.join(updates)}
            WHERE memory_id = ?
        """, params)
        
        self.metadata_db.commit()
        
        # Simulate async processing
        asyncio.create_task(self._activate_memory_after_delay(memory_id))
        
        return await self.get_memory(memory_id)
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory resource."""
        await self._ensure_initialized()
        
        # Delete ChromaDB collection
        try:
            collection_name = self._get_collection_name(memory_id)
            self.client.delete_collection(collection_name)
        except Exception:
            pass  # Collection might not exist
        
        # Delete metadata
        cursor = self.metadata_db.execute(
            "DELETE FROM memories WHERE memory_id = ?",
            (memory_id,)
        )
        
        self.metadata_db.commit()
        return cursor.rowcount > 0
    
    # ==================== DATA PLANE OPERATIONS ====================
    
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
        """Create a new conversation event."""
        await self._ensure_initialized()
        
        event_id = self.generate_event_id()
        event_timestamp = datetime.utcnow().isoformat()
        
        self.metadata_db.execute("""
            INSERT INTO memory_events (event_id, memory_id, actor_id, session_id,
                                     event_timestamp, payload, branch_name, root_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (event_id, memory_id, actor_id, session_id, 
              event_timestamp, json.dumps(payload), branch_name, root_event_id))
        
        self.metadata_db.commit()
        
        # Extract memories immediately (for reliable testing)
        await self._extract_memories(memory_id, event_id, payload, actor_id, session_id)
        
        return {
            'eventId': event_id,
            'memoryId': memory_id,
            'actorId': actor_id,
            'sessionId': session_id,
            'eventTimestamp': event_timestamp,
            'payload': payload,
            'branch': {'name': branch_name, 'rootEventId': root_event_id} if branch_name else None
        }
    
    async def _extract_memories(
        self, 
        memory_id: str, 
        event_id: str, 
        payload: List[Dict[str, Any]],
        actor_id: str,
        session_id: str
    ):
        """Extract memories from event payload and store in ChromaDB."""
        
        # Get memory strategies
        memory = await self.get_memory(memory_id)
        if not memory or memory['status'] != MemoryStatus.ACTIVE.value:
            return
        
        strategies = memory.get('strategies', [])
        collection = self._get_or_create_collection(memory_id)
        
        documents = []
        metadatas = []
        ids = []
        
        for strategy in strategies:
            # Handle nested strategy structure
            if 'semanticMemoryStrategy' in strategy:
                strategy_config = strategy['semanticMemoryStrategy']
                strategy_type = 'semantic'
            elif 'summaryMemoryStrategy' in strategy:
                strategy_config = strategy['summaryMemoryStrategy']
                strategy_type = 'summary'
            else:
                continue
                
            # Get namespace templates from the strategy config
            namespace_templates = strategy_config.get('namespaces', ['/actor/{actorId}/strategy/{strategyId}'])
            
            # Extract content from payload (once per strategy)
            content_parts = []
            for msg in payload:
                if 'conversational' in msg:
                    conv = msg['conversational']
                    role = conv.get('role', '')
                    content = conv.get('content', {})
                    text = content.get('text', '') if isinstance(content, dict) else str(content)
                    if text:
                        content_parts.append(f"{role}: {text}")
            
            if content_parts:
                content = '\n'.join(content_parts)
                
                # Create one record per namespace template
                for namespace_template in namespace_templates:
                    # Resolve namespace template
                    namespace = namespace_template.format(
                        actorId=actor_id,
                        strategyId=strategy_config.get('strategyId', 'default'),
                        sessionId=session_id
                    )
                    
                    record_id = str(uuid.uuid4())
                    
                    documents.append(content)
                    metadatas.append({
                        'namespace': namespace,
                        'actor_id': actor_id,
                        'session_id': session_id,
                        'source_event_id': event_id,
                        'strategy_type': strategy_type,
                        'created_at': datetime.utcnow().isoformat()
                    })
                    ids.append(record_id)
        
        if documents:
            # Add documents to ChromaDB collection
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Extracted {len(documents)} memories for event {event_id}")
    
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
        """List conversation events with filtering and pagination."""
        await self._ensure_initialized()
        
        conditions = ["memory_id = ?"]
        params = [memory_id]
        
        if actor_id:
            conditions.append("actor_id = ?")
            params.append(actor_id)
        
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        
        if branch_name:
            conditions.append("branch_name = ?")
            params.append(branch_name)
        
        if start_time:
            conditions.append("event_timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            conditions.append("event_timestamp <= ?")
            params.append(end_time.isoformat())
        
        offset = int(next_token) if next_token else 0
        params.extend([max_results + 1, offset])
        
        cursor = self.metadata_db.execute(f"""
            SELECT event_id, memory_id, actor_id, session_id, event_timestamp,
                   payload, branch_name, root_event_id
            FROM memory_events
            WHERE {' AND '.join(conditions)}
            ORDER BY event_timestamp DESC
            LIMIT ? OFFSET ?
        """, params)
        
        rows = cursor.fetchall()
        
        events = []
        for row in rows[:max_results]:
            event = {
                'eventId': row['event_id'],
                'memoryId': row['memory_id'],
                'actorId': row['actor_id'],
                'sessionId': row['session_id'],
                'eventTimestamp': row['event_timestamp'],
                'payload': json.loads(row['payload'])
            }
            
            if row['branch_name']:
                event['branch'] = {
                    'name': row['branch_name'],
                    'rootEventId': row['root_event_id']
                }
            
            events.append(event)
        
        result = {'events': events}
        if len(rows) > max_results:
            result['nextToken'] = str(offset + max_results)
        
        return result
    
    async def get_event(
        self,
        memory_id: str,
        event_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a specific event by ID."""
        await self._ensure_initialized()
        
        cursor = self.metadata_db.execute("""
            SELECT event_id, memory_id, actor_id, session_id, event_timestamp,
                   payload, branch_name, root_event_id
            FROM memory_events
            WHERE memory_id = ? AND event_id = ?
        """, (memory_id, event_id))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        event = {
            'eventId': row['event_id'],
            'memoryId': row['memory_id'],
            'actorId': row['actor_id'],
            'sessionId': row['session_id'],
            'eventTimestamp': row['event_timestamp'],
            'payload': json.loads(row['payload'])
        }
        
        if row['branch_name']:
            event['branch'] = {
                'name': row['branch_name'],
                'rootEventId': row['root_event_id']
            }
        
        return event
    
    # ==================== MEMORY RETRIEVAL OPERATIONS ====================
    
    async def retrieve_memories(
        self,
        memory_id: str,
        namespace: str,
        query: str,
        top_k: int = 10,
        actor_id: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memory records using semantic search."""
        await self._ensure_initialized()
        
        collection = self._get_or_create_collection(memory_id)
        
        # Build metadata filter for ChromaDB
        where_conditions = {"$and": [{"namespace": {"$eq": namespace}}]}
        if actor_id:
            where_conditions["$and"].append({"actor_id": {"$eq": actor_id}})
        
        try:
            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_conditions
            )
            
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, document in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if 'distances' in results else 0.0
                    
                    memories.append({
                        'recordId': results['ids'][0][i],
                        'content': document,
                        'metadata': metadata,
                        'actorId': metadata.get('actor_id'),
                        'sessionId': metadata.get('session_id'),
                        'sourceEventId': metadata.get('source_event_id'),
                        'strategyType': metadata.get('strategy_type'),
                        'score': 1.0 - distance,  # Convert distance to similarity score
                        'createdAt': metadata.get('created_at')
                    })
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return []
    
    # ==================== CONVERSATION BRANCHING ====================
    
    async def fork_conversation(
        self,
        memory_id: str,
        session_id: str,
        branch_name: str,
        root_event_id: str
    ) -> Dict[str, Any]:
        """Create a new conversation branch."""
        await self._ensure_initialized()
        
        # Verify root event exists
        cursor = self.metadata_db.execute("""
            SELECT EXISTS(
                SELECT 1 FROM memory_events 
                WHERE memory_id = ? AND event_id = ? AND session_id = ?
            )
        """, (memory_id, root_event_id, session_id))
        
        if not cursor.fetchone()[0]:
            raise MemoryBackendError(f"Root event {root_event_id} not found")
        
        return {
            'branchName': branch_name,
            'rootEventId': root_event_id,
            'sessionId': session_id,
            'memoryId': memory_id,
            'createdAt': datetime.utcnow().isoformat()
        }
    
    async def list_branches(
        self,
        memory_id: str,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """List all branches in a conversation session."""
        await self._ensure_initialized()
        
        cursor = self.metadata_db.execute("""
            SELECT DISTINCT branch_name, root_event_id,
                   MIN(event_timestamp) as created_at
            FROM memory_events
            WHERE memory_id = ? AND session_id = ? AND branch_name IS NOT NULL
            GROUP BY branch_name, root_event_id
            ORDER BY created_at
        """, (memory_id, session_id))
        
        branches = []
        for row in cursor.fetchall():
            branches.append({
                'branchName': row['branch_name'],
                'rootEventId': row['root_event_id'],
                'sessionId': session_id,
                'memoryId': memory_id,
                'createdAt': row['created_at']
            })
        
        return branches
    
    async def get_conversation_tree(
        self,
        memory_id: str,
        session_id: str
    ) -> Dict[str, Any]:
        """Get hierarchical conversation structure."""
        await self._ensure_initialized()
        
        # Get all events in session
        events_result = await self.list_events(
            memory_id=memory_id,
            session_id=session_id,
            max_results=1000  # Large limit to get all events
        )
        
        events = events_result.get('events', [])
        
        # Build tree structure
        main_branch = []
        branches = {}
        
        for event in events:
            if 'branch' in event and event['branch']:
                branch_name = event['branch']['name']
                if branch_name not in branches:
                    branches[branch_name] = []
                branches[branch_name].append(event)
            else:
                main_branch.append(event)
        
        return {
            'sessionId': session_id,
            'memoryId': memory_id,
            'mainBranch': main_branch,
            'branches': branches
        }
    
    # ==================== UTILITY METHODS ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform backend health check."""
        try:
            await self._ensure_initialized()
            
            # Test ChromaDB connection
            collections = self.client.list_collections()
            
            # Test metadata database
            cursor = self.metadata_db.execute("SELECT COUNT(*) FROM memories")
            memory_count = cursor.fetchone()[0]
            
            return {
                'status': 'healthy',
                'backend': 'chromadb',
                'collections': len(collections),
                'memory_count': memory_count,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'backend': 'chromadb',
                'error': str(e)
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend performance metrics."""
        await self._ensure_initialized()
        
        # Get metadata counts
        cursor = self.metadata_db.execute("""
            SELECT 
                (SELECT COUNT(*) FROM memories) as memory_count,
                (SELECT COUNT(*) FROM memory_events) as event_count
        """)
        
        stats = cursor.fetchone()
        
        # Get ChromaDB collection info
        collections = self.client.list_collections()
        total_documents = 0
        
        for collection in collections:
            try:
                count = collection.count()
                total_documents += count
            except Exception:
                pass
        
        return {
            'backend': 'chromadb',
            'memory_count': stats[0],
            'event_count': stats[1],
            'record_count': total_documents,
            'collections': len(collections),
            'persist_directory': str(self.persist_directory)
        }
    
    async def close(self):
        """Close connections and cleanup."""
        if self.metadata_db:
            self.metadata_db.close()
            self.metadata_db = None
        
        # ChromaDB client cleanup is handled automatically
        self.client = None
        self._initialized = False


# Register with factory
from .base import MemoryBackendFactory
MemoryBackendFactory.register_backend(MemoryBackendType.CHROMADB, ChromaDBMemoryBackend)