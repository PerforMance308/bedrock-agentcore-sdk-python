# Multi-Backend Memory Architecture Guide

## Overview

The AgentCore Memory SDK now supports multiple backend implementations while maintaining 100% API compatibility with existing code. You can seamlessly switch between:

- **AWS Bedrock AgentCore** (default) - Cloud-native, fully managed
- **PostgreSQL + pgvector** - Self-hosted, enterprise-grade RDBMS  
- **ChromaDB** - Local vector database optimized for AI workloads

## Key Benefits

✅ **Zero Code Changes** - Existing code works without modification  
✅ **Same API Surface** - All methods and parameters remain identical  
✅ **Feature Parity** - All backends support the complete feature set  
✅ **Configuration-Driven** - Backend selection via simple config  
✅ **Fallback Support** - Graceful degradation to AWS if backend fails  

## Usage Examples

### AWS Bedrock (Default - No Changes Required)

```python
from bedrock_agentcore.memory import MemoryClient

# Existing usage - works exactly as before
client = MemoryClient(region_name="us-west-2")

memory = client.create_memory_and_wait(
    name="my-memory",
    strategies=[],
    event_expiry_days=90
)
```

### PostgreSQL Backend

```python
from bedrock_agentcore.memory import MemoryClient

# Switch to PostgreSQL with configuration
client = MemoryClient(
    backend_type="postgresql",
    backend_config={
        "host": "localhost",
        "port": 5432,
        "database": "memory_db", 
        "user": "postgres",
        "password": "your_password",
        "min_connections": 5,
        "max_connections": 20
    }
)

# Same API calls work identically
memory = client.create_memory_and_wait(
    name="my-memory", 
    strategies=[{
        "strategyType": "semantic",
        "namespace": "/user/{actorId}/session/{sessionId}"
    }]
)

# Store conversation  
event = client.create_event(
    memory_id=memory["memoryId"],
    actor_id="user-123",
    session_id="session-456", 
    payload=[{
        'conversational': {
            'role': 'USER',
            'content': {'text': "Hello, I'm interested in AI"}
        }
    }]
)

# Retrieve memories with vector search
memories = client.retrieve_memories(
    memory_id=memory["memoryId"],
    namespace="/user/user-123/session/session-456",
    query="AI interests"
)
```

### ChromaDB Backend (Local)

```python
from bedrock_agentcore.memory import MemoryClient

# Local ChromaDB with persistence
client = MemoryClient(
    backend_type="chromadb",
    backend_config={
        "persist_directory": "./my_memory_db",
        "embedding_model": "all-MiniLM-L6-v2",
        "max_batch_size": 100
    }
)

# Identical API usage
memory = client.create_memory_and_wait(
    name="local-memory",
    strategies=[{
        "strategyType": "semantic", 
        "namespace": "/agent/{actorId}/knowledge"
    }]
)

# Same conversation storage
turns = client.get_last_k_turns(
    memory_id=memory["memoryId"],
    session_id="session-001",
    k=10
)
```

## Backend Comparison

| Feature | AWS Bedrock | PostgreSQL | ChromaDB |
|---------|-------------|------------|----------|
| **Hosting** | Managed Cloud | Self-Hosted | Local/Self-Hosted |
| **Vector Search** | ✅ Native | ✅ pgvector | ✅ Native |
| **ACID Transactions** | ✅ | ✅ | ⚠️ Eventual Consistency |
| **Scalability** | ✅ Auto-scaling | ✅ Manual scaling | ⚠️ Single-node |
| **Cost** | Pay-per-use | Infrastructure + ops | Infrastructure only |
| **Data Privacy** | AWS-managed | Full control | Full control |
| **Setup Complexity** | None | Medium | Low |

## Installation Requirements

### PostgreSQL Backend
```bash
pip install asyncpg  # Async PostgreSQL driver
# Install PostgreSQL and pgvector extension
```

### ChromaDB Backend  
```bash
pip install chromadb  # Vector database
```

## Configuration Options

### PostgreSQL Configuration
```python
backend_config = {
    "host": "localhost",                    # Database host
    "port": 5432,                          # Database port  
    "database": "memory_db",               # Database name
    "user": "postgres",                    # Database user
    "password": "password",                # Database password
    "min_connections": 5,                  # Min connection pool size
    "max_connections": 20,                 # Max connection pool size  
    "embedding_dimension": 1536,           # Vector dimension (1536 for OpenAI)
}
```

### ChromaDB Configuration
```python
backend_config = {
    "persist_directory": "./chroma_memory", # Local storage path
    "collection_prefix": "memory_",         # Collection name prefix
    "embedding_model": "all-MiniLM-L6-v2", # Embedding model
    "max_batch_size": 100,                 # Batch size for operations
    "enable_telemetry": False              # Disable telemetry
}
```

## Migration Between Backends

The SDK maintains data format compatibility between backends. You can migrate data using the built-in tools:

```python
# Export from AWS to PostgreSQL
from bedrock_agentcore.memory.migration import migrate_memory

migrate_memory(
    source_client=MemoryClient(region_name="us-west-2"),
    target_client=MemoryClient(
        backend_type="postgresql",
        backend_config=pg_config
    ),
    memory_id="memory-to-migrate"
)
```

## Performance Characteristics

### AWS Bedrock
- **Latency**: ~200-500ms (network dependent)
- **Throughput**: High (AWS managed scaling)  
- **Vector Search**: Optimized managed service
- **Best For**: Production workloads, minimal ops

### PostgreSQL + pgvector
- **Latency**: ~10-50ms (local network)
- **Throughput**: High (configurable resources)
- **Vector Search**: Excellent with proper indexing
- **Best For**: Enterprise deployments, data sovereignty

### ChromaDB
- **Latency**: ~1-10ms (local disk)
- **Throughput**: Medium (single-node limitations)
- **Vector Search**: Optimized for similarity search
- **Best For**: Development, prototyping, edge deployments

## Schema Details

### PostgreSQL Schema
```sql
-- Memory resources
CREATE TABLE memories (
    memory_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    status VARCHAR(20) DEFAULT 'CREATING',
    strategies JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation events
CREATE TABLE memory_events (
    event_id VARCHAR(255) PRIMARY KEY,
    memory_id VARCHAR(255) REFERENCES memories(memory_id),
    actor_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    payload JSONB NOT NULL,
    event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Memory records with vector embeddings
CREATE TABLE memory_records (
    record_id VARCHAR(255) PRIMARY KEY,
    memory_id VARCHAR(255) REFERENCES memories(memory_id),
    namespace VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),              -- pgvector column
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vector similarity index
CREATE INDEX idx_memory_records_embedding 
ON memory_records USING ivfflat (embedding vector_cosine_ops);
```

### ChromaDB Collections
```python
# Collections are auto-created per memory resource
collection_name = f"memory_{memory_id.replace('-', '_')}"

# Documents stored with metadata:
{
    "documents": ["conversation content"],
    "metadatas": [{
        "namespace": "/user/123/session/456", 
        "actor_id": "user-123",
        "session_id": "session-456",
        "strategy_type": "semantic"
    }],
    "ids": ["record-uuid"]
}
```

## Error Handling & Fallbacks

The client automatically handles backend failures:

```python
# If PostgreSQL backend fails to initialize
client = MemoryClient(
    backend_type="postgresql", 
    backend_config={"host": "unreachable-host"}
)
# ⚠️ Logs warning and falls back to AWS Bedrock

# Graceful degradation ensures existing code never breaks
memory = client.create_memory(name="test")  # Works via AWS fallback
```

## Health Monitoring

```python
# Check backend health
health = client.health_check()
print(health)
# Output: {'status': 'healthy', 'backend': 'postgresql', 'connections': 5}

# Get performance metrics  
metrics = client.get_metrics()
print(metrics)
# Output: {'memory_count': 10, 'event_count': 1000, 'record_count': 500}
```

## Best Practices

### Development
- Use **ChromaDB** for local development and testing
- Benefits: Fast, no setup, data isolation

### Production  
- Use **PostgreSQL** for self-hosted production
- Benefits: Full control, ACID compliance, enterprise features
- Use **AWS Bedrock** for managed production  
- Benefits: Zero ops, auto-scaling, AWS integration

### Hybrid Deployments
```python
# Environment-based backend selection
import os

backend_type = os.getenv('MEMORY_BACKEND', 'aws_bedrock')
client = MemoryClient(backend_type=backend_type, backend_config=config)
```

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Install pgvector extension  
CREATE EXTENSION vector;

# Verify connection permissions
GRANT ALL PRIVILEGES ON DATABASE memory_db TO postgres;
```

### ChromaDB Issues
```python
# Clear corrupted ChromaDB data
import shutil
shutil.rmtree('./chroma_memory')

# Reset with fresh database
client = MemoryClient(backend_type="chromadb")
```

## Complete Compatibility

The multi-backend architecture maintains **100% backward compatibility**:

- ✅ All existing method signatures unchanged
- ✅ All parameter names and types identical  
- ✅ All response formats consistent
- ✅ All error handling preserved
- ✅ All async/sync patterns maintained

**Your existing code will work without any modifications!**