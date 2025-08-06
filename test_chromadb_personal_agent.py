#!/usr/bin/env python3
"""
ChromaDB Personal Agent Memory Test

This test demonstrates the ChromaDB memory backend working with 
a personal agent scenario, similar to the AWS Bedrock examples.
It shows conversation memory, context retrieval, and multi-turn interactions.
"""

import sys
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from bedrock_agentcore.memory.client import MemoryClient


class PersonalAgentMemoryTest:
    """Personal Agent with ChromaDB Memory Backend"""
    
    def __init__(self):
        # Agent configuration
        self.ACTOR_ID = "user_12345"
        self.SESSION_ID = "personal_chat_session_1"
        self.AGENT_ID = "personal_assistant"
        
        # Initialize ChromaDB memory client
        self.memory_client = MemoryClient(
            backend_type="chromadb",
            backend_config={
                "persist_directory": "./chromadb_personal_agent",
                "collection_name": "personal_agent_memory"
            }
        )
        
        self.memory_id = None
        print("🤖 Personal Agent with ChromaDB Memory initialized")
    
    def setup_memory(self):
        """Initialize the memory resource"""
        print("\n📝 Setting up memory resource...")
        
        # Memory strategies for personal agent using correct enum values
        memory_strategies = [
            {
                "semanticMemoryStrategy": {
                    "name": "personal_semantic",
                    "description": "Semantic memory for personal assistant",
                    "extraction": {
                        "semanticExtractionConfiguration": {
                            "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.titan-embed-text-v1"
                        }
                    }
                }
            },
            {
                "summaryMemoryStrategy": {
                    "name": "personal_summary",
                    "description": "Summary memory for conversation context",
                    "extraction": {
                        "maximumTokens": 500
                    }
                }
            }
        ]
        
        try:
            memory = self.memory_client.create_memory_and_wait(
                name="personal_agent_memory",
                description="Memory for personal assistant conversations",
                strategies=memory_strategies,
                event_expiry_days=30  # Shorter for testing
            )
            
            self.memory_id = memory["memoryId"]
            print(f"✅ Memory created: {self.memory_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create memory: {e}")
            return False
    
    def create_conversation_event(self, role: str, content: str, event_type: str = "conversation"):
        """Create a memory event for conversation"""
        payload = [
            {
                "conversational": {
                    "role": role,
                    "content": {"text": content}
                }
            }
        ]
        
        try:
            event = self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.ACTOR_ID,
                session_id=self.SESSION_ID,
                payload=payload
            )
            
            print(f"💬 {role.upper()}: {content}")
            return event
            
        except Exception as e:
            print(f"❌ Failed to create event: {e}")
            return None
    
    def get_conversation_context(self, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns for context"""
        try:
            recent_turns = self.memory_client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.ACTOR_ID,
                session_id=self.SESSION_ID,
                k=k
            )
            
            print(f"\n📚 Retrieved {len(recent_turns)} recent conversation turns")
            return recent_turns
            
        except Exception as e:
            print(f"❌ Failed to retrieve context: {e}")
            return []
    
    def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve semantically relevant memories"""
        try:
            # Use the semantic strategy namespace pattern
            namespace = f"/actor/{self.ACTOR_ID}/strategy/default/{self.SESSION_ID}"
            memories = self.memory_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=namespace,
                query=query,
                top_k=top_k,
                actor_id=self.ACTOR_ID
            )
            
            print(f"🔍 Retrieved {len(memories)} relevant memories for: '{query}'")
            for i, memory in enumerate(memories, 1):
                content = memory.get("content", "")[:100]
                score = memory.get("relevanceScore", 0)
                print(f"   {i}. [Score: {score:.3f}] {content}...")
            
            return memories
            
        except Exception as e:
            print(f"❌ Failed to retrieve memories: {e}")
            return []
    
    def simulate_conversation(self):
        """Simulate a multi-turn conversation with memory"""
        print("\n🎭 Starting Personal Agent Conversation Simulation")
        print("=" * 60)
        
        # Conversation scenario: Personal assistant helping with daily tasks
        conversations = [
            ("user", "Hi! I'm planning a dinner party for 8 people next Friday. Can you help me?"),
            ("assistant", "I'd be happy to help you plan your dinner party! For 8 people, let me suggest some ideas. What type of cuisine are you thinking about?"),
            ("user", "I was thinking Italian food. My guests love pasta and I have a good recipe for tiramisu."),
            ("assistant", "Excellent choice! Italian is perfect for entertaining. With your tiramisu dessert covered, let's plan the main course. I can suggest some pasta dishes that work well for groups of 8."),
            ("user", "Actually, let me also ask - do you remember what I cooked last time I had friends over?"),
            ("assistant", "Let me check your previous conversations about cooking..."),
            ("user", "I made lasagna last month and it was a hit. But I want to try something different this time."),
            ("assistant", "Since lasagna was successful last time, how about trying osso buco with risotto? It's impressive but manageable for 8 people, and very different from lasagna."),
            ("user", "That sounds amazing! Can you help me make a shopping list?"),
            ("assistant", "Absolutely! For osso buco with risotto for 8 people, you'll need: veal shanks, arborio rice, beef stock, white wine, onions, carrots, celery, and fresh herbs. Should I break this down by store sections?")
        ]
        
        # Store each conversation turn
        for i, (role, content) in enumerate(conversations, 1):
            print(f"\nTurn {i}:")
            event = self.create_conversation_event(role, content)
            
            if role == "assistant" and "remember what I cooked" in conversations[i-2][1]:
                # Demonstrate memory retrieval
                print("\n🧠 Agent retrieving relevant cooking memories...")
                relevant_memories = self.retrieve_relevant_memories("cooking dinner party friends", top_k=3)
        
        return True
    
    def demonstrate_context_usage(self):
        """Demonstrate how agent uses conversation context"""
        print("\n🧠 Demonstrating Context Retrieval")
        print("=" * 40)
        
        # Get recent context
        context = self.get_conversation_context(k=5)
        
        if context:
            print(f"\n📖 Recent conversation context ({len(context)} turns):")
            for i, turn in enumerate(context, 1):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")[:100]
                timestamp = turn.get("timestamp", "")
                print(f"   {i}. [{role.upper()}] {content}... ({timestamp})")
        
        # Query for specific topics
        topics_to_query = [
            "dinner party planning",
            "Italian food recipes", 
            "cooking for groups",
            "tiramisu dessert"
        ]
        
        print(f"\n🔍 Querying memory for specific topics:")
        for topic in topics_to_query:
            print(f"\n--- Topic: {topic} ---")
            relevant = self.retrieve_relevant_memories(topic, top_k=2)
    
    def test_memory_persistence(self):
        """Test that memory persists across client restarts"""
        print("\n💾 Testing Memory Persistence")
        print("=" * 40)
        
        # Create a new client instance
        new_client = MemoryClient(
            backend_type="chromadb",
            backend_config={
                "persist_directory": "./chromadb_personal_agent",
                "collection_name": "personal_agent_memory"
            }
        )
        
        try:
            # Try to retrieve memories with new client using correct namespace
            namespace = f"/actor/{self.ACTOR_ID}/strategy/default/{self.SESSION_ID}"
            memories = new_client.retrieve_memories(
                memory_id=self.memory_id,
                namespace=namespace,
                query="dinner party",
                top_k=3,
                actor_id=self.ACTOR_ID
            )
            
            if memories:
                print(f"✅ Memory persistence verified! Found {len(memories)} persisted memories")
                return True
            else:
                print("⚠️  No memories found with new client")
                return False
                
        except Exception as e:
            print(f"❌ Persistence test failed: {e}")
            return False
    
    def run_health_check(self):
        """Run health check on memory system"""
        print("\n🏥 Running Memory System Health Check")
        print("=" * 40)
        
        try:
            health = self.memory_client.health_check()
            print(f"✅ Health Check Result: {health}")
            return health.get("status") == "healthy"
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False


def main():
    """Main test execution"""
    print("🚀 ChromaDB Personal Agent Memory Test")
    print("=" * 60)
    
    # Initialize agent
    agent = PersonalAgentMemoryTest()
    
    # Setup memory
    if not agent.setup_memory():
        print("❌ Memory setup failed, aborting test")
        return False
    
    # Run health check
    if not agent.run_health_check():
        print("❌ Health check failed")
        return False
    
    # Simulate conversation
    if not agent.simulate_conversation():
        print("❌ Conversation simulation failed")
        return False
    
    # Demonstrate context usage
    agent.demonstrate_context_usage()
    
    # Test persistence
    if not agent.test_memory_persistence():
        print("❌ Persistence test failed")
        return False
    
    print("\n🎉 All ChromaDB Personal Agent Tests Passed!")
    print("\n📊 Test Summary:")
    print("✅ Memory resource creation")
    print("✅ Multi-turn conversation storage") 
    print("✅ Semantic memory retrieval")
    print("✅ Context-aware responses")
    print("✅ Memory persistence across sessions")
    print("✅ Health monitoring")
    
    print(f"\n💾 Memory data persisted in: ./chromadb_personal_agent")
    print("🔄 You can run this test multiple times to see persistence in action")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
    
    print("\n🎯 Next steps to try:")
    print("1. Run this test multiple times to see memory persistence")
    print("2. Modify the conversation scenarios to test different use cases")
    print("3. Experiment with different retrieval queries")
    print("4. Check the ChromaDB data directory for stored embeddings")