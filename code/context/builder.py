"""ContextBuilder - GSSC Pipeline Implementation

Implements the Gather-Select-Structure-Compress context building process:
1. Gather: Collect candidate information from multiple sources (history, tool results)
2. Select: Filter based on priority, relevance, and diversity
3. Structure: Organize into a structured context template
4. Compress: Compress and normalize within the budget

Note: MemoryTool and RAGTool have been removed. If you need to use them, please implement them yourself.
"""
import math
import tiktoken
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from ..core.message import Message


@dataclass
class ContextPacket:
    """Context information packet"""
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    relevance_score: float = 0.0  # 0.0-1.0
    
    def __post_init__(self):
        """Automatically calculate token count"""
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


@dataclass
class ContextConfig:
    """Context build configuration"""
    max_tokens: int = 8000  # Total budget
    reserve_ratio: float = 0.15  # Generation margin (10-20%)
    min_relevance: float = 0.3   # Minimum relevance threshold
    enable_mmr: bool = True  # Enable Maximum Marginal Relevance (diversity)
    mmr_lambda: float = 0.7  # MMR balance parameter (0=pure diversity, 1=pure relevance)
    system_prompt_template: str = ""  # System prompt template
    enable_compression: bool = True  # Enable compression
    
    def get_available_tokens(self) -> int:
        """Get available token budget (deducting margin)"""
        return int(self.max_tokens * (1 - self.reserve_ratio))


class ContextBuilder:
    """Context Builder - GSSC Pipeline

    Note: MemoryTool and RAGTool have been removed. This class is temporarily unavailable.

    Usage example:
    ```python
    builder = ContextBuilder(
        config=ContextConfig(max_tokens=8000)
    )

    context = builder.build(
        user_query="User query",
        conversation_history=[...],
        system_instructions="System instructions"
    )
    ```
    """

    def __init__(
        self,
        config: Optional[ContextConfig] = None
    ):
        self.config = config or ContextConfig()
        self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        additional_packets: Optional[List[ContextPacket]] = None
    ) -> str:
        """Build the complete context
        
        Args:
            user_query: User query
            conversation_history: Conversation history
            system_instructions: System instructions
            additional_packets: Additional context packets
            
        Returns:
            Structured context string
        """
        # 1. Gather: Collect candidate information
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_packets=additional_packets or []
        )
        
        # 2. Select: Filter and sort
        selected_packets = self._select(packets, user_query)
        
        # 3. Structure: Organize into a structured template
        structured_context = self._structure(
            selected_packets=selected_packets,
            user_query=user_query,
            system_instructions=system_instructions
        )
        
        # 4. Compress: Compress and normalize (if over budget)
        final_context = self._compress(structured_context)
        
        return final_context
    
    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_packets: List[ContextPacket]
    ) -> List[ContextPacket]:
        """Gather: Collect candidate information"""
        packets = []
        
        # P0: System instructions (strong constraint)
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                metadata={"type": "instructions"}
            ))

        # Note: MemoryTool and RAGTool have been removed
        # If you need memory and knowledge base features, please implement them yourself

        # P3: Conversation history (supplementary material)
        if conversation_history:
            # Keep only the most recent N items
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            packets.append(ContextPacket(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))

        # Add additional packets
        packets.extend(additional_packets)

        return packets
    
    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str
    ) -> List[ContextPacket]:
        """Select: Filtering based on score and budget"""
        # 1) Calculate relevance (keyword overlap)
        query_tokens = set(user_query.lower().split())
        for packet in packets:
            content_tokens = set(packet.content.lower().split())
            if len(query_tokens) > 0:
                overlap = len(query_tokens & content_tokens)
                packet.relevance_score = overlap / len(query_tokens)
            else:
                packet.relevance_score = 0.0
        
        # 2) Calculate recency (exponential decay)
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600  # 1 hour time scale, can be exposed to config
            return math.exp(-delta / tau)
        
        # 3) Calculate composite score: 0.7*relevance + 0.3*recency
        scored_packets: List[Tuple[float, ContextPacket]] = []
        for p in packets:
            rec = recency_score(p.timestamp)
            score = 0.7 * p.relevance_score + 0.3 * rec
            scored_packets.append((score, p))
        
        # 4) Extract system instructions separately, fixed inclusion
        system_packets = [p for (_, p) in scored_packets if p.metadata.get("type") == "instructions"]
        remaining = [p for (s, p) in sorted(scored_packets, key=lambda x: x[0], reverse=True)
                     if p.metadata.get("type") != "instructions"]
        
        # 5) Filter by min_relevance (for non-system packets)
        filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]
        
        # 6) Fill according to budget
        available_tokens = self.config.get_available_tokens()
        selected: List[ContextPacket] = []
        used_tokens = 0
        
        # Put system instructions first (unsorted)
        for p in system_packets:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count
        
        # Then add the rest based on score
        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count
        
        return selected
    
    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str,
        system_instructions: Optional[str]
    ) -> str:
        """Structure: Organize into a structured context template"""
        sections = []
        
        # [Role & Policies] - System instructions
        p0_packets = [p for p in selected_packets if p.metadata.get("type") == "instructions"]
        if p0_packets:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_packets])
            sections.append(role_section)
        
        # [Task] - Current task
        sections.append(f"[Task]\nUser Query: {user_query}")
        
        # [State] - Task state
        p1_packets = [p for p in selected_packets if p.metadata.get("type") == "task_state"]
        if p1_packets:
            state_section = "[State]\nKey Progress and Pending Issues:\n"
            state_section += "\n".join([p.content for p in p1_packets])
            sections.append(state_section)
        
        # [Evidence] - Factual evidence
        p2_packets = [
            p for p in selected_packets
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result"}
        ]
        if p2_packets:
            evidence_section = "[Evidence]\nFacts and References:\n"
            for p in p2_packets:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)
        
        # [Context] - Supplementary material (history, etc.)
        p3_packets = [p for p in selected_packets if p.metadata.get("type") == "history"]
        if p3_packets:
            context_section = "[Context]\nConversation History and Background:\n"
            context_section += "\n".join([p.content for p in p3_packets])
            sections.append(context_section)
        
        # [Output] - Output constraints
        output_section = """[Output]
                            Please answer in the following format:
                            1. Conclusion (concise and clear)
                            2. Basis (list supporting evidence and sources)
                            3. Risks and Assumptions (if any)
                            4. Next Steps (if applicable)"""
        sections.append(output_section)
        
        return "\n\n".join(sections)
    
    def _compress(self, context: str) -> str:
        """Compress: Compress and normalize"""
        if not self.config.enable_compression:
            return context
        
        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()
        
        if current_tokens <= available_tokens:
            return context
        
        # Simple truncation strategy (keep the first N tokens)
        # In practice, an LLM can be used for high-fidelity summarization
        print(f"⚠️ Context exceeds budget ({current_tokens} > {available_tokens}), performing truncation")
        
        # Truncate by paragraph, preserving structure
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens
        
        return "\n".join(compressed_lines)


def count_tokens(text: str) -> int:
    """Calculate text token count (using tiktoken)"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback plan: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4