"""流式输出支持 - SSE (Server-Sent Events) 实现"""

from typing import Dict, Any
from dataclasses import dataclass
import json
import time
from enum import Enum


class StreamEventType(Enum):
    """流式事件类型"""
    AGENT_START = "agent_start"
    AGENT_FINISH = "agent_finish"
    STEP_START = "step_start"
    STEP_FINISH = "step_finish"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_FINISH = "tool_call_finish"
    LLM_CHUNK = "llm_chunk"  # LLM 流式输出的文本块
    THINKING = "thinking"  # Agent 思考过程
    ERROR = "error"


@dataclass
class StreamEvent:
    """流式事件"""
    type: StreamEventType
    timestamp: float
    agent_name: str
    data: Dict[str, Any]
    
    @classmethod
    def create(cls, event_type: StreamEventType, agent_name: str, **data) -> 'StreamEvent':
        """创建事件"""
        return cls(
            type=event_type,
            timestamp=time.time(),
            agent_name=agent_name,
            data=data
        )
    
    def to_sse(self) -> str:
        """转换为 SSE 格式
        
        SSE 格式：
        event: <event_type>
        data: <json_data>
        
        """
        event_dict = {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "data": self.data
        }
        
        # SSE 格式要求
        lines = [
            f"event: {self.type.value}",
            f"data: {json.dumps(event_dict, ensure_ascii=False)}",
            ""  # 空行表示事件结束
        ]
        return "\n".join(lines) + "\n"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "agent_name": self.agent_name,
            "data": self.data
        }

