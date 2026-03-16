"""TraceLogger - Dual-format Trace Logger

Output formats:
- JSONL: Machine-readable, streaming append, supports jq analysis
- HTML: Human-readable, visual interface, built-in stats panel
"""
import re
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class TraceLogger:
    """Dual-format Trace Logger
    
    Features:
    - JSONL streaming write (real-time append)
    - HTML incremental rendering (real-time viewable)
    - Automatic sanitization (API Keys, paths)
    - Built-in stats panel (Tokens, tool calls, errors)
    
    Usage Example:
        logger = TraceLogger(output_dir="memory/traces")
        logger.log_event("session_start", {"agent_name": "MyAgent"})
        logger.log_event("tool_call", {"tool_name": "Calculator"}, step=1)
        logger.finalize()  # Generate final HTML
    """
    
    def __init__(
        self,
        output_dir: str = "memory/traces",
        sanitize: bool = True,
        html_include_raw_response: bool = False
    ):
        """Initialize TraceLogger
        
        Args:
            output_dir: Output directory
            sanitize: Whether to sanitize sensitive information
            html_include_raw_response: Whether HTML includes the raw response
        """
        self.output_dir = Path(output_dir)
        self.sanitize = sanitize
        self.html_include_raw = html_include_raw_response
        
        # Generate session ID
        self.session_id = self._generate_session_id()
        
        # Event cache (used for generating stats and final HTML)
        self._events: List[Dict] = []
        
        # Ensure directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # JSONL file path
        self.jsonl_path = self.output_dir / f"trace-{self.session_id}.jsonl"
        
        # Open JSONL file (streaming write)
        self.jsonl_file = open(self.jsonl_path, 'w', encoding='utf-8')
        
        # HTML file path
        self.html_path = self.output_dir / f"trace-{self.session_id}.html"
        
        # Open HTML file (incremental write)
        self.html_file = open(self.html_path, 'w', encoding='utf-8')
        
        # Write HTML header
        self._write_html_header()
    
    def _generate_session_id(self) -> str:
        """Generate session ID
        
        Format: s-YYYYMMDD-HHMMSS-xxxx
        Example: s-20250118-143052-a3f2
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        random_suffix = uuid.uuid4().hex[:4]
        return f"s-{timestamp}-{random_suffix}"
    
    def log_event(
        self,
        event: str,
        payload: Dict[str, Any],
        step: Optional[int] = None
    ):
        """Log event
        
        Args:
            event: Event type (session_start, tool_call, tool_result, etc.)
            payload: Event data
            step: Step number of the ReAct loop (optional)
        """
        # Construct event object
        event_obj = {
            "ts": datetime.now().isoformat(),
            "session_id": self.session_id,
            "step": step,
            "event": event,
            "payload": payload
        }
        
        # Sanitize
        if self.sanitize:
            event_obj = self._sanitize_event(event_obj)
        
        # Append to cache
        self._events.append(event_obj)
        
        # Stream write to JSONL
        self.jsonl_file.write(json.dumps(event_obj, ensure_ascii=False) + "\n")
        self.jsonl_file.flush()
        
        # Incrementally write HTML event fragment
        self._write_html_event(event_obj)
    
    def _sanitize_event(self, event: Dict) -> Dict:
        """Sanitize sensitive information

        Sanitization rules:
        - API Key: sk-xxx, Bearer xxx -> sk-***, Bearer ***
        - Username in paths: /Users/xxx/ -> /Users/***/
        """
        import copy
        event = copy.deepcopy(event)

        # Recursively sanitize payload
        event["payload"] = self._sanitize_value(event.get("payload", {}))

        return event

    def _sanitize_value(self, value: Any) -> Any:
        """Recursively sanitize value

        Args:
            value: Value to be sanitized (could be string, dict, list, etc.)

        Returns:
            Sanitized value
        """
        if isinstance(value, str):
            # Sanitize string
            # API Key: sk-xxx -> sk-***
            value = re.sub(r'sk-[a-zA-Z0-9]+', 'sk-***', value)
            # Bearer Token: Bearer xxx -> Bearer ***
            value = re.sub(r'Bearer\s+[a-zA-Z0-9_\-]+', 'Bearer ***', value)
            # Username in paths
            value = re.sub(r'(/Users/|/home/|C:\\Users\\)[^/\\]+', r'\1***', value)
            return value
        elif isinstance(value, dict):
            # Recursively process dictionary
            return {k: self._sanitize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            # Recursively process list
            return [self._sanitize_value(item) for item in value]
        else:
            # Return other types directly
            return value

    def finalize(self):
        """Generate final HTML and close files

        Steps:
        1. Compute statistics
        2. Write HTML footer (including stats panel)
        3. Close all files
        """
        # Compute statistics
        stats = self._compute_stats()

        # Write HTML footer (stats panel + script)
        self._write_html_footer(stats)

        # Close files
        self.jsonl_file.close()
        self.html_file.close()

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute statistics

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_steps": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "tool_calls": {},  # {tool_name: count}
            "errors": [],
            "duration_seconds": 0.0,
            "model_calls": 0,
        }

        session_start = None
        session_end = None

        for event in self._events:
            # Session duration
            if event["event"] == "session_start":
                session_start = datetime.fromisoformat(event["ts"])
            if event["event"] == "session_end":
                session_end = datetime.fromisoformat(event["ts"])

            # Step count
            if event.get("step"):
                stats["total_steps"] = max(stats["total_steps"], event["step"])

            # Token statistics
            if event["event"] == "model_output":
                usage = event.get("payload", {}).get("usage", {})
                stats["total_tokens"] += usage.get("total_tokens", 0)
                stats["total_cost"] += usage.get("cost", 0.0)
                stats["model_calls"] += 1

            # Tool call statistics
            if event["event"] == "tool_call":
                tool_name = event["payload"].get("tool_name", "unknown")
                stats["tool_calls"][tool_name] = stats["tool_calls"].get(tool_name, 0) + 1

            # Error statistics
            if event["event"] == "error":
                stats["errors"].append({
                    "step": event.get("step"),
                    "type": event["payload"].get("error_type"),
                    "message": event["payload"].get("message")
                })

        # Compute duration
        if session_start and session_end:
            stats["duration_seconds"] = (session_end - session_start).total_seconds()

        return stats

    def _write_html_header(self):
        """Write HTML header"""
        header = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trace: {self.session_id}</title>
    <style>
        body {{
            font-family: 'Consolas', 'Monaco', monospace;
            padding: 20px;
            background: #1a1a1a;
            color: #e0e0e0;
            margin: 0;
        }}
        .header {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            color: #4af626;
        }}
        .stats-panel {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-item {{
            background: #1a1a1a;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #4af626;
        }}
        .stat-label {{
            display: block;
            color: #888;
            font-size: 12px;
            margin-bottom: 5px;
        }}
        .stat-value {{
            display: block;
            color: #e0e0e0;
            font-size: 24px;
            font-weight: bold;
        }}
        .tool-stats {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        .tool-stats th, .tool-stats td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        .tool-stats th {{
            color: #4af626;
        }}
        .error-list {{
            list-style: none;
            padding: 0;
        }}
        .error-list li {{
            background: #331111;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #ff4444;
        }}
        .events-container {{
            background: #2a2a2a;
            padding: 20px;
            border-radius: 8px;
        }}
        .event {{
            border: 1px solid #333;
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
            background: #1a1a1a;
        }}
        .event-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .step {{
            color: #888;
            font-size: 12px;
        }}
        .timestamp {{
            color: #666;
            font-size: 11px;
        }}
        .event-type {{
            color: #4af626;
            font-weight: bold;
        }}
        .expandable {{
            cursor: pointer;
            color: #4af626;
            user-select: none;
        }}
        .expandable:hover {{
            color: #6fff48;
        }}
        .details {{
            display: none;
            margin-top: 10px;
            padding: 10px;
            background: #0d0d0d;
            border-radius: 5px;
            overflow-x: auto;
        }}
        .details pre {{
            margin: 0;
            color: #e0e0e0;
        }}
        .tool-call {{
            border-left: 3px solid #4af626;
        }}
        .tool-result {{
            border-left: 3px solid #ffd700;
        }}
        .error {{
            border-left: 3px solid #ff4444;
            background: #2a1a1a;
        }}
        .model-output {{
            border-left: 3px solid #00bfff;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Trace Session: {self.session_id}</h1>
        <p>Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="events-container">
        <h2>📋 Event List</h2>
"""
        self.html_file.write(header)
        self.html_file.flush()

    def _write_html_event(self, event: Dict):
        """Write HTML fragment for a single event (incremental write)"""
        event_type = event["event"]
        step = event.get("step", "")
        timestamp = event["ts"]
        payload = event.get("payload", {})

        # Determine CSS class
        css_class = "event"
        if event_type == "tool_call":
            css_class += " tool-call"
        elif event_type == "tool_result":
            css_class += " tool-result"
        elif event_type == "error":
            css_class += " error"
        elif event_type == "model_output":
            css_class += " model-output"

        # Generate unique ID
        details_id = f"details-{len(self._events)}"

        # Format payload
        payload_json = json.dumps(payload, indent=2, ensure_ascii=False)

        # Generate event HTML
        event_html = f"""
        <div class="{css_class}">
            <div class="event-header">
                <span class="step">Step {step if step else '-'}</span>
                <span class="timestamp">{timestamp}</span>
                <span class="event-type">{event_type}</span>
                <span class="expandable" onclick="toggleDetails('{details_id}')">[▼ Details]</span>
            </div>
            <div id="{details_id}" class="details">
                <pre>{payload_json}</pre>
            </div>
        </div>
"""
        self.html_file.write(event_html)
        self.html_file.flush()

    def _write_html_footer(self, stats: Dict[str, Any]):
        """Write HTML footer (stats panel + script)"""
        # Build tool call stats table
        tool_stats_rows = ""
        for tool_name, count in sorted(stats["tool_calls"].items(), key=lambda x: x[1], reverse=True):
            tool_stats_rows += f"<tr><td>{tool_name}</td><td>{count}</td></tr>\n"

        # Build error list
        error_list_html = ""
        if stats["errors"]:
            error_items = ""
            for error in stats["errors"]:
                step = error.get("step", "?")
                error_type = error.get("type", "UNKNOWN")
                message = error.get("message", "")
                error_items += f"<li>Step {step}: <strong>{error_type}</strong> - {message}</li>\n"
            error_list_html = f"""
        <h3>❌ Error List ({len(stats["errors"])})</h3>
        <ul class="error-list">
            {error_items}
        </ul>
"""

        footer = f"""
    </div>

    <div class="stats-panel">
        <h2>📊 Session Statistics</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <span class="stat-label">Total Steps</span>
                <span class="stat-value">{stats["total_steps"]}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Tokens</span>
                <span class="stat-value">{stats["total_tokens"]:,}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Total Cost</span>
                <span class="stat-value">${stats["total_cost"]:.4f}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Session Duration</span>
                <span class="stat-value">{stats["duration_seconds"]:.1f}s</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Model Calls</span>
                <span class="stat-value">{stats["model_calls"]}</span>
            </div>
        </div>

        <h3>🔧 Tool Call Statistics</h3>
        <table class="tool-stats">
            <tr><th>Tool Name</th><th>Call Count</th></tr>
            {tool_stats_rows if tool_stats_rows else '<tr><td colspan="2">No tool calls</td></tr>'}
        </table>

        {error_list_html}
    </div>

    <script>
        function toggleDetails(id) {{
            const el = document.getElementById(id);
            if (el.style.display === 'none' || el.style.display === '') {{
                el.style.display = 'block';
            }} else {{
                el.style.display = 'none';
            }}
        }}
    </script>
</body>
</html>
"""
        self.html_file.write(footer)
        self.html_file.flush()

    def __enter__(self):
        """Context manager: enter"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: exit (auto finalize)"""
        # If an exception occurs, log error event
        if exc_type is not None:
            self.log_event(
                "error",
                {
                    "error_type": exc_type.__name__,
                    "message": str(exc_val),
                    "stacktrace": str(exc_tb)
                }
            )

        # Auto finalize
        self.finalize()

        # Do not suppress exception
        return False