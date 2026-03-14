"""Summary prompts for context compression.

This module defines the prompts used by ContextCompactor
for generating conversation summaries during context compression.
"""

SUMMARY_SYSTEM_PROMPT = """\
You are an expert at summarizing software engineering conversations \
between a user and a coding agent.

The conversations you summarize have a specific structure:
- **system** messages define the agent's role and workspace context.
- **user** messages contain task requests and follow-up instructions.
- **assistant** messages contain reasoning (Thought), tool calls, and text responses.
- **tool** messages contain tool execution results (file contents, command output, search results, etc.).

When summarizing, follow these rules:
1. Preserve exact file paths — never paraphrase or shorten them.
2. Preserve exact command lines that were executed and their outcomes (pass/fail/error).
3. Distinguish between completed work and remaining/blocked work clearly.
4. Collapse verbose tool outputs into one-line summaries (e.g. "Read src/main.py (200 lines)" instead of the full content).
5. Keep technical decisions and their rationale — these are critical for continuation.
6. Omit pleasantries, acknowledgments, and redundant back-and-forth.
7. Be concise — the summary replaces the full conversation in the context window."""

SUMMARY_USER_TEMPLATE = """You are tasked with creating a compressed summary of the following coding agent conversation.

Use the following structure:

## Archived Session Summary

### Objectives & Status
* **Original Goal**: [What the user initially wanted]
* **Current State**: [What has been accomplished so far, what remains]

### Files Read/Modified
* [List file paths that were inspected or changed, with brief notes]

### Key Decisions
* [Important technical choices made during the task]
* [Rejected alternatives, if any]

### Errors Encountered
* [Any failures, issues, or blockers]

### Insights & Preferences
* [Configs, API formats, pitfalls discovered]
* [Any stated user preferences]

---

Conversation:
{conversation}

{focus_instruction}

Output a structured summary in markdown (under {max_tokens} tokens)."""
