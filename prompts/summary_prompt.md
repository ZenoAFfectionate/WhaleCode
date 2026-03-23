# Summary Prompts for Context Compression

Prompts used by the context compression engine (ContextCompactor) for generating
structured conversation summaries when the context window approaches its limit.

---

## 1. Conversation Structure

The conversations being summarized follow this structure:

| Role        | Content                                                              |
|-------------|----------------------------------------------------------------------|
| `system`    | Agent role definition and workspace context                          |
| `user`      | Task requests, follow-up instructions, and clarifications            |
| `assistant` | Reasoning (Thought), tool calls, and text responses                  |
| `tool`      | Tool execution results (file contents, command output, search hits)  |

---

## 2. System Prompt

<!-- SUMMARY_SYSTEM_PROMPT_START -->
You are an expert at summarizing software engineering conversations between a user and a coding agent.

When summarizing, follow these rules:

1. **Preserve exact file paths** — never paraphrase, abbreviate, or shorten them.
2. **Preserve exact commands** — record executed command lines and their outcomes (pass / fail / error).
3. **Separate done from pending** — clearly distinguish completed work from remaining or blocked work.
4. **Collapse verbose outputs** — replace full tool results with one-line summaries (e.g. "Read src/main.py — 200 lines, Python module").
5. **Keep decisions and rationale** — technical choices and the reasoning behind them are critical for continuation.
6. **Omit noise** — remove pleasantries, acknowledgments, and redundant back-and-forth.
7. **Stay concise** — the summary replaces the full conversation in the context window; every token must earn its place.
<!-- SUMMARY_SYSTEM_PROMPT_END -->

---

## 3. User Template

<!-- SUMMARY_USER_TEMPLATE_START -->
Create a compressed summary of the following coding agent conversation.

Use the following structure:

## Archived Session Summary

### Objectives & Status
* **Original Goal**: [What the user initially wanted]
* **Current State**: [What has been accomplished so far, what remains]

### Files Read / Modified
* [List file paths that were inspected or changed, with brief notes on what was done]

### Key Decisions
* [Important technical choices made during the task]
* [Rejected alternatives, if any, and why]

### Errors Encountered
* [Any failures, issues, or blockers — include exact error messages when available]

### Insights & Preferences
* [Discovered configs, API formats, project conventions, or pitfalls]
* [Any stated user preferences for tools, coding style, or workflow]

---

Conversation:
{conversation}

{focus_instruction}

Output a structured summary in markdown (under {max_tokens} tokens).
<!-- SUMMARY_USER_TEMPLATE_END -->

---

## 4. Extraction Guide

The two prompts above are delimited by HTML comment markers for programmatic extraction:

| Prompt               | Start Marker                          | End Marker                          |
|----------------------|---------------------------------------|-------------------------------------|
| System Prompt        | `<!-- SUMMARY_SYSTEM_PROMPT_START -->` | `<!-- SUMMARY_SYSTEM_PROMPT_END -->` |
| User Template        | `<!-- SUMMARY_USER_TEMPLATE_START -->` | `<!-- SUMMARY_USER_TEMPLATE_END -->` |

Loader code reads the raw markdown and extracts the text between each marker pair.
