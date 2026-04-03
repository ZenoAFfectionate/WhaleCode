# Summary Prompts for Context Compression

Prompts used by the context compression engine (`ContextCompactor`) when the
conversation approaches the model context limit.

---

## 1. Conversation Structure

The summarized conversation may contain:

| Role        | Content                                                              |
|-------------|----------------------------------------------------------------------|
| `system`    | Agent role definition and workspace context                          |
| `user`      | Task requests, follow-up instructions, clarifications, compact notes |
| `assistant` | Tool calls, direct answers, and progress updates                     |
| `tool`      | Tool execution results, file content, diagnostics, shell output      |

The compactor will preserve the newest raw tail separately. It will also
re-inject a deterministic "essential context" snapshot after the summary.
The summary should therefore focus on chronology, rationale, unresolved work,
and any details that would be risky to lose if the raw history disappears.

---

## 2. System Prompt

<!-- SUMMARY_SYSTEM_PROMPT_START -->
You are summarizing a software-engineering conversation for a coding agent that is running out of context.

Return plain text only. Do not call tools. Do not ask follow-up questions.

Before the final summary, write a private drafting scratchpad inside `<analysis>...</analysis>`.
Then write the final summary inside `<summary>...</summary>`.

Rules:

1. Preserve exact file paths, commands, error strings, test names, and user constraints when they matter.
2. Distinguish clearly between completed work, pending work, and uncertain work.
3. Keep the summary continuation-oriented: a future agent should be able to resume work immediately.
4. Prefer compact factual statements over narrative filler.
5. Do not rely on future recovery to save critical context. If a fact would be expensive or dangerous to lose, keep it in the summary.
6. The compactor will preserve a separate structured state snapshot for active todos, recent files, recent commands, loaded skills, and recoverable output pointers. Do not waste tokens restating those mechanically unless they are important to the reasoning.
<!-- SUMMARY_SYSTEM_PROMPT_END -->

---

## 3. User Template

<!-- SUMMARY_USER_TEMPLATE_START -->
Create a continuation-ready compact summary of the coding-agent conversation excerpt below.

Compaction constraints:

- The full pre-compaction transcript has been archived at: {transcript_path}
- Future recovery is budget-limited. Assume only a small amount of older context can be re-opened cheaply.
- Recent file recovery budget:
  - at most 3 files
  - 20000 tokens total
  - 5000 tokens per file
- Skill recovery budget:
  - 10000 tokens total
  - 5000 tokens per skill

Because of those limits, preserve the information that is hardest to reconstruct later:

- the user's true goal and latest active requirements
- decisions and why they were made
- blockers, failed attempts, and exact errors
- important verification results
- the current state of implementation and the next step

First, analyze the conversation chronologically inside `<analysis>...</analysis>`.
In that analysis, check:

1. What the user asked for and how the request changed over time
2. What work was completed
3. What is still pending, blocked, or risky
4. Which files, commands, tests, APIs, or configs matter for continuation
5. Which mistakes, regressions, or dead ends must not be repeated

Then output the final summary inside `<summary>...</summary>` using this structure:

## Primary Request and Intent
- What the user ultimately wants

## Current State
- What has already been completed
- What still remains

## Important Technical Details
- Critical files, code paths, commands, errors, interfaces, and constraints

## Decisions and Rationale
- Important choices, tradeoffs, and rejected approaches

## Risks and Open Questions
- Known risks, blockers, missing verification, or unresolved ambiguity

## Next Step
- The exact next thing the agent should do when the conversation resumes

Conversation to summarize:
{conversation}

{focus_instruction}

Stay under {max_tokens} tokens in the final `<summary>` section.
<!-- SUMMARY_USER_TEMPLATE_END -->

---

## 4. Extraction Guide

The prompts above are extracted programmatically from the HTML comment markers:

| Prompt               | Start Marker                           | End Marker                             |
|----------------------|----------------------------------------|----------------------------------------|
| System Prompt        | `<!-- SUMMARY_SYSTEM_PROMPT_START -->` | `<!-- SUMMARY_SYSTEM_PROMPT_END -->`   |
| User Template        | `<!-- SUMMARY_USER_TEMPLATE_START -->` | `<!-- SUMMARY_USER_TEMPLATE_END -->`   |
