---
description: Answer a question in 3-8 bullet points, changing nothing
argument-hint: [question]
allowed-tools: ["Read", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"]
---

# Ask

Answer the question below. This is a question, not a task — nothing is being
requested except the answer.

<question>
$ARGUMENTS
</question>

## Rules

- **Answer in 3 to 8 bullet points.** Never fewer than 3, never more than 8. The
  bullets are the entire answer: no preamble, no lead-in sentence, no closing
  summary, no "let me know if" offer.
- **Change nothing.** No edits, no new files, no commits, no installs, no
  training runs or other long-running jobs. Shell access is for *inspecting*
  (`ls`, `grep`, `git log`, `cat`) — never for writing, moving, or deleting.
- **Investigate before answering** whenever the answer depends on this
  repository or on live state. Read the actual file, run the actual query. Do
  not answer a repo-specific question from memory or inference when it can be
  checked, and do not present a guess as a finding.
- **One idea per bullet**, claim first. Put the conclusion in the opening
  clause and the justification after it, so the answer is skimmable.
- **Cite `path/to/file.py:123`** whenever a claim comes from the codebase, so
  it can be verified without a second search.
- **Say plainly when you don't know**, or when answering properly would need
  something you were not able to check. An honest gap is a valid bullet; a
  confident invention is not.
- **If the answer is "you should change X"**, describe the change and where it
  goes — then stop. Proposing is answering; applying is not.

If the question is broad enough that 8 bullets cannot cover it, spend the
bullets on the parts that most change what the reader would do, and say in one
of them what you left out.
