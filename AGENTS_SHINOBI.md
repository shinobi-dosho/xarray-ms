# shinobi-dosho -- conventions for coding agents

Organisation-wide baseline. Every repository here also has its own
`AGENTS.md` carrying the design conventions specific to it -- read that
one too, and where the two disagree, **the repository's file wins**. This
file only holds what is true across all of them.

## Core rule

Avoid unnecessary complexity like the plague.

Prefer the boring construct that a reader understands on sight. A
mechanism earns its place by removing more complexity than it adds, and a
feature nobody asked for is a liability, not a head start. When a
repository's `AGENTS.md` says a thing is deliberately left out, it is
left out on purpose -- do not helpfully add it back.

## Toolchain

The projects use [uv](https://docs.astral.sh/uv/), `ruff` for lint and
format, and `pytest`. From a fresh clone:

```bash
uv sync --group dev
uv run pytest
uv run ruff check .

# enable the repo's pre-commit hook (once per clone)
git config core.hooksPath .githooks
```

`.githooks/pre-commit` is a tracked shell script, not the `pre-commit`
framework, and runs the project's own pinned ruff -- so it agrees with CI
by construction. It never rewrites files behind your back. CI runs
`uv sync --locked`, so a dependency change means committing the updated
`uv.lock`.

## Tests describe behaviour, not construction

A test that asserts an object was built without error has tested nothing
a reviewer cares about. Assert the shape that a real caller depends on --
the argv a cab produces, the schema a step exposes, the values a reader
returns from a real file. A bug fix comes with a regression test that
fails without the fix.

## Attribution: commit trailers yes, PR trailers no

A commit made with an assistant's help says so in a trailer on the
**commit message**, naming the assistant and the model behind it:

```
Assisted-by: <LLM> <MODEL>
```

-- e.g. `Assisted-by: Claude Opus 5`, `Assisted-by: Codex GPT-5`. One
line, last in the message, after any `Co-authored-by:` for real people.

Agents default to a `Co-authored-by:` trailer with an address; replace
it. Both halves of that default are wrong here. `Co-authored-by:` claims
more than happened -- these tools assist, and the person who ran them
owns the change and answers for it. The address is what makes the claim
bite: GitHub renders a trailer as co-authorship only when one is
present, so `Co-authored-by: Claude <noreply@anthropic.com>` attaches a
vendor to the authorship of work this organisation owns. `Assisted-by:`
with no address records the same fact as plain text and attaches nobody.

This reverses the rule as it stood between 5b9eeb8 and now, which argued
that co-authorship was the honest description of tools doing real work.
The consideration that changed it is ownership, not accuracy of credit:
these are paid services operating on our IP, and the history should not
carry anything a vendor could read as a stake in it.

**Pull request descriptions carry no trailer at all** -- no
`Assisted-by:`, no `Co-authored-by:`, no "Generated with", no tool
badge. A PR body is review material: it exists to tell a reviewer what
changed and why, and what to check. Provenance already lives on every
commit the PR contains, where it is attached to the specific change
rather than repeated once per PR, so a trailer in the description is
duplication in the one place that has no room for it. Agents default to
adding one; delete it.

Neither form is a substitute for the message itself. A commit that
explains a decision badly does not improve by naming the model that
helped make it -- see the existing history for the standard: what
changed, what it deviates from and why, and what a reviewer should not
assume held still.

## Reviewing changes: check the tree, not just the diff

A diff shows what moved; it does not show what the change made wrong
elsewhere. Before calling work done, look at the surrounding module, the
docs that describe the behaviour you changed, and the tests that assumed
the old shape. A change that leaves the documentation describing
something that is no longer true is not finished.
