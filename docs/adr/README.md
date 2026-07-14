# Architecture Decision Records

Every architectural decision gets a numbered ADR here **before implementation**
(house rule). ADRs are immutable once accepted; a decision is changed by a new
ADR that supersedes the old one.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-scope.md) | Project scope: corpus domains, corpus size, model sizes, GPU budget | Accepted |
| [0002](0002-chunking.md) | Chunking policy: one title+abstract chunk per document | Accepted |
| [0003](0003-tokenizer-corpus-mix.md) | Tokenizer training corpus mix and vocabulary size | Accepted |
| [0004](0004-training-curriculum.md) | Dense retriever training curriculum (MLM → contrastive → domain) | Accepted |

## Template

```markdown
# ADR-NNNN: Title

- **Status:** Proposed | Accepted | Superseded by ADR-XXXX
- **Date:** YYYY-MM-DD

## Context
Why a decision is needed; the forces at play.

## Decision
The choice made, stated imperatively.

## Alternatives considered
What else was on the table and why it lost.

## Consequences
What becomes easier/harder; follow-up work created.
```
