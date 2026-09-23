# Jev 1.13 event-trigger evaluation prompt

## Request format

Jev is called through OpenRouter's typed Decisions API. It is not asked to generate annotation JSON. Each request contains:

- `state`: the entire document as a plain numbered-token string, exactly as `[0] token\n[1] token\n...`;
- `questions`: up to 128 separate `noul` yes/no questions, one per candidate token ID in that batch;
- each answer: a probability that the answer to that candidate question is yes.

There are no examples and no gold answers. Every token position, including nonverbal and punctuation tokens, is included as a candidate, so the model is not given gold trigger locations or a shortlist derived from the annotations. Documents are not split into independent text windows; the full document state is repeated in each batch request.

## Exact state prefix

```text
PASSAGE TOKENS (IDs are zero-based and inclusive):
[0] {token_0}
[1] {token_1}
...
```

The placeholders above show the format only; at runtime they are replaced with the document's complete token sequence. Book text is not included in this prompt file.

## Exact per-token question template

For candidate token `i` with surface form `token`, the code submits a `noul` question whose instruction is:

```text
Is token ID {i}, whose surface text is {token!r}, the lexical trigger of a specific event asserted to actually occur in this passage, with participants at a specific time? Answer yes only for an asserted realis event. Exclude hypothetical, intended, future, counterfactual, and nonspecific narrated events.
```

`{token!r}` is Python's repr of that token. Each question key is `t{i}`. The model-facing meaning is in the instruction text, not the key. Question IDs and instructions are generated independently of the gold annotations.

## Decision rule

The probability returned as `answers.t{i}.noul` is converted to a positive one-token event span when it is **at least 0.5**. The threshold was fixed before the held-out run and was not tuned on the test set. Missing answers are uncovered and count as negative predictions. No answers were missing in the completed test cohort.

## API and generation record

- Endpoint: `POST https://openrouter.ai/api/alpha/decisions`
- Requested model: `typesafe/jev-1.13`
- Resolved model in the measured run: `typesafe/jev-1.13-20260917`
- Primitive: `noul` (yes probability)
- Questions per request: 128, except the final partial batch of a document
- Temperature/sampling: not exposed by the typed Decisions API; no chat-generation parameters were sent
- BookNLP test set: the same official 30-document literary-event test split used by the local models
- Local run cache and aggregate metrics: `cache/openrouter_jev/test/`, `logs/jev_test_requests.jsonl`, and `results/openrouter_jev/test_events.json`
