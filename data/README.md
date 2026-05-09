# Data

Two benchmarks used by the paper. Both are author-constructed, English-only,
small, and intended for matched-pair / matched-triplet evaluation rather than
broad coverage.

## causal_pairs.json

50 matched pairs across 5 categories of causal reasoning (10 per category):
confounding, intervention, counterfactual, simpsons_paradox, mediation.

Each pair shares a premise. The `causal` question requires reasoning about
interventions or confounders; the `correlational` question asks what
association the premise reports. Both are 3-option multiple choice with the
correct answer at a balanced position.

Schema:
```
[
  {
    "id": "pair_000",
    "category": "confounding",
    "causal": {
      "question": "...",
      "options": ["...", "...", "..."],
      "correct_index": 0
    },
    "correlational": {
      "question": "...",
      "options": ["...", "...", "..."],
      "correct_index": 0
    }
  },
  ...
]
```

## hallucination_triples.json

15 factual questions, each presented in three context conditions: no context,
relevant (supports the correct answer), misleading (emphasises a plausible but
incorrect alternative without explicit falsehood). Relevant and misleading
contexts are length-matched within +/-20%.

Schema:
```
[
  {
    "id": "item_000",
    "category": "geography",
    "question": "What is the capital of Australia?",
    "correct_answer": "Canberra",
    "context_relevant": "Canberra is Australia's capital, where Parliament House is located.",
    "context_misleading": "Sydney is Australia's largest city with the Opera House. Melbourne is the cultural capital."
  },
  ...
]
```

The `category` field is a coarse topical label (geography / science / history)
provided for convenience; the experiments do not use it.
