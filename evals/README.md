# Switchboard Evaluation Module

Evaluation framework for testing Switchboard agent with mocked calendar data.

## Structure

```
evals/
├── fixtures.py          # Mock calendar events organized by date
├── mock_calendar.py     # Dynamic mock client (returns different data based on query)
├── dataset.json         # 10 evaluation examples
├── run_agent.py         # Agent runner with mocked calendar (frozen clock)
├── evaluators.py        # LLM-as-judge accuracy evaluator (LOCAL only)
├── evaluators_code.py   # Sandbox-safe code evaluators (UPLOADABLE)
└── run_eval.py          # Main evaluation script
```

## Quick Start

```bash
# Run evaluation locally (registered console script)
uv run switchboard-eval

# Or run the module directly
uv run python -m evals.run_eval
```

Results will appear in LangSmith under experiment prefix `switchboard-eval`.

## How It Works

### 1. Dynamic Mocking

The `MockCalendarClient` returns different events based on query parameters:

- **Date-based**: `list_events(time_min="2026-04-29T...")` → returns TODAY_EVENTS
- **Search-based**: `search_events(query="Alex")` → returns events matching "Alex"
- **ID-based**: `get_event(event_id="event1")` → returns specific event

### 2. Fixed Evaluation Time

All evaluations run with a fixed timestamp: **2026-04-29T17:00:00+09:00**

This is wired through `build_agent(now_provider=lambda: EVAL_TIME)` in
`run_agent.py`, which causes the agent's `dynamic_prompt` middleware to embed
the frozen time into every system prompt instead of `datetime.now()`. As a
result:

- "today" always resolves to April 29
- "tomorrow" always resolves to April 30
- "Friday" always resolves to May 2
- Results are reproducible regardless of the wall clock

### 3. Dataset Coordination

Each dataset example is coordinated with mock data:

```json
{
  "inputs": {"query": "What's on my calendar today?"},
  "outputs": {"response": "You have 2 events today: Team Meeting at 10:00 AM..."}
}
```

The mock returns TODAY_EVENTS (Team Meeting + Lunch with Alex), and the expected output describes those events.

### 4. Evaluators

- **accuracy_evaluator**: LLM-as-judge checks if response conveys correct information
- **response_length_evaluator**: Checks if response is concise (<= 200 chars)

## Uploading to LangSmith

### Upload Dataset

```bash
langsmith dataset upload evals/dataset.json \
  --name "Switchboard Eval" \
  --api-key $LANGSMITH_API_KEY
```

### Upload Evaluators (Optional)

Sandbox-safe code evaluators live in **`evals/evaluators_code.py`** (stdlib
imports only, no top-level side effects). They can be uploaded to auto-run
on experiments:

```bash
langsmith evaluator upload evals/evaluators_code.py \
  --name "Response Length" \
  --function response_length_evaluator \
  --dataset "Switchboard Eval" \
  --replace \
  --api-key $LANGSMITH_API_KEY
```

**Note**: `evals/evaluators.py` contains the LLM-as-judge `accuracy_evaluator`,
which imports `langchain` and `Settings`. It CANNOT be uploaded to the LangSmith
sandbox — run it locally via `evaluate(evaluators=[...])`.

## Adding New Test Cases

1. **Add mock data** to `fixtures.py` (if needed)
2. **Add dataset example** to `dataset.json` with expected output
3. **Verify coordination**: Run the agent manually to confirm mock returns expected data

Example:

```python
# 1. Add to fixtures.py
MONDAY_EVENTS = [
    {"summary": "Sprint Planning", "start": {"dateTime": "2026-05-05T09:00:00+09:00"}, ...}
]

# 2. Add to dataset.json
{
  "inputs": {"query": "What's happening Monday morning?"},
  "outputs": {"response": "On Monday morning, you have Sprint Planning at 9:00 AM."}
}

# 3. Update mock_calendar.py to return MONDAY_EVENTS for 2026-05-05
```

## Troubleshooting

**Agent returns wrong data:**
- Check that mock returns expected events for the query
- Verify fixed eval time (2026-04-29T17:00:00+09:00) resolves dates correctly
- Print `run_outputs` in evaluator to inspect actual response

**Evaluator fails:**
- Ensure output schema matches: `{"response": "..."}`
- Check both RunTree and dict access patterns are handled

**Low accuracy scores:**
- Review LLM judge comments in LangSmith
- Adjust expected outputs to be more flexible (semantic match, not exact)
- Consider if mock data needs to be richer
