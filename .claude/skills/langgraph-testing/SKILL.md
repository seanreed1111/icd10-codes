---
name: langgraph-testing
description: Writes tests for LangGraph agents and graphs with pytest. Use when testing state machines, graph nodes, agent flows, or partial execution paths.
---

# LangGraph Testing with Pytest

**Core principle:** Test graph behavior at multiple levels — individual nodes, partial flows, and complete executions. Use MemorySaver for test isolation.

> "Create your graph before each test where you use it, then compile it within tests with a new checkpointer instance." — LangGraph Docs

**Why this matters:** LangGraph agents are stateful. Each test needs isolated state to prevent cross-contamination. The checkpointer enables partial execution testing and state inspection.

## Testing Levels

| Level       | What to Test                  | Tool                              |
| ----------- | ----------------------------- | --------------------------------- |
| Node        | Individual node logic         | `graph.nodes["name"].invoke()`    |
| Partial     | Subset of graph flow          | `update_state()` + `interrupt_after` |
| End-to-End  | Complete graph execution      | `graph.invoke()` with thread_id   |
| Integration | Real LLMs with recording      | `pytest-recording` / `vcrpy`      |

## Graph Factory Pattern

Always create graphs fresh in tests:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

def create_graph() -> StateGraph:
    class MyState(TypedDict):
        messages: list
        status: str

    graph = StateGraph(MyState)
    graph.add_node("process", process_node)
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "process")
    graph.add_edge("process", "validate")
    graph.add_edge("validate", END)
    return graph

def test_graph_execution():
    checkpointer = MemorySaver()  # Fresh checkpointer per test
    graph = create_graph()
    compiled = graph.compile(checkpointer=checkpointer)

    result = compiled.invoke(
        {"messages": [], "status": "pending"},
        config={"configurable": {"thread_id": "test-1"}}
    )
    assert result["status"] == "completed"
```

## When to Use Each Test Type

```
Testing individual node logic? → Node test (bypasses checkpointer)
Testing specific subflow? → Partial execution test
Testing complete workflow? → End-to-end test
Testing with real LLM? → Integration test with recording
```

## Mocking Strategy

**Default: Use `GenericFakeChatModel` for deterministic tests**

**Only use real LLMs:**
- Integration/E2E tests with HTTP recording
- Validating prompt changes

**Mock utilities:**
- `GenericFakeChatModel` — Mock text/tool responses
- `pytest-recording` — Record/replay HTTP calls
- `MemorySaver` — In-memory checkpointer for tests

## Test Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
testpaths = ["tests"]
```

## Anti-Patterns

| Pattern                          | Fix                                    |
| -------------------------------- | -------------------------------------- |
| Shared checkpointer across tests | Create fresh MemorySaver per test      |
| Hardcoded thread_ids             | Use unique IDs or parametrize          |
| Testing node via full graph      | Use `graph.nodes["name"].invoke()`     |
| No interrupt points in flow test | Use `interrupt_after` for partial tests|
| Real LLM calls without recording | Add `@pytest.mark.vcr()` decorator     |

## Quality Checklist

- [ ] Graph created fresh per test (factory pattern)
- [ ] MemorySaver used for test checkpointing
- [ ] Thread IDs unique per test
- [ ] LLM responses mocked or recorded
- [ ] Node tests isolated from graph flow
- [ ] State transitions verified
- [ ] Error conditions handled

## Language-Specific Patterns

- **Python**: See [references/python.md](references/python.md)

---

**Remember:** Isolated state. Fresh graphs. Test at the right level.
