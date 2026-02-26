# LangGraph Python Testing Patterns

Patterns for testing LangGraph agents and graphs with `pytest`, including node testing, partial execution, mocking LLMs, and HTTP recording.

## Contents

- [Dependencies](#dependencies)
- [Pytest Configuration](#pytest-configuration)
- [Graph Factory Pattern](#graph-factory-pattern)
- [Testing Individual Nodes](#testing-individual-nodes)
- [End-to-End Execution Tests](#end-to-end-execution-tests)
- [Partial Execution Tests](#partial-execution-tests)
- [Mocking LLM Responses](#mocking-llm-responses)
- [Testing Tool Calls](#testing-tool-calls)
- [Async Graph Testing](#async-graph-testing)
- [HTTP Recording for Integration Tests](#http-recording-for-integration-tests)
- [Testing Interrupts and Human-in-the-Loop](#testing-interrupts-and-human-in-the-loop)
- [State Assertions](#state-assertions)
- [Fixtures Reference](#fixtures-reference)

## Dependencies

```bash
uv add pytest pytest-asyncio langgraph langchain-core
# Optional for HTTP recording
uv add pytest-recording vcrpy
```

## Pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "-ra -q"
testpaths = ["tests"]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
markers = [
    "integration: marks tests requiring real LLM calls",
]
```

## Graph Factory Pattern

Create graphs fresh for test isolation. Define the graph structure once, compile per test with a new checkpointer.

```python
import pytest
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


class OrderState(TypedDict):
    items: list[str]
    total: float
    status: str


def create_order_graph() -> StateGraph:
    """Factory function to create a fresh graph."""
    def add_item(state: OrderState) -> dict:
        return {"status": "item_added"}

    def calculate_total(state: OrderState) -> dict:
        total = len(state["items"]) * 5.99
        return {"total": total, "status": "calculated"}

    graph = StateGraph(OrderState)
    graph.add_node("add_item", add_item)
    graph.add_node("calculate", calculate_total)
    graph.add_edge(START, "add_item")
    graph.add_edge("add_item", "calculate")
    graph.add_edge("calculate", END)
    return graph


@pytest.fixture
def order_graph():
    """Fixture providing compiled graph with fresh checkpointer."""
    checkpointer = MemorySaver()
    graph = create_order_graph()
    return graph.compile(checkpointer=checkpointer)
```

## Testing Individual Nodes

Test nodes in isolation using `graph.nodes["name"].invoke()`. This bypasses checkpointers.

```python
def test_individual_node_execution():
    """Test a single node without running the full graph."""
    checkpointer = MemorySaver()
    graph = create_order_graph()
    compiled = graph.compile(checkpointer=checkpointer)

    # Invoke just the calculate node directly
    result = compiled.nodes["calculate"].invoke({
        "items": ["burger", "fries", "drink"],
        "total": 0.0,
        "status": "pending"
    })

    assert result["total"] == 17.97  # 3 * 5.99
    assert result["status"] == "calculated"


def test_node_with_empty_state():
    """Test node handles empty input gracefully."""
    graph = create_order_graph()
    compiled = graph.compile(checkpointer=MemorySaver())

    result = compiled.nodes["calculate"].invoke({
        "items": [],
        "total": 0.0,
        "status": "pending"
    })

    assert result["total"] == 0.0
```

## End-to-End Execution Tests

Test complete graph flows with unique thread IDs.

```python
def test_full_order_flow(order_graph):
    """Test complete order processing flow."""
    result = order_graph.invoke(
        {"items": ["Big Mac", "McFlurry"], "total": 0.0, "status": "new"},
        config={"configurable": {"thread_id": "order-001"}}
    )

    assert result["status"] == "calculated"
    assert result["total"] > 0


@pytest.mark.parametrize("items,expected_total", [
    pytest.param(["burger"], 5.99, id="single_item"),
    pytest.param(["burger", "fries"], 11.98, id="two_items"),
    pytest.param([], 0.0, id="empty_order"),
])
def test_order_totals(order_graph, items, expected_total):
    """Parametrized test for various order sizes."""
    result = order_graph.invoke(
        {"items": items, "total": 0.0, "status": "new"},
        config={"configurable": {"thread_id": f"test-{len(items)}"}}
    )
    assert result["total"] == pytest.approx(expected_total)
```

## Partial Execution Tests

Test specific subflows using `update_state()` and `interrupt_after`.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


def create_pipeline_graph() -> StateGraph:
    """Graph with 4 nodes for partial execution demo."""
    class PipelineState(TypedDict):
        value: str

    graph = StateGraph(PipelineState)
    graph.add_node("step1", lambda s: {"value": "step1_done"})
    graph.add_node("step2", lambda s: {"value": "step2_done"})
    graph.add_node("step3", lambda s: {"value": "step3_done"})
    graph.add_node("step4", lambda s: {"value": "step4_done"})
    graph.add_edge(START, "step1")
    graph.add_edge("step1", "step2")
    graph.add_edge("step2", "step3")
    graph.add_edge("step3", "step4")
    graph.add_edge("step4", END)
    return graph


def test_partial_execution_step2_to_step3():
    """Execute only step2 and step3, skipping step1 and step4."""
    checkpointer = MemorySaver()
    graph = create_pipeline_graph()
    compiled = graph.compile(checkpointer=checkpointer)

    # Simulate state as if step1 already completed
    compiled.update_state(
        config={"configurable": {"thread_id": "partial-1"}},
        values={"value": "initial"},
        as_node="step1",  # Pretend step1 produced this state
    )

    # Resume from step2, stop after step3
    result = compiled.invoke(
        None,  # Pass None to resume from saved state
        config={"configurable": {"thread_id": "partial-1"}},
        interrupt_after="step3",  # Stop before step4
    )

    assert result["value"] == "step3_done"
```

## Mocking LLM Responses

Use `GenericFakeChatModel` for deterministic unit tests.

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall


@pytest.fixture
def mock_llm():
    """Create a fake LLM that returns predetermined responses."""
    responses = iter([
        AIMessage(content="I'll help you with that order."),
        AIMessage(content="Your total is $12.99."),
    ])
    return GenericFakeChatModel(messages=responses)


def test_agent_with_mock_llm(mock_llm):
    """Test agent logic with mocked LLM responses."""
    # Use mock_llm in your graph nodes
    response = mock_llm.invoke([HumanMessage(content="Hello")])
    assert "order" in response.content.lower()


def test_llm_tool_calls():
    """Test LLM returning tool calls."""
    responses = iter([
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(name="add_to_cart", args={"item": "Big Mac"}, id="call_1")
            ]
        ),
        AIMessage(content="Added Big Mac to your cart."),
    ])
    mock_llm = GenericFakeChatModel(messages=responses)

    response = mock_llm.invoke([HumanMessage(content="Add a Big Mac")])
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "add_to_cart"
```

## Testing Tool Calls

```python
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode


@tool
def get_menu_item(name: str) -> dict:
    """Look up a menu item by name."""
    menu = {"Big Mac": 5.99, "McFlurry": 3.49}
    return {"name": name, "price": menu.get(name, 0)}


def test_tool_node_execution():
    """Test ToolNode processes tool calls correctly."""
    tool_node = ToolNode([get_menu_item])

    # Simulate state with a tool call from the LLM
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(name="get_menu_item", args={"name": "Big Mac"}, id="call_1")
                ]
            )
        ]
    }

    result = tool_node.invoke(state)

    # Tool results are added as messages
    tool_message = result["messages"][-1]
    assert "5.99" in str(tool_message.content)
```

## Async Graph Testing

With `asyncio_mode = "auto"`, async tests work without decorators.

```python
from langgraph.graph import StateGraph, START, END


async def async_process(state: dict) -> dict:
    """Async node function."""
    # Simulate async operation
    return {"processed": True}


def create_async_graph() -> StateGraph:
    class AsyncState(TypedDict):
        processed: bool

    graph = StateGraph(AsyncState)
    graph.add_node("process", async_process)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    return graph


async def test_async_graph_execution():
    """Test async graph with ainvoke."""
    graph = create_async_graph()
    compiled = graph.compile(checkpointer=MemorySaver())

    result = await compiled.ainvoke(
        {"processed": False},
        config={"configurable": {"thread_id": "async-1"}}
    )

    assert result["processed"] is True
```

## HTTP Recording for Integration Tests

Record real LLM calls for reproducible integration tests.

```python
# conftest.py
import pytest


@pytest.fixture(scope="session")
def vcr_config():
    """Filter sensitive data from recorded cassettes."""
    return {
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("x-api-key", "REDACTED"),
            ("openai-api-key", "REDACTED"),
        ],
        "filter_query_parameters": [
            ("api_key", "REDACTED"),
        ],
        "record_mode": "once",  # Record on first run, replay after
    }


# test_integration.py
import pytest


@pytest.mark.vcr()
@pytest.mark.integration
def test_real_llm_integration(real_agent):
    """Integration test with recorded HTTP responses."""
    result = real_agent.invoke(
        {"messages": [HumanMessage(content="What's on the menu?")]},
        config={"configurable": {"thread_id": "integration-1"}}
    )

    assert result["messages"][-1].content  # LLM responded
```

## Testing Interrupts and Human-in-the-Loop

```python
from langgraph.types import interrupt, Command


def approval_node(state: dict) -> dict:
    """Node that requires human approval."""
    approved = interrupt("Approve this order?")
    return {"approved": approved}


def test_interrupt_and_resume():
    """Test interrupt pauses graph and resume continues."""
    # ... setup graph with approval_node ...
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "approval-1"}}

    # First invoke hits interrupt
    result = compiled.invoke({"order": "Big Mac"}, config)
    assert "__interrupt__" in result

    # Resume with approval
    final = compiled.invoke(
        Command(resume=True),  # Human approved
        config
    )
    assert final["approved"] is True
```

## State Assertions

```python
def test_state_transitions(order_graph):
    """Verify state changes through graph execution."""
    initial_state = {
        "items": ["burger"],
        "total": 0.0,
        "status": "new"
    }

    result = order_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": "state-1"}}
    )

    # Assert state transitions
    assert result["status"] != initial_state["status"]
    assert result["total"] > initial_state["total"]
    assert result["items"] == initial_state["items"]  # Unchanged


def test_state_history(order_graph):
    """Inspect checkpointed state history."""
    config = {"configurable": {"thread_id": "history-1"}}

    order_graph.invoke(
        {"items": ["burger"], "total": 0.0, "status": "new"},
        config
    )

    # Get all checkpoints
    history = list(order_graph.get_state_history(config))
    assert len(history) >= 2  # At least start and end states
```

## Fixtures Reference

```python
# conftest.py
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


@pytest.fixture
def checkpointer():
    """Fresh in-memory checkpointer."""
    return MemorySaver()


@pytest.fixture
def thread_id(request):
    """Unique thread ID based on test name."""
    return f"test-{request.node.name}"


@pytest.fixture
def mock_llm_responses():
    """Override to provide custom mock responses."""
    return iter([AIMessage(content="Default response")])


@pytest.fixture
def mock_llm(mock_llm_responses):
    """Fake LLM for unit tests."""
    return GenericFakeChatModel(messages=mock_llm_responses)
```
