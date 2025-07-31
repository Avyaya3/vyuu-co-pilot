# LangGraph Studio Setup Guide

This guide will help you set up LangGraph Studio for the Vyuu Copilot v2 project, enabling you to visualize, test, and debug your intent orchestration graph.

## Prerequisites

- Python 3.9 or higher
- All project dependencies installed
- Environment variables configured

## Quick Setup

### 1. Install LangGraph CLI

```bash
pip install langgraph[cli]
```

### 2. Verify Configuration

Run the setup script to check your configuration:

```bash
python setup_studio.py
```

This will verify:
- LangGraph CLI installation
- `langraph.json` configuration
- Required dependencies
- Environment variables

### 3. Verify Test User

Verify that your test user exists and has data:

```bash
python scripts/verify_test_user.py
```

This will check:
- Test user exists in database
- User has accounts and transactions
- User has goals and other data for testing

### 4. Set Environment Variables

Copy the example environment file and fill in your values:

```bash
cp env.example .env
```

Edit `.env` with your actual values:
- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_ANON_KEY`: Your Supabase anonymous key
- `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key
- `DATABASE_URL`: Your direct PostgreSQL connection URL
- `STUDIO_TEST_USER_ID`: UUID of a test user from your database (e.g., 0575867a-743a-4f26-99b3-95b87d116d7b)

### 5. Start LangGraph Studio

```bash
langgraph dev
```

This will:
- Start a local development server
- Open LangGraph Studio in your browser
- Enable live reloading of code changes

## Configuration Details

### langraph.json

The `langraph.json` file configures how LangGraph Studio connects to your graph:

```json
{
  "name": "vyuu-copilot-v2",
  "description": "LangGraph-based intent orchestration system for financial assistance",
  "graph": {
    "module": "src.graphs.main_orchestrator_graph",
    "function": "main_orchestrator_graph"
  },
  "environment": {
    "OPENAI_API_KEY": {
      "description": "OpenAI API key for LLM operations",
      "required": true
    }
    // ... other environment variables
  }
}
```

### Main Entry Point

The `main.py` file provides the `invoke_graph` function that LangGraph Studio uses to execute your graph:

```python
async def invoke_graph(
    user_input: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    conversation_history: Optional[list] = None,
    **kwargs
) -> Dict[str, Any]:
    # Graph execution logic
```

## Using LangGraph Studio

### Graph Visualization

Once Studio opens, you'll see:
- **Graph View**: Visual representation of your intent orchestration flow
- **Node Details**: Information about each node and its connections
- **Execution Flow**: Real-time visualization of graph execution

### Testing Your Graph

1. **Single Invocation**: Test individual graph executions
2. **Multi-turn Conversations**: Use the chat interface for conversation testing
3. **Conversation Forking**: Create branches to test different scenarios
4. **Interrupt Testing**: Test human-in-the-loop scenarios

### Debugging Features

- **Step-by-step Execution**: Walk through each node execution
- **State Inspection**: View state at each step
- **Error Tracking**: Identify and debug issues
- **Performance Monitoring**: Track execution times

### Live Development

- **Auto-reload**: Changes to your code are instantly reflected
- **Prompt Iteration**: Test and refine prompts in real-time
- **Model Switching**: Experiment with different models
- **Configuration Testing**: Test different parameter combinations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -e .
   ```

2. **Environment Variables**: Check that all required variables are set
   ```bash
   python setup_studio.py
   ```

3. **Graph Compilation Errors**: Check your graph configuration
   ```bash
   python -c "from src.graphs.main_orchestrator_graph import main_orchestrator_graph; print('Graph compiled successfully')"
   ```

4. **Port Conflicts**: If port 8123 is in use, Studio will use the next available port

### Debug Mode

Enable debug logging for more detailed information:

```bash
export LOG_LEVEL=DEBUG
langgraph dev
```

## Advanced Features

### LangSmith Integration

For production debugging, integrate with LangSmith:

1. Set up LangSmith environment variables
2. Filter production traces in LangSmith
3. Use "Run in Studio" feature to debug production issues locally

### Custom Tools

Add custom tools to your graph for enhanced functionality:

1. Define tools in `src/tools/`
2. Register tools in your graph
3. Test tool execution in Studio

### Performance Optimization

Use Studio to identify and optimize performance bottlenecks:

1. Monitor execution times
2. Identify slow nodes
3. Optimize prompts and configurations

## Next Steps

After setting up LangGraph Studio:

1. **Test Basic Flows**: Verify intent classification and routing
2. **Test Clarification Subgraph**: Test human-in-the-loop scenarios
3. **Test Direct Orchestrator**: Test complete execution flows
4. **Optimize Prompts**: Refine prompts based on test results
5. **Add Monitoring**: Set up production monitoring with LangSmith

## Resources

- [LangGraph Studio Documentation](https://docs.langchain.com/langgraph/studio)
- [LangGraph CLI Reference](https://docs.langchain.com/langgraph/cli)
- [LangSmith Integration](https://docs.langchain.com/langsmith)
- [Project Documentation](./README.md) 