# Vyuu Copilot v2 - LangGraph Intent Orchestration System

A sophisticated intent orchestration system built with LangGraph, designed to handle complex conversational workflows and intent management.

## 🏗️ Project Structure

```
vyuu-copilot-v2/
├── src/
│   ├── subgraphs/          # LangGraph subgraph implementations
│   ├── tools/              # Custom tools and integrations
│   ├── nodes/              # Individual workflow nodes
│   ├── schemas/            # Pydantic schemas and data models
│   └── utils/              # Utility functions and helpers
├── tests/                  # Test suite
├── docs/                   # Documentation
├── config/                 # Configuration files
├── ui/                     # User interface components
├── prompts/                # Prompt templates and management
├── main.py                 # Application entry point
├── pyproject.toml          # Project dependencies and configuration
└── .env.template           # Environment variables template
```

## 🚀 Features

- **LangGraph Integration**: Advanced workflow orchestration using LangGraph
- **Intent Recognition**: Sophisticated intent classification and routing
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Supabase Integration**: Database and authentication backend
- **FastAPI Backend**: High-performance API server
- **Type Safety**: Full type annotations with Pydantic schemas
- **Testing Suite**: Comprehensive test coverage with pytest

## 📋 Prerequisites

- Python 3.9 or higher
- Node.js 16 or higher (for UI components)
- Supabase account (for database and authentication)
- OpenAI API key (or other LLM provider)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vyuu/vyuu-copilot-v2.git
   cd vyuu-copilot-v2
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**
   ```bash
   cp .env.template .env
   # Edit .env with your actual configuration values
   ```

5. **Initialize the database**
   ```bash
   # Instructions for database setup will be added here
   ```

## ⚙️ Configuration

Copy `.env.template` to `.env` and configure the following variables:

### Supabase Configuration
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase anon key
- `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key

### API Configuration
- `API_HOST`: API server host (default: localhost)
- `API_PORT`: API server port (default: 8000)
- `API_ENVIRONMENT`: Environment (development/staging/production)

### LangChain Configuration
- `LANGCHAIN_API_KEY`: LangChain API key for tracing
- `LANGCHAIN_TRACING_V2`: Enable LangChain tracing
- `LANGCHAIN_PROJECT`: Project name for LangChain

### Model Provider Configuration
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI models)

## 🏃‍♂️ Running the Application

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=src
```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Architecture Overview](docs/architecture.md)
- [Development Guide](docs/development.md)
- [Deployment Guide](docs/deployment.md)

## 🔧 Development

### Code Style
This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
black src tests
isort src tests
flake8 src tests
mypy src
```

### Pre-commit Hooks
Install pre-commit hooks:
```bash
pre-commit install
```

## 📁 Module Overview

### `src/subgraphs/`
Contains LangGraph subgraph implementations for different workflow patterns.

### `src/tools/`
Custom tools and integrations for external services and APIs.

### `src/nodes/`
Individual workflow nodes that can be composed into larger graphs.

### `src/schemas/`
Pydantic schemas and data models for type safety and validation.

### `src/utils/`
Utility functions and helper modules used throughout the application.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [documentation](docs/)
2. Search existing [GitHub Issues](https://github.com/vyuu/vyuu-copilot-v2/issues)
3. Create a new issue if needed

## 🗺️ Roadmap

- [ ] Core LangGraph workflow implementation
- [ ] Intent classification system
- [ ] Supabase integration
- [ ] FastAPI backend setup
- [ ] User interface development
- [ ] Testing and validation
- [ ] Documentation completion
- [ ] Deployment configuration

---

Built with ❤️ by the Vyuu Team 