# Completed Lab Source

This `src/` folder contains the completed Day 17 memory-system lab.

- It keeps the same high-level structure as the scaffold
- The Python files implement deterministic offline behavior for tests and benchmark runs
- The benchmark structure should include: standard benchmark + long-context stress benchmark
- The runtime should support these providers: `openai`, `custom`, `gemini`, `anthropic`, `ollama`, `openrouter`

Suggested flow:

1. Start with `config.py`
2. Implement `memory_store.py`
3. Finish `agent_baseline.py`
4. Finish `agent_advanced.py`
5. Implement `benchmark.py`
6. Make `test_agents.py` pass

Datasets are available at the repo root in `data/`.
