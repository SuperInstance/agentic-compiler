# agentic-compiler

Markdown-to-runtime agentic compilation. The compiler doesn't just transform code — it deliberates, experiments, and evolves the codebase in real-time through swarm consensus. Part of the Cocapn fleet's compilation pipeline.

## Brand Line

> Agentic compilation as fleet deliberation — markdown in, optimal runtime out.

## Installation

```bash
pip install cocapn-agentic-compiler
```

## Usage

```python
from agentic_compiler import Compiler

# Compile markdown specification to runtime
compiler = Compiler()
result = compiler.compile("spec.md")

# Swarm deliberation for optimal output
best_output = result.deliberate(rounds=10)
```

## Fleet Context

Part of the Cocapn fleet. Related repos:
- [cudaclaw](https://github.com/SuperInstance/cudaclaw) — GPU-accelerated agent orchestration
- [bordercollie](https://github.com/SuperInstance/bordercollie) — Fleet task herding and orchestration
- [ai-character-sdk](https://github.com/SuperInstance/ai-character-sdk) — Unified AI character SDK with memory
- [crab-traps](https://github.com/SuperInstance/crab-traps) — Lure collection for fleet learning

---
🦐 Cocapn fleet — lighthouse keeper architecture