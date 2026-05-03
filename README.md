**Topics:** `agentic-compilation` `markdown-spec` `swarm-deliberation` `refinement-amplifier` `code-generation` `collective-intelligence` `cocapn`

---

# Agentic Compiler

> Agentic compilation as fleet deliberation — markdown in, optimal runtime out.

**Agentic Compiler** compiles natural language markdown specifications into executable runtime code through **swarm deliberation** — multiple agents deliberating in rounds, refining the output until consensus emerges. Part of the Cocapn fleet's compilation pipeline.

Part of the [Cocapn fleet](https://github.com/SuperInstance) — lighthouse keeper architecture.

---

## What It Does

Traditional compilers read code and emit executables. The Agentic Compiler reads **markdown specs** and emits **runtime code** through a multi-agent deliberative process:

1. **Markdown input** — A human (or agent) writes what they want in plain language
2. **RA rounds** — Refinement Amplifier runs N rounds of agent deliberation
3. **Swarm consensus** — Multiple agents evaluate, critique, and improve the output
4. **Optimal output** — The best result emerges from collective intelligence

### Key Features

- **Markdown-first** — Write specs in natural language, not code
- **Swarm deliberation** — Multiple agents, multiple rounds, collective refinement
- **RA (Refinement Amplifier)** — Each round amplifies what's good, narrows what's wrong
- **Plurality voting** — Agents vote on best output, reducing individual bias

---

## Quick Start

### Install

```bash
pip install cocapn-agentic-compiler
```

### Basic Usage

```python
from agentic_compiler import Compiler

# Initialize the compiler
compiler = Compiler()

# Compile a markdown specification
result = compiler.compile("spec.md")

# Get the deliberated output
best_output = result.deliberate(rounds=10)

print(best_output["code"])
print(f"Consensus score: {best_output['score']}")
print(f"Rounds: {best_output['rounds']}")
```

### Run the RA Pipeline Manually

```bash
python scripts/run-ra.py --input spec.md --rounds 10 --output out/
```

### Programmatic RA

```python
from agentic_compiler import RefinementAmplifier

ra = RefinementAmplifier(n_agents=5, rounds=10)
output = ra.deliberate(
    initial_spec="A REST API for a fleet registry with CRUD operations",
    domain="api"
)

print(output.best_candidate)
print(f"Winners: {output.vote_tally()}")
```

---

## Architecture

```
agentic-compiler/
├── README.md
├── LICENSE
├── docs/
│   └── ra/
│       ├── round-1.md       # Round 1 deliberation notes
│       ├── round-2.md
│       ├── ...
│       └── round-10.md      # Final round
├── scripts/
│   └── run-ra.py           # CLI entry point for RA pipeline
└── tests/
    └── test_agentic_compiler_docs.py
```

### RA Round Structure

Each `round-N.md` captures the deliberation at that stage:

```
Round 1: Initial spec → 3 agents propose candidates
Round 2: Candidates → cross-evaluation, identify weaknesses
Round 3: Refinement → agents revise based on feedback
...
Round N: Final vote → plurality vote, consensus score
```

### Refinement Amplifier Pipeline

```
Markdown Spec
    │
    ▼
Round 1: 3 agents propose candidate implementations
    │
    ▼
Each Round: Agents evaluate ALL candidates
            + identify weaknesses
            + vote on best
            + revise their own candidate
    │
    ▼  (repeat N times)
Round N: Final vote
    │
    ▼
Output: Best candidate + consensus score
```

---

## Example: Full RA Deliberation

```python
from agentic_compiler import RefinementAmplifier, Compiler

compiler = Compiler()

# Example spec: fleet routing algorithm
spec = """
# Fleet Routing Algorithm

## Input
- List of agent positions (x, y)
- List of target waypoints
- Max hops per agent

## Output
- Optimal routing table
- Estimated completion time

## Constraints
- No agent collision
- Respect max-hop limit
"""

result = compiler.compile_str(spec)
best = result.deliberate(rounds=10)

print(f"=== Best Output (score: {best['score']:.2f}) ===")
print(best['code'])
print()
print(f"Voting breakdown: {best['votes']}")
```

### Sample Round Output (from docs/ra/round-3.md)

```
## Round 3 Deliberation

### Agent A (proponent-dijkstra)
Proposed: Dijkstra's algorithm for routing
Votes received: 2
Concerns: O(n²) complexity may not scale to 10k agents

### Agent B (proponent-a*)
Proposed: A* with spatial heuristic
Votes received: 3
Concerns: Memory overhead for large grids

### Agent C (proponent-rr)
Proposed: Round-robin with collision avoidance
Votes received: 1
Concerns: Not optimal, but simple

### Cross-evaluation
- All agents agree: collision avoidance is mandatory
- Agents A and B agree: optimal path matters for fuel efficiency
- Agent C's simplicity wins on implementation cost

### Round 3 Decision
Winner: Agent B (A* with spatial heuristic)
Consensus score: 0.73
```

---

## Fleet Context

Part of the Cocapn fleet. Related repos:

| Repo | Role |
|------|-------|
| [cudaclaw](https://github.com/SuperInstance/cudaclaw) | GPU-accelerated agent orchestration |
| [bordercollie](https://github.com/SuperInstance/bordercollie) | Fleet task herding and orchestration |
| [ai-character-sdk](https://github.com/SuperInstance/ai-character-sdk) | Unified AI character SDK with memory |
| [crab-traps](https://github.com/SuperInstance/crab-traps) | Lure collection for fleet learning |

---
🦐 Cocapn fleet — lighthouse keeper architecture
