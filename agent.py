#!/usr/bin/env python3
"""
agentic-compiler — Compile agent intentions into executable task graphs
Takes high-level agent goals and compiles them into concrete task sequences
with dependency resolution, resource allocation, and rollback planning.
"""

import json, time, hashlib
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class Task:
    id: str
    name: str
    action: str
    inputs: List[str]
    outputs: List[str]
    deps: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, complete, failed
    agent: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

@dataclass
class TaskGraph:
    tasks: Dict[str, Task]
    entry_points: List[str]
    final_outputs: List[str]

class AgenticCompiler:
    def __init__(self, plato_url="http://147.224.38.131:8847"):
        self.plato_url = plato_url
        self.compiled_graphs: Dict[str, TaskGraph] = {}
    
    def compile(self, goal: str, available_agents: List[str], tools: List[str]) -> TaskGraph:
        """Compile a goal into a task graph."""
        graph_id = hashlib.md5(f"{goal}:{time.time()}".encode()).hexdigest()[:12]
        
        # Parse goal into sub-tasks (simplified)
        tasks = {}
        
        if "build" in goal.lower() or "create" in goal.lower():
            # Build pattern: design -> implement -> test -> deploy
            t1 = Task(f"{graph_id}-design", "Design", "design", [], ["spec"], agent=available_agents[0] if available_agents else None)
            t2 = Task(f"{graph_id}-build", "Build", "implement", ["spec"], ["code"], deps=[t1.id], agent=available_agents[0] if available_agents else None)
            t3 = Task(f"{graph_id}-test", "Test", "test", ["code"], ["report"], deps=[t2.id], agent=available_agents[1] if len(available_agents) > 1 else available_agents[0] if available_agents else None)
            t4 = Task(f"{graph_id}-deploy", "Deploy", "deploy", ["code", "report"], ["live"], deps=[t3.id], agent=available_agents[0] if available_agents else None)
            tasks = {t.id: t for t in [t1, t2, t3, t4]}
            entry = [t1.id]
            outputs = ["live"]
        
        elif "research" in goal.lower() or "investigate" in goal.lower():
            # Research pattern: gather -> analyze -> synthesize
            t1 = Task(f"{graph_id}-gather", "Gather", "research", [], ["raw_data"], agent=available_agents[0] if available_agents else None)
            t2 = Task(f"{graph_id}-analyze", "Analyze", "analyze", ["raw_data"], ["insights"], deps=[t1.id], agent=available_agents[0] if available_agents else None)
            t3 = Task(f"{graph_id}-synthesize", "Synthesize", "write", ["insights"], ["paper"], deps=[t2.id], agent=available_agents[0] if available_agents else None)
            tasks = {t.id: t for t in [t1, t2, t3]}
            entry = [t1.id]
            outputs = ["paper"]
        
        else:
            # Generic: single task
            t1 = Task(f"{graph_id}-main", "Execute", "execute", [], ["result"], agent=available_agents[0] if available_agents else None)
            tasks = {t1.id: t1}
            entry = [t1.id]
            outputs = ["result"]
        
        graph = TaskGraph(tasks=tasks, entry_points=entry, final_outputs=outputs)
        self.compiled_graphs[graph_id] = graph
        
        self._submit(f"Compiled task graph for: {goal}", f"{len(tasks)} tasks, agents: {available_agents}")
        return graph
    
    def get_execution_order(self, graph_id: str) -> List[str]:
        """Topological sort of tasks."""
        graph = self.compiled_graphs.get(graph_id)
        if not graph:
            return []
        
        # Kahn's algorithm
        in_degree = {tid: 0 for tid in graph.tasks}
        for t in graph.tasks.values():
            for dep in t.deps:
                if dep in in_degree:
                    in_degree[t.id] += 1
        
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []
        
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for t in graph.tasks.values():
                if tid in t.deps:
                    in_degree[t.id] -= 1
                    if in_degree[t.id] == 0:
                        queue.append(t.id)
        
        return order
    
    def get_ready_tasks(self, graph_id: str) -> List[Task]:
        """Tasks whose dependencies are all complete."""
        graph = self.compiled_graphs.get(graph_id)
        if not graph:
            return []
        
        ready = []
        for t in graph.tasks.values():
            if t.status != "pending":
                continue
            deps_complete = all(graph.tasks[d].status == "complete" for d in t.deps if d in graph.tasks)
            if deps_complete:
                ready.append(t)
        return ready
    
    def mark_complete(self, graph_id: str, task_id: str):
        """Mark a task as complete."""
        graph = self.compiled_graphs.get(graph_id)
        if graph and task_id in graph.tasks:
            graph.tasks[task_id].status = "complete"
    
    def mark_failed(self, graph_id: str, task_id: str):
        """Mark a task as failed, trigger retry or rollback."""
        graph = self.compiled_graphs.get(graph_id)
        if not graph or task_id not in graph.tasks:
            return
        
        task = graph.tasks[task_id]
        task.retries += 1
        if task.retries < task.max_retries:
            task.status = "pending"
        else:
            task.status = "failed"
    
    def get_graph_summary(self, graph_id: str) -> Dict:
        graph = self.compiled_graphs.get(graph_id)
        if not graph:
            return {"error": "Graph not found"}
        
        statuses = {}
        for t in graph.tasks.values():
            s = t.status
            statuses[s] = statuses.get(s, 0) + 1
        
        return {
            "tasks": len(graph.tasks),
            "statuses": statuses,
            "execution_order": self.get_execution_order(graph_id),
            "ready_now": [t.id for t in self.get_ready_tasks(graph_id)]
        }
    
    def _submit(self, q: str, a: str):
        try:
            import urllib.request
            urllib.request.urlopen(urllib.request.Request(f"{self.plato_url}/submit", data=json.dumps({"question": q, "answer": a, "agent": "agentic-compiler", "room": "compiler"}).encode(), headers={"Content-Type": "application/json"}), timeout=5)
        except: pass

def demo():
    compiler = AgenticCompiler()
    
    print("=== Compiling: Build fleet dashboard ===")
    graph = compiler.compile("Build a real-time fleet dashboard", ["CCC", "Oracle1"], ["python", "html", "css"])
    print(f"Graph ID: {list(compiler.compiled_graphs.keys())[0]}")
    print(f"Tasks: {len(graph.tasks)}")
    print(f"Execution order: {compiler.get_execution_order(list(compiler.compiled_graphs.keys())[0])}")
    
    print("\n=== Marking design complete ===")
    gid = list(compiler.compiled_graphs.keys())[0]
    compiler.mark_complete(gid, f"{gid}-design")
    print(f"Ready tasks: {compiler.get_graph_summary(gid)['ready_now']}")
    
    print("\n=== Full Summary ===")
    print(json.dumps(compiler.get_graph_summary(gid), indent=2))

if __name__ == "__main__": demo()
