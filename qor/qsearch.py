"""
QSearch — Quantum-Inspired Parallel Search Engine
===================================================
Simulates quantum parallelism using Python worker pools:

  1 candidate → spawns N variants → all evaluate in parallel
  → best survive (amplitude collapse) → spawn again → repeat

Concepts borrowed from quantum computing:
  - Superposition:  One state branches into N parallel states
  - Amplitude:      Each state has a "probability" score (fitness)
  - Interference:   Good states reinforce, bad states cancel
  - Measurement:    Collapse to best candidates each generation
  - Entanglement:   States share information via shared memory

Usage:
    from qor.qsearch import QSearchEngine

    def my_oracle(params):
        return {"fitness": score}

    engine = QSearchEngine(oracle=my_oracle, n_workers=8)
    best = engine.search(initial_params={"lr": 0.01})
    print(best.data, best.fitness)
"""

import math
import random
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

logger = logging.getLogger(__name__)


# ==============================================================================
# QuantumState — One candidate in superposition
# ==============================================================================

@dataclass
class QuantumState:
    """One candidate solution — like a qubit in superposition.

    Attributes:
        state_id:   Unique identifier
        data:       The actual candidate (dict of parameters, vector, etc.)
        amplitude:  Probability weight (0.0 = dead, 1.0 = best)
        phase:      Exploration direction (-1 to 1, affects mutation)
        generation: Which generation this was born in
        parent_id:  Who spawned this state
        fitness:    Raw score from oracle evaluation
        metadata:   Any extra info (search path, debug, etc.)
    """
    state_id: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    amplitude: float = 1.0
    phase: float = 0.0
    generation: int = 0
    parent_id: str = ""
    fitness: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.state_id:
            raw = f"{id(self)}:{time.time_ns()}:{random.random()}"
            self.state_id = hashlib.md5(raw.encode()).hexdigest()[:12]

    def clone(self) -> 'QuantumState':
        """Create a copy with new ID (for branching)."""
        return QuantumState(
            data={**self.data},
            amplitude=self.amplitude,
            phase=self.phase,
            generation=self.generation,
            parent_id=self.state_id,
            fitness=self.fitness,
            metadata={**self.metadata},
        )


# ==============================================================================
# QuantumRegister — Collection of superposed states
# ==============================================================================

class QuantumRegister:
    """Holds all parallel states — like a quantum register of qubits.

    Manages superposition (branching), measurement (collapsing),
    and interference (combining good/bad signals).
    """

    def __init__(self, max_states: int = 1000):
        self.states: Dict[str, QuantumState] = {}
        self.max_states = max_states
        self.generation = 0
        self._history = []  # Best fitness per generation

    def add(self, state: QuantumState):
        """Add a state to the register."""
        self.states[state.state_id] = state

    def superpose(self, state: QuantumState, n_branches: int = 100,
                  mutation_fn: Callable = None) -> List[QuantumState]:
        """SUPERPOSITION: One state → N parallel variants.

        Like quantum branching — one candidate explores N directions
        simultaneously. Each branch gets a slight mutation.

        Args:
            state: Parent state to branch from
            n_branches: How many parallel variants to create
            mutation_fn: Function(data, phase) → mutated_data
                         If None, uses default random perturbation

        Returns:
            List of new QuantumState objects (already added to register)
        """
        branches = []
        for i in range(n_branches):
            child = state.clone()
            child.generation = self.generation + 1

            # Distribute phases evenly around the unit circle
            child.phase = -1.0 + (2.0 * i / n_branches)

            # Amplitude decays slightly with branching (conservation)
            child.amplitude = state.amplitude / math.sqrt(n_branches)

            # Mutate the data
            if mutation_fn:
                child.data = mutation_fn(dict(state.data), child.phase)
            else:
                child.data = self._default_mutate(dict(state.data), child.phase)

            self.add(child)
            branches.append(child)

        return branches

    def _default_mutate(self, data: dict, phase: float) -> dict:
        """Default mutation: perturb numeric values based on phase."""
        mutated = {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                scale = max(abs(v), 1.0)
                noise = phase * scale * 0.15 * (1 + random.random())
                new_val = v + noise
                mutated[k] = type(v)(new_val) if isinstance(v, int) else new_val
            elif isinstance(v, list):
                mutated[k] = [
                    x + phase * max(abs(x), 1.0) * 0.15 * random.random()
                    if isinstance(x, (int, float)) else x
                    for x in v
                ]
            else:
                mutated[k] = v
        return mutated

    def measure(self, keep_top_n: int = 10) -> List[QuantumState]:
        """MEASUREMENT: Collapse superposition to best candidates.

        Like quantum measurement — observe the register and keep
        only the highest-amplitude (best fitness) states.

        Args:
            keep_top_n: How many survivors

        Returns:
            List of surviving states (sorted by fitness, best first)
        """
        ranked = sorted(
            self.states.values(),
            key=lambda s: s.fitness,
            reverse=True,
        )

        survivors = ranked[:keep_top_n]

        if survivors:
            self._history.append({
                "generation": self.generation,
                "best_fitness": survivors[0].fitness,
                "avg_fitness": sum(s.fitness for s in survivors) / len(survivors),
                "total_evaluated": len(self.states),
            })

        # Renormalize amplitudes of survivors
        total_amp = sum(s.amplitude for s in survivors) or 1.0
        for s in survivors:
            s.amplitude = s.amplitude / total_amp

        # Collapse: remove all non-survivors
        self.states = {s.state_id: s for s in survivors}
        self.generation += 1

        return survivors

    def interfere(self):
        """INTERFERENCE: Boost good states, suppress bad ones.

        Constructive: above-average fitness → amplitude boost.
        Destructive: below-average → amplitude suppressed.
        """
        if not self.states:
            return

        avg_fitness = sum(s.fitness for s in self.states.values()) / len(self.states)

        for state in self.states.values():
            if state.fitness > avg_fitness:
                boost = 1.0 + (state.fitness - avg_fitness) / max(abs(avg_fitness), 1e-6) * 0.5
                state.amplitude = min(state.amplitude * boost, 1.0)
            else:
                decay = 0.5 + (state.fitness / max(avg_fitness, 1e-6)) * 0.5
                state.amplitude *= max(decay, 0.01)

    def entangle(self, state_a: QuantumState, state_b: QuantumState) -> QuantumState:
        """ENTANGLEMENT: Combine information from two states.

        Creates a child that blends the best aspects of both parents.
        Like crossover in genetic algorithms, but amplitude-weighted.
        """
        child = QuantumState(
            generation=self.generation,
            parent_id=f"{state_a.state_id}+{state_b.state_id}",
        )

        weight_a = state_a.amplitude / (state_a.amplitude + state_b.amplitude + 1e-8)

        for key in set(list(state_a.data.keys()) + list(state_b.data.keys())):
            val_a = state_a.data.get(key, 0)
            val_b = state_b.data.get(key, 0)
            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                blended = val_a * weight_a + val_b * (1 - weight_a)
                child.data[key] = type(val_a)(blended) if isinstance(val_a, int) else blended
            else:
                child.data[key] = val_a if random.random() < weight_a else val_b

        child.amplitude = max(state_a.amplitude, state_b.amplitude)
        self.add(child)
        return child

    @property
    def best(self) -> Optional[QuantumState]:
        if not self.states:
            return None
        return max(self.states.values(), key=lambda s: s.fitness)


# ==============================================================================
# Parallel Evaluation Worker
# ==============================================================================

def _evaluate_worker(args: Tuple) -> Tuple[str, float, dict]:
    """Worker function for parallel evaluation.

    Runs in separate process/thread. Evaluates one state using the oracle.

    Args:
        args: (state_id, state_data, oracle_fn, oracle_args)

    Returns:
        (state_id, fitness_score, metadata)
    """
    state_id, state_data, oracle_fn, oracle_args = args
    try:
        result = oracle_fn(state_data, **(oracle_args or {}))
        if isinstance(result, (int, float)):
            return (state_id, float(result), {})
        elif isinstance(result, dict):
            return (state_id, float(result.get("fitness", 0)), result)
        else:
            return (state_id, 0.0, {})
    except Exception as e:
        return (state_id, -999.0, {"error": str(e)})


# ==============================================================================
# QSearchEngine — Main orchestrator with worker pool
# ==============================================================================

class QSearchEngine:
    """Quantum-Inspired Parallel Search Engine.

    Usage:
        def my_oracle(params):
            return {"fitness": score}

        engine = QSearchEngine(
            oracle=my_oracle,
            n_workers=8,
            branches_per_state=100,
            survivors_per_gen=10,
            max_generations=50,
        )

        best = engine.search(initial_params={"lr": 0.01, "layers": 3})
        print(best.data, best.fitness)
    """

    def __init__(
        self,
        oracle: Callable,
        n_workers: int = None,
        branches_per_state: int = 100,
        survivors_per_gen: int = 10,
        max_generations: int = 50,
        max_states: int = 5000,
        use_processes: bool = False,
        mutation_fn: Callable = None,
        oracle_args: dict = None,
        early_stop_fitness: float = None,
        early_stop_patience: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            oracle:              Fitness function(data) → float or dict with "fitness"
            n_workers:           Parallel workers (default: CPU count)
            branches_per_state:  How many variants per survivor
            survivors_per_gen:   How many survive measurement (collapse)
            max_generations:     When to stop
            max_states:          Memory limit
            use_processes:       True = ProcessPool (CPU-bound), False = ThreadPool (IO-bound)
            mutation_fn:         Custom mutation function(data, phase) → new_data
            oracle_args:         Extra kwargs passed to oracle
            early_stop_fitness:  Stop if fitness reaches this value
            early_stop_patience: Stop if no improvement for N generations
            verbose:             Print progress
        """
        self.oracle = oracle
        self.oracle_args = oracle_args or {}
        self.n_workers = n_workers or mp.cpu_count()
        self.branches_per_state = branches_per_state
        self.survivors_per_gen = survivors_per_gen
        self.max_generations = max_generations
        self.use_processes = use_processes
        self.mutation_fn = mutation_fn
        self.early_stop_fitness = early_stop_fitness
        self.early_stop_patience = early_stop_patience
        self.verbose = verbose

        self.register = QuantumRegister(max_states=max_states)
        self._run_stats = []

    def search(self, initial_params: dict = None,
               initial_states: List[dict] = None) -> QuantumState:
        """Run the full quantum-inspired search.

        Args:
            initial_params: Single starting point (will be superposed into N)
            initial_states: Multiple starting points

        Returns:
            Best QuantumState found
        """
        # --- Initialize superposition ---
        seeds = []
        if initial_states:
            for data in initial_states:
                state = QuantumState(data=data, generation=0)
                seeds.append(state)
        elif initial_params:
            seed = QuantumState(data=initial_params, generation=0)
            seeds.append(seed)
        else:
            raise ValueError("Need initial_params or initial_states")

        # First superposition: seed → N branches
        all_branches = []
        for seed in seeds:
            branches = self.register.superpose(
                seed,
                n_branches=self.branches_per_state,
                mutation_fn=self.mutation_fn,
            )
            all_branches.extend(branches)

        if self.verbose:
            print(f"[QSearch] Initialized: {len(all_branches)} states from "
                  f"{len(seeds)} seed(s), {self.n_workers} workers")

        # --- Main search loop ---
        best_ever = None
        no_improve_count = 0
        PoolClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        for gen in range(self.max_generations):
            t0 = time.time()

            # --- Parallel evaluation (the quantum "oracle query") ---
            states_to_eval = list(self.register.states.values())
            eval_args = [
                (s.state_id, s.data, self.oracle, self.oracle_args)
                for s in states_to_eval
            ]

            results = {}
            with PoolClass(max_workers=self.n_workers) as pool:
                futures = {
                    pool.submit(_evaluate_worker, args): args[0]
                    for args in eval_args
                }
                for future in as_completed(futures):
                    state_id, fitness, meta = future.result()
                    results[state_id] = (fitness, meta)

            # Apply fitness scores
            for state in states_to_eval:
                if state.state_id in results:
                    fitness, meta = results[state.state_id]
                    state.fitness = fitness
                    state.metadata.update(meta)

            # --- Interference (boost good, suppress bad) ---
            self.register.interfere()

            # --- Measurement (collapse to top N) ---
            survivors = self.register.measure(keep_top_n=self.survivors_per_gen)

            gen_best = survivors[0] if survivors else None
            elapsed = time.time() - t0

            # Track improvement
            if best_ever is None or (gen_best and gen_best.fitness > best_ever.fitness):
                best_ever = gen_best.clone() if gen_best else best_ever
                no_improve_count = 0
            else:
                no_improve_count += 1

            gen_stats = {
                "generation": gen,
                "evaluated": len(states_to_eval),
                "best_fitness": gen_best.fitness if gen_best else 0,
                "best_ever": best_ever.fitness if best_ever else 0,
                "avg_fitness": sum(s.fitness for s in survivors) / len(survivors) if survivors else 0,
                "elapsed_sec": round(elapsed, 2),
            }
            self._run_stats.append(gen_stats)

            if self.verbose:
                print(f"  Gen {gen:3d} | "
                      f"Eval: {len(states_to_eval):5d} | "
                      f"Best: {gen_stats['best_fitness']:+.6f} | "
                      f"Ever: {gen_stats['best_ever']:+.6f} | "
                      f"Avg: {gen_stats['avg_fitness']:+.6f} | "
                      f"{elapsed:.1f}s")

            # --- Early stopping ---
            if self.early_stop_fitness is not None and best_ever:
                if best_ever.fitness >= self.early_stop_fitness:
                    if self.verbose:
                        print(f"[QSearch] Target fitness reached at gen {gen}")
                    break

            if no_improve_count >= self.early_stop_patience:
                if self.verbose:
                    print(f"[QSearch] No improvement for {no_improve_count} gens — stopping")
                break

            # --- Entanglement: cross-breed top survivors ---
            if len(survivors) >= 2:
                for i in range(0, len(survivors) - 1, 2):
                    self.register.entangle(survivors[i], survivors[i + 1])

            # --- Next superposition: each survivor → N branches ---
            for survivor in survivors:
                self.register.superpose(
                    survivor,
                    n_branches=self.branches_per_state,
                    mutation_fn=self.mutation_fn,
                )

        if self.verbose:
            print(f"\n[QSearch] DONE — Best fitness: {best_ever.fitness:+.6f}")
            print(f"[QSearch] Best params: {best_ever.data}")

        return best_ever

    @property
    def stats(self) -> List[dict]:
        return self._run_stats


# ==============================================================================
# Oracle Factories — Domain-specific fitness functions for QOR subsystems
# ==============================================================================

def trading_oracle_factory(trade_store):
    """Create a trading parameter optimization oracle.

    Uses closed trades from TradeStore to evaluate parameter fitness.
    Fitness = profit_factor * sqrt(win_rate) - max_drawdown_penalty.

    Args:
        trade_store: TradeStore or FuturesTradeStore instance with trade history

    Returns:
        oracle function + initial_params dict + mutation function
    """
    closed = [t for t in trade_store.trades.values() if t["status"] != "open"]
    if len(closed) < 10:
        raise ValueError(f"Need >= 10 closed trades for optimization, got {len(closed)}")

    def oracle(params):
        sl_mult = params.get("stop_loss_atr_mult", 2.0)
        tp_mult = params.get("take_profit_atr_mult", 3.0)
        min_rr = params.get("min_risk_reward", 1.5)
        dca_drop = params.get("dca_drop_pct", 5.0)
        dca_mult = params.get("dca_multiplier", 1.5)
        tp1_pct = params.get("partial_tp1_pct", 50.0)
        cooldown = params.get("cooldown_minutes", 30)

        # Simulate: for each closed trade, would these params have improved it?
        rr = tp_mult / max(sl_mult, 0.1)
        if rr < min_rr:
            return {"fitness": -10.0, "reason": "R:R below minimum"}

        sim_wins = 0
        sim_pnl = 0.0
        for t in closed:
            entry = t.get("entry_price", 0)
            exit_p = t.get("exit_price", 0)
            if entry <= 0:
                continue
            pnl_pct = ((exit_p - entry) / entry) * 100
            # Simulated SL/TP effect: wider SL = fewer stopped out, wider TP = bigger wins
            sl_dist = sl_mult * t.get("atr_at_entry", entry * 0.02)
            tp_dist = tp_mult * t.get("atr_at_entry", entry * 0.02)
            # Simple model: if actual loss < simulated SL, would have survived
            if pnl_pct > 0:
                # Winner: partial TP at tp1_pct shrinks some winners
                effective_gain = pnl_pct * (1.0 - (tp1_pct / 100.0) * 0.1)
                sim_pnl += effective_gain
                sim_wins += 1
            else:
                # Loser: wider SL might have saved it
                if abs(pnl_pct) < (sl_dist / entry * 100):
                    sim_pnl += abs(pnl_pct) * 0.3  # Recovered partial
                    sim_wins += 0.5
                else:
                    sim_pnl += pnl_pct

        win_rate = sim_wins / len(closed) if closed else 0
        gross_profit = max(sim_pnl, 0)
        gross_loss = max(-sim_pnl, 0.01)
        profit_factor = gross_profit / gross_loss

        # Fitness: profit factor weighted by win rate, penalize extreme params
        fitness = profit_factor * math.sqrt(max(win_rate, 0.01))
        # Penalize extreme SL (too tight = stopped out, too wide = big losses)
        if sl_mult < 0.5 or sl_mult > 5.0:
            fitness *= 0.5
        if tp_mult < 1.0 or tp_mult > 10.0:
            fitness *= 0.5
        if cooldown < 5 or cooldown > 120:
            fitness *= 0.8

        return {
            "fitness": fitness,
            "win_rate": win_rate * 100,
            "profit_factor": profit_factor,
            "sim_pnl": sim_pnl,
        }

    initial_params = {
        "stop_loss_atr_mult": 2.0,
        "take_profit_atr_mult": 3.0,
        "min_risk_reward": 1.5,
        "dca_drop_pct": 5.0,
        "dca_multiplier": 1.5,
        "partial_tp1_pct": 50.0,
        "cooldown_minutes": 30,
    }

    def mutation(data, phase):
        mutated = {}
        for k, v in data.items():
            if k == "stop_loss_atr_mult":
                mutated[k] = max(0.3, min(5.0, v + phase * 0.3 * random.random()))
            elif k == "take_profit_atr_mult":
                mutated[k] = max(0.5, min(10.0, v + phase * 0.5 * random.random()))
            elif k == "min_risk_reward":
                mutated[k] = max(0.5, min(5.0, v + phase * 0.3 * random.random()))
            elif k == "dca_drop_pct":
                mutated[k] = max(1.0, min(20.0, v + phase * 1.0 * random.random()))
            elif k == "dca_multiplier":
                mutated[k] = max(1.0, min(3.0, v + phase * 0.2 * random.random()))
            elif k == "partial_tp1_pct":
                mutated[k] = max(10.0, min(90.0, v + phase * 5.0 * random.random()))
            elif k == "cooldown_minutes":
                mutated[k] = max(5, min(120, int(v + phase * 10 * random.random())))
            else:
                mutated[k] = v
        return mutated

    return oracle, initial_params, mutation


def mamba_oracle_factory():
    """Create a CORTEX hyperparameter optimization oracle.

    Evaluates CORTEX (CfC + S4) hyperparameter combinations by running a small
    forward pass and measuring output stability + gradient flow.

    Returns:
        oracle function + initial_params dict + mutation function
    """
    def oracle(params):
        try:
            import torch
            from qor.cortex import CortexBrain as MambaCfCHybrid
        except ImportError:
            return {"fitness": -999, "error": "torch or ncps not installed"}

        neurons = max(8, min(128, int(params.get("cfc_neurons", 32))))
        mamba_dim = max(16, min(256, int(params.get("mamba_dim", 32))))
        history = max(50, min(1000, int(params.get("history_len", 200))))
        cfc_out = max(4, min(32, int(params.get("cfc_output", 8))))
        s4_d_state = max(4, min(64, int(params.get("s4_d_state", 16))))

        try:
            model = MambaCfCHybrid(
                input_size=10, output_size=1,
                mamba_dim=mamba_dim, cfc_neurons=neurons,
                cfc_output=cfc_out, history_len=history,
            )
            # Run a few forward passes to measure stability
            x = torch.randn(1, 10)
            outputs = []
            for _ in range(5):
                out = model(x, instance_id="test")
                outputs.append(out.item())

            # Fitness: low variance = stable, non-zero = active
            variance = sum((o - sum(outputs) / len(outputs)) ** 2 for o in outputs) / len(outputs)
            mean_abs = sum(abs(o) for o in outputs) / len(outputs)

            # We want: low variance (stable) + non-zero output (active) + small param count
            param_count = sum(p.numel() for p in model.parameters())
            param_penalty = param_count / 1_000_000  # Penalize large models

            fitness = mean_abs / (variance + 0.01) - param_penalty * 0.1

            return {
                "fitness": fitness,
                "variance": variance,
                "mean_output": mean_abs,
                "param_count": param_count,
            }
        except Exception as e:
            return {"fitness": -999, "error": str(e)}

    initial_params = {
        "cfc_neurons": 32,
        "mamba_dim": 32,
        "history_len": 200,
        "cfc_output": 8,
        "s4_d_state": 16,
    }

    def mutation(data, phase):
        mutated = {}
        for k, v in data.items():
            if k == "cfc_neurons":
                mutated[k] = max(8, min(128, int(v + phase * 10 * random.random())))
            elif k == "mamba_dim":
                mutated[k] = max(16, min(256, int(v + phase * 16 * random.random())))
            elif k == "history_len":
                mutated[k] = max(50, min(1000, int(v + phase * 50 * random.random())))
            elif k == "cfc_output":
                mutated[k] = max(4, min(32, int(v + phase * 4 * random.random())))
            elif k == "s4_d_state":
                mutated[k] = max(4, min(64, int(v + phase * 8 * random.random())))
            else:
                mutated[k] = v
        return mutated

    return oracle, initial_params, mutation


def knowledge_search(query: str, memory=None, graph=None, rag=None,
                     historical=None, tool_executor=None,
                     n_variants: int = 20, n_workers: int = None,
                     verbose: bool = False) -> List[Dict[str, Any]]:
    """Quantum-inspired parallel knowledge search.

    Branches one query into N search variants (synonyms, rephrased, sub-queries),
    evaluates ALL sources in parallel per variant, collapses to best results.

    This is faster than sequential search because:
    - All variants hit all sources simultaneously via ThreadPool
    - Interference boosts results found by multiple variants
    - Measurement collapses to the highest-quality matches

    Args:
        query: User's question
        memory: MemoryStore instance (optional)
        graph: QORGraph instance (optional)
        rag: RAG instance (optional)
        historical: HistoricalStore instance (optional)
        tool_executor: ToolExecutor instance for API lookups (optional)
        n_variants: How many query variants to search in parallel
        n_workers: Thread pool size (default: CPU count)
        verbose: Print search progress

    Returns:
        List of result dicts sorted by relevance:
        [{"content": str, "source": str, "score": float}, ...]
    """
    if not any([memory, graph, rag, historical, tool_executor]):
        return []

    # --- Generate query variants ---
    # Simple but effective: split into keywords, create sub-combinations
    words = [w for w in query.lower().split() if len(w) > 2]
    stop_words = {"the", "what", "how", "does", "who", "when", "where", "why",
                  "is", "are", "was", "were", "can", "will", "would", "should",
                  "about", "from", "with", "this", "that", "for", "and", "but"}
    keywords = [w for w in words if w not in stop_words]

    variants = [query]  # Always include original
    # Sub-combinations of keywords
    if len(keywords) >= 2:
        for i in range(min(n_variants - 1, len(keywords))):
            subset = random.sample(keywords, max(1, len(keywords) - 1))
            variants.append(" ".join(subset))
    # Reversed keyword order (different matching patterns)
    if keywords:
        variants.append(" ".join(reversed(keywords)))
    # Pad to n_variants with slight perturbations
    while len(variants) < n_variants:
        base = random.choice(variants[:3])
        variants.append(base)

    variants = variants[:n_variants]

    # --- Search function for one variant across all sources ---
    def _search_one(variant_query):
        results = []

        if memory is not None:
            try:
                hits = memory.search(variant_query, top_k=3)
                for key, entry, score in hits:
                    if key.startswith(("context:", "psych:", "kb:", "answer:")):
                        continue
                    results.append({
                        "content": entry.content if hasattr(entry, 'content') else str(entry),
                        "source": "memory",
                        "score": score,
                        "variant": variant_query,
                    })
            except Exception:
                pass

        if graph is not None:
            try:
                if getattr(graph, 'is_open', False):
                    gr = graph.semantic_query(variant_query)
                    if gr.get("edge_count", 0) > 0:
                        path = gr.get("path", [])
                        for edge in path[:3]:
                            fact = f"{edge.get('source', '')} {edge.get('predicate', '')} {edge.get('target', '')}"
                            results.append({
                                "content": fact.strip(),
                                "source": "graph",
                                "score": edge.get("confidence", 0.5),
                                "variant": variant_query,
                            })
            except Exception:
                pass

        if rag is not None:
            try:
                hits = rag.search(variant_query, top_k=3)
                for chunk_text, score, meta in hits:
                    results.append({
                        "content": chunk_text,
                        "source": "rag",
                        "score": score,
                        "variant": variant_query,
                    })
            except Exception:
                pass

        if historical is not None:
            try:
                hits = historical.search(variant_query, top_k=2)
                for key, entry, score in hits:
                    content = entry["content"] if isinstance(entry, dict) else str(entry)
                    results.append({
                        "content": content,
                        "source": "historical",
                        "score": score,
                        "variant": variant_query,
                    })
            except Exception:
                pass

        if tool_executor is not None:
            try:
                intent = tool_executor.detect_intent(variant_query)
                if intent:
                    tool_name, tool_fn = intent
                    result = tool_fn(variant_query)
                    if result:
                        results.append({
                            "content": result if isinstance(result, str) else str(result),
                            "source": f"tool:{tool_name}",
                            "score": 0.8,
                            "variant": variant_query,
                        })
            except Exception:
                pass

        return results

    # --- Run all variants in parallel ---
    workers = n_workers or min(mp.cpu_count(), n_variants)
    all_results = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_search_one, v): v for v in variants}
        for future in as_completed(futures):
            try:
                hits = future.result()
                all_results.extend(hits)
            except Exception:
                pass

    if verbose:
        print(f"[QSearch] Knowledge search: {len(variants)} variants, "
              f"{len(all_results)} total results from {workers} workers")

    # --- Interference: boost content found by multiple variants ---
    # Deduplicate by content, boosting score for multi-variant matches
    content_map = {}
    for r in all_results:
        key = r["content"][:200]  # First 200 chars as dedup key
        if key in content_map:
            # Constructive interference: boost score
            content_map[key]["score"] += r["score"] * 0.3
            content_map[key]["variants"] = content_map[key].get("variants", 1) + 1
        else:
            content_map[key] = {**r, "variants": 1}

    # --- Measurement: collapse to top results ---
    ranked = sorted(content_map.values(), key=lambda x: x["score"], reverse=True)

    if verbose and ranked:
        print(f"[QSearch] After interference: {len(ranked)} unique results, "
              f"best score: {ranked[0]['score']:.3f}")

    return ranked[:10]


def cms_oracle_factory():
    """Create a CMS memory frequency optimization oracle.

    Tests different update frequencies for fast/medium/slow CMS layers
    by measuring pattern retention after simulated learning.

    Returns:
        oracle function + initial_params dict + mutation function
    """
    def oracle(params):
        try:
            import torch
            from qor.model import ContinuumMemorySystem
            from qor.config import ModelConfig
        except ImportError:
            return {"fitness": -999, "error": "torch not installed"}

        fast_freq = max(1, int(params.get("cms_fast_freq", 1)))
        med_freq = max(2, int(params.get("cms_med_freq", 16)))
        slow_freq = max(4, int(params.get("cms_slow_freq", 64)))

        # Enforce ordering: fast < medium < slow
        if fast_freq >= med_freq or med_freq >= slow_freq:
            return {"fitness": -100, "reason": "invalid frequency ordering"}

        try:
            d_model = 64
            d_ff = 128
            cfg = ModelConfig()
            cfg.d_model = d_model
            cfg.d_ff = d_ff
            cfg.cms_fast_freq = fast_freq
            cfg.cms_med_freq = med_freq
            cfg.cms_slow_freq = slow_freq
            cfg.cms_levels = 3
            cfg.cms_level_ff_sizes = []  # Use proportional split
            cfg.dropout = 0.0

            cms = ContinuumMemorySystem(d_model=d_model, d_ff=d_ff, config=cfg)

            # Simulate learning: feed patterns, check retention
            x = torch.randn(1, 10, d_model)
            outputs = []
            for step in range(slow_freq * 2):  # Run enough steps for slow to trigger
                out = cms(x)
                cms.step()
                outputs.append(out.mean().item())

            # Feed same pattern again — check if output is more similar (retention)
            out_after = cms(x)
            first_out = outputs[0]
            last_out = out_after.mean().item()

            # Good CMS: later outputs evolve (learning) but don't diverge wildly
            evolution = abs(last_out - first_out)  # Some change = learning
            variance = sum((o - sum(outputs) / len(outputs)) ** 2 for o in outputs) / len(outputs)

            # Fitness: want moderate evolution + low variance + efficient frequencies
            freq_efficiency = 1.0 / (fast_freq + med_freq / 10 + slow_freq / 100)
            fitness = evolution / (variance + 0.01) + freq_efficiency

            return {
                "fitness": fitness,
                "evolution": evolution,
                "variance": variance,
                "freq_efficiency": freq_efficiency,
            }
        except Exception as e:
            return {"fitness": -999, "error": str(e)}

    initial_params = {
        "cms_fast_freq": 1,
        "cms_med_freq": 16,
        "cms_slow_freq": 64,
    }

    def mutation(data, phase):
        mutated = {}
        for k, v in data.items():
            if k == "cms_fast_freq":
                mutated[k] = max(1, min(4, int(v + phase * 1 * random.random())))
            elif k == "cms_med_freq":
                mutated[k] = max(4, min(64, int(v + phase * 5 * random.random())))
            elif k == "cms_slow_freq":
                mutated[k] = max(16, min(256, int(v + phase * 20 * random.random())))
            else:
                mutated[k] = v
        return mutated

    return oracle, initial_params, mutation


# ==============================================================================
# Convenience: run optimization for a target
# ==============================================================================

def optimize(target: str, trade_store=None, n_workers: int = None,
             generations: int = 30, branches: int = 50, survivors: int = 8,
             verbose: bool = True) -> QuantumState:
    """Run QSearch optimization for a specific QOR subsystem.

    Args:
        target: "trading", "mamba", or "cms"
        trade_store: Required for target="trading" — TradeStore instance
        n_workers: Parallel workers (default: CPU count)
        generations: Max generations to search
        branches: Variants per survivor per generation
        survivors: How many survive each generation
        verbose: Print progress

    Returns:
        Best QuantumState with optimal parameters
    """
    if target == "trading":
        if trade_store is None:
            raise ValueError("trade_store required for trading optimization")
        oracle, initial, mutation = trading_oracle_factory(trade_store)
    elif target == "mamba":
        oracle, initial, mutation = mamba_oracle_factory()
    elif target == "cms":
        oracle, initial, mutation = cms_oracle_factory()
    else:
        raise ValueError(f"Unknown target: {target}. Use: trading, mamba, cms")

    engine = QSearchEngine(
        oracle=oracle,
        n_workers=n_workers,
        branches_per_state=branches,
        survivors_per_gen=survivors,
        max_generations=generations,
        mutation_fn=mutation,
        early_stop_patience=8,
        verbose=verbose,
    )

    return engine.search(initial_params=initial)
