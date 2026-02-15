"""
QOR Query Pool — Parallel Chat Query Processing
==================================================
Runs gate.answer() in background threads so the user can type the next
question while the previous one is still processing.

Model inference is serialized via ConfidenceGate._model_lock (not thread-safe),
but all I/O (tool calls, DB searches, API fetches) runs concurrently.

Usage:
    from qor.query_pool import QueryPool, OutputManager

    output_mgr = OutputManager()
    pool = QueryPool(gate, chat_store, session_id, output_mgr, max_workers=3)

    ticket = pool.submit("What is the price of Bitcoin?")
    # User can immediately type next question...
    # Answer prints automatically when ready.

    pool.wait_all()
    pool.shutdown()
"""

import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QueryTicket:
    """A submitted query tracked by the pool."""
    query_id: int
    question: str
    future: Future
    submitted_at: float = field(default_factory=time.time)


class OutputManager:
    """Thread-safe terminal output for query results.

    Prevents interleaved text when multiple queries complete around
    the same time by serializing all print operations.
    """

    def __init__(self):
        self._print_lock = threading.Lock()

    def print_result(self, ticket: QueryTicket, answer: str,
                     confidence: float = 0.0, source: str = ""):
        """Print a completed query result with prompt restoration."""
        with self._print_lock:
            # Move to new line in case user is mid-typing
            conf_bar = ""
            if confidence > 0:
                bar = "#" * int(confidence * 10) + "." * (10 - int(confidence * 10))
                conf_bar = f"  [{bar}] {source}"

            sys.stdout.write(f"\nQOR: {answer}\n")
            if conf_bar:
                sys.stdout.write(f"{conf_bar}\n")
            sys.stdout.write("\nYou: ")
            sys.stdout.flush()

    def print_status(self, msg: str):
        """Print a status message (e.g., 'processing...')."""
        with self._print_lock:
            sys.stdout.write(f"  {msg}\n")
            sys.stdout.flush()


class QueryPool:
    """Worker pool for parallel chat query processing.

    Submits gate.answer() calls to a thread pool so the main loop
    can accept the next question immediately. Model inference is
    serialized by ConfidenceGate._model_lock; I/O runs concurrently.
    """

    def __init__(self, gate, chat_store, session_id: str,
                 output_mgr: OutputManager, max_workers: int = 3,
                 verbose: bool = True):
        """
        Args:
            gate: ConfidenceGate instance (has .answer() and ._model_lock)
            chat_store: ChatStore instance (thread-safe via ._lock)
            session_id: Current chat session identifier
            output_mgr: OutputManager for thread-safe printing
            max_workers: Max concurrent query workers
            verbose: Pass verbose flag to gate.answer()
        """
        self.gate = gate
        self.chat_store = chat_store
        self.session_id = session_id
        self.output_mgr = output_mgr
        self.verbose = verbose
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                         thread_name_prefix="qor-query")
        self._active: list[QueryTicket] = []
        self._active_lock = threading.Lock()
        self._next_id = 0

    def submit(self, question: str, chat_context: list = None,
               graph=None, profile=None, data_dir: str = "",
               user_id: str = "") -> QueryTicket:
        """Submit a query for background processing.

        Returns a QueryTicket immediately. The answer will print
        automatically when the worker completes.
        """
        query_id = self._next_id
        self._next_id += 1

        future = self._pool.submit(
            self._run_query, question, query_id, chat_context,
            graph, profile, data_dir, user_id,
        )

        ticket = QueryTicket(
            query_id=query_id,
            question=question,
            future=future,
        )

        with self._active_lock:
            self._active.append(ticket)

        return ticket

    def _run_query(self, question: str, query_id: int,
                   chat_context: list = None,
                   graph=None, profile=None,
                   data_dir: str = "", user_id: str = ""):
        """Worker function: runs gate.answer() and prints the result.

        Called in a background thread by the pool.
        """
        try:
            result = self.gate.answer(
                question,
                verbose=self.verbose,
                chat_context=chat_context,
            )

            answer_text = result.get("answer", "")
            confidence = result.get("confidence", 0.0)
            source = result.get("source", "")

            # Save conversation turn (ChatStore is thread-safe)
            if self.chat_store is not None:
                try:
                    self.chat_store.add_turn(self.session_id, question, result)
                except Exception:
                    pass

            # Track user interests
            try:
                from qor.__main__ import _update_interests
                if profile is not None and graph is not None:
                    _update_interests(profile, question, result, data_dir,
                                      graph=graph, user_id=user_id)
            except Exception:
                pass

            # Print result
            if not self.verbose:
                self.output_mgr.print_result(
                    None, answer_text, confidence, source)
            else:
                self.output_mgr.print_result(None, answer_text)

            return result

        except Exception as e:
            self.output_mgr.print_status(f"[Query #{query_id} error: {e}]")
            return None

    def check_done(self):
        """Remove completed futures from the active list."""
        with self._active_lock:
            self._active = [t for t in self._active if not t.future.done()]

    def wait_all(self, timeout: float = 60.0):
        """Block until all active queries complete."""
        with self._active_lock:
            pending = list(self._active)

        for ticket in pending:
            try:
                ticket.future.result(timeout=timeout)
            except Exception:
                pass

        self.check_done()

    def shutdown(self):
        """Shut down the worker pool."""
        self._pool.shutdown(wait=True)

    @property
    def active_count(self) -> int:
        """Number of in-flight queries."""
        with self._active_lock:
            return len([t for t in self._active if not t.future.done()])
