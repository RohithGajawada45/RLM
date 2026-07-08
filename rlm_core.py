import os
import re
import sys
import json
import threading
import traceback
from io import StringIO
from textwrap import dedent
from typing import Any, Optional, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from openai import AzureOpenAI
from colorama import Fore, Style, init

init(autoreset=True)


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0
    reasoning: int = 0
    calls: int = 0
    by_stage: dict = field(default_factory=dict)
    
    _lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)
    
    @property
    def total(self) -> int:
        return self.prompt + self.completion
    
    def add(self, stage: str, prompt: int, completion: int, reasoning: int = 0) -> None:
        with self._lock:
            self.prompt += prompt
            self.completion += completion
            self.reasoning += reasoning
            self.calls += 1

            s = self.by_stage.setdefault(stage, {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0})
            s["calls"] += 1
            s["prompt"] += prompt
            s["completion"] += completion
            s["reasoning"] += reasoning
        
    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "prompt": self.prompt,
            "completion": self.completion,
            "reasoning": self.reasoning,
            "calls": self.calls,
            "by_stage": self.by_stage
        }
    
    def footer(self) -> str:
        if self.calls == 0:
            return ""
        parts= [f"{k}={v['calls']}c/{v['prompt']+v['completion']}t" for k, v in self.by_stage.items()]
        breakdown = " . ".join(parts)
        return (
            f"\n\n---\n"
            f"**Token usage** - {self.total: ,} total "
            f"({self.prompt: ,} prompt + {self.completion: ,} completion"
            + (f", incl. {self.reasoning: ,} reasoning" if self.reasoning else "")
            +f") across {self.calls} LM calls(s)\n"
            f"Breakdown: {breakdown}"
        )

_usage_ctx: ContextVar[Optional[TokenUsage]] = ContextVar("rlm_token_usage", default=None)


def reset_token_usage() -> TokenUsage:
   
   u = TokenUsage()
   _usage_ctx.set(u)
   return u

def get_token_usage() -> TokenUsage:
    u = _usage_ctx.get()
    if u is None:
        u = TokenUsage()
        _usage_ctx.set(u)
    return u


def _enter_token_usage(tracker: TokenUsage):
    
    return _usage_ctx.set(tracker)

def _exit_token_usage(token) -> None:
    _usage_ctx.reset(token)
    
#-------------Azure Client-------------------
#
# NOTE ON MULTI-TENANCY:
# This app is used by many different visitors, each supplying their OWN
# Azure OpenAI credentials via the /static/settings.html page (see
# user_config.py + app.py). To keep every function in this module working
# with "whichever credentials/deployments belong to the caller", each
# AzureOpenAI client we build carries its own config as a plain attribute
# (`client._rlm_config`, a ClientConfig instance). Every place that used to
# read a module-level constant (ROOT_DEPLOYMENT, SUB_DEPLOYMENT, etc.) now
# reads it off the client that was explicitly passed in — which means it
# works correctly even inside the ThreadPoolExecutor fan-out in
# cross_doc.py, since `client` (and therefore its config) is passed as a
# normal function argument, not a context/thread-global.
#
# The os.environ fallbacks below are ONLY used for local/dev runs where no
# per-user config has been set.

from dataclasses import dataclass as _dataclass


@_dataclass
class ClientConfig:
    root_deployment: str
    sub_deployment: str
    embedding_deployment: str
    root_reasoning_effort: str
    sub_reasoning_effort: str


_MINI = os.environ.get("AZURE_GPT_5_MINI_DEPLOYMENT", "gpt-5-mini")
_NANO = os.environ.get("AZURE_GPT_5_NANO_DEPLOYMENT", "gpt-5-nano")

# Fallback defaults (dev-only — a real multi-user deployment always attaches
# a ClientConfig to the client it builds, see make_azure_client() below).
ROOT_DEPLOYMENT = os.environ.get("AZURE_ROOT_DEPLOYMENT", _MINI)
SUB_DEPLOYMENT = os.environ.get("AZURE_SUB_DEPLOYMENT", _MINI)
EMBEDDING_DEPLOYMENT = os.environ.get("EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
ROOT_REASONING_EFFORT = os.environ.get("ROOT_REASONING_EFFORT", "high")
SUB_REASONING_EFFORT = os.environ.get("SUB_REASONING_EFFORT", "high")

_DEFAULT_CONFIG = ClientConfig(
    root_deployment=ROOT_DEPLOYMENT,
    sub_deployment=SUB_DEPLOYMENT,
    embedding_deployment=EMBEDDING_DEPLOYMENT,
    root_reasoning_effort=ROOT_REASONING_EFFORT,
    sub_reasoning_effort=SUB_REASONING_EFFORT,
)


def cfg(client: Optional[AzureOpenAI]) -> ClientConfig:
    """Return the ClientConfig attached to `client`, or the process-wide
    fallback (env-var based) if the client wasn't built with one."""
    return getattr(client, "_rlm_config", _DEFAULT_CONFIG)


def make_azure_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    api_version: Optional[str] = None,
    root_deployment: Optional[str] = None,
    sub_deployment: Optional[str] = None,
    embedding_deployment: Optional[str] = None,
    root_reasoning_effort: Optional[str] = None,
    sub_reasoning_effort: Optional[str] = None,
) -> AzureOpenAI:
    """
    Build an AzureOpenAI client from EXPLICIT credentials (per-user), falling
    back to the process .env only when a value isn't supplied — which keeps
    this usable for local single-user dev exactly as before.
    """
    client = AzureOpenAI(
        azure_endpoint=endpoint or os.environ["AZURE_ENDPOINT"],
        api_key=api_key or os.environ["AZURE_API_KEY"],
        api_version=api_version or os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
    )
    client._rlm_config = ClientConfig(
        root_deployment=root_deployment or ROOT_DEPLOYMENT,
        sub_deployment=sub_deployment or SUB_DEPLOYMENT,
        embedding_deployment=embedding_deployment or EMBEDDING_DEPLOYMENT,
        root_reasoning_effort=root_reasoning_effort or ROOT_REASONING_EFFORT,
        sub_reasoning_effort=sub_reasoning_effort or SUB_REASONING_EFFORT,
    )
    return client


def embed_text(client: AzureOpenAI, text: str, deployment: Optional[str] = None) -> Optional[list]:
    deployment = deployment or cfg(client).embedding_deployment
    try:
        resp=client.embeddings.create(model=deployment, input=text)
        return resp.data[0].embedding
    except Exception as e:
        print(f" [embedding] failed ({type(e).__name__}): {str(e)[:120]}")
        return None

#------------config-------------------

MAX_DEPTH=3
MAX_ROOT_ITERS=30
MAX_OUTPUT_TOKENS=32_768
CHUNK_SIZE_CHARS=200_000
STDOUT_PREVIEW_LEN=4_000


def _create_completion(client:AzureOpenAI, deployment: str, messages: list, reasoning_effort: Optional[str] = None, max_tokens: int= MAX_OUTPUT_TOKENS, stage: str= "other"):
    kwargs= dict(model=deployment, messages=messages, max_completion_tokens=max_tokens)
    if reasoning_effort:
        kwargs["reasoning_effort"]=reasoning_effort
    try:
        resp=client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort", None)
        resp=client.chat.completions.create(**kwargs)
    except Exception as e:
        msg=str(e).lower()
        if "reasoning_effort" in msg or "unrecognized" in msg or "unknown parameter" in msg:
            kwargs.pop("reasoning_effort", None)
            resp=client.chat.completions.create(**kwargs)
        else:
            raise
    
    try:
        u=resp.usage
        if u is not None:
            pt= getattr(u, "prompt_tokens", 0) or 0
            ct= getattr(u, "completion_tokens", 0) or 0
            rt=0
            details= getattr(u, "completion_token_details", None)
            if details is not None:
                rt=getattr(details, "reasoning_tokens", 0) or 0
            get_token_usage().add(stage, pt, ct, rt)
            
            prompt_details= getattr(u, "prompt_token_details", None)
            if prompt_details is not None:
                cached= getattr(prompt_details, "cached_tokens", 0) or 0
                if cached>0 and pt>0:
                    pct= cached/pt * 100
                    print(f" [prompt-cache] {cached:,}/{pt:,} prompt tokens cached ({pct:.0f}%)")
    
    except Exception:
        pass
    return resp

# ---- REPL Environment --------------------------------------------------------

class REPLEnvironment:
    """Persistent Python REPL holding the prompt as the `context` variable."""

    def __init__(
        self,
        context: str,
        llm_query_fn: Callable,
        rlm_query_fn: Callable,
    ):
        self._globals: dict[str, Any] = {
            "context": context,
            "llm_query": llm_query_fn,
            "rlm_query": rlm_query_fn,
            "__builtins__": __builtins__,
            "re": re,
            "json": json,
            "os": os,
        }

        self.final_value: Optional[str] = None

    def run(self, code: str) -> str:
        """Execute code, capture stdout, allow FINAL/FINAL_VAR to set state[final]."""

        stdout_capture = StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        try:
            self._globals["FINAL"] = self._final
            self._globals["FINAL_VAR"] = self._final_var

            exec(
                compile(code, "<repl>", "exec"),
                self._globals,
            )

        except Exception as e:
            tb = traceback.format_exc(limit=3)
            print(f"[REPL ERROR] {e}\n{tb}")

        finally:
            sys.stdout = old_stdout

        return stdout_capture.getvalue()

    def _final(self, answer: str):
        self.final_value = str(answer)
        print("[FINAL ANSWER SET]")

    def _final_var(self, var_name: str):
        if var_name in self._globals:
            self.final_value = str(self._globals[var_name])
            print(f"[FINAL_VAR resolved from '{var_name}']")
        else:
            print(f"[FINAL_VAR ERROR: '{var_name}' not in REPL scope]")

    def is_done(self) -> bool:
        return self.final_value is not None

    def get_state_info(self) -> str:
        skip = {
            "__builtins__",
            "context",
            "llm_query",
            "rlm_query",
            "re",
            "json",
            "os",
            "FINAL",
            "FINAL_VAR",
        }

        user_vars = {
            k: type(v).__name__
            for k, v in self._globals.items()
            if k not in skip
        }

        return str(user_vars) if user_vars else "(no user variables yet)"
        

# --- Leaf sub-LM call (single, non-recursive) -------------------------------

def _stage_from_label(label: str) -> str:
    lbl = (label or "").lower()

    if lbl.startswith("describe"):
        return "describe"

    if "router" in lbl:
        return "router"

    if "aggregator" in lbl:
        return "aggregator"

    if "fallback" in lbl:
        return "fallback"

    if "sub-lm" in lbl or "sub-rlm" in lbl:
        return "sub_lm"

    return "other"


def llm_call(
    prompt: str,
    client: AzureOpenAI,
    label: str = "sub-LM",
    deployment: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> str:

    deployment = deployment or cfg(client).sub_deployment
    reasoning_effort = reasoning_effort or cfg(client).sub_reasoning_effort

    print(
        f"{Fore.CYAN} [{label}] {deployment} (effort={reasoning_effort}, "
        f"{len(prompt)} chars)...{Style.RESET_ALL}"
    )

    try:
        resp = _create_completion(
            client,
            deployment=deployment,
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort=reasoning_effort,
            stage=_stage_from_label(label),
        )

        return resp.choices[0].message.content or ""

    except Exception as e:
        return f"[LLM ERROR: {e}]"


# --- FINAL / FINAL_VAR extraction (balanced-paren, not regex) ----------------

# The earlier regex `FINAL\(((.+?)\)\)\s*$` with MULTILINE was broken: non-greedy
# + line-end anchor caused it to stop at the first ')' followed by EOL — so any
# answer containing parens like "(74% of total revenue)" got chopped at that
# inner ')'. The parser below walks paren depth properly, handles triple-quoted
# string literals, and tolerates the model writing FINAL multiple times.

_FINAL_OPEN_RE = re.compile(r'(?:^|\n)[ \t]*FINAL\(', re.MULTILINE)
_FINAL_VAR_OPEN_RE = re.compile(r'(?:^|\n)[ \t]*FINAL_VAR\(', re.MULTILINE)
_QUOTE_WRAPPERS = ('"""', "'''", '"', "'")


def _strip_string_wrapper(content: str) -> str:
    """Strip a balanced surrounding string-quote wrapper if present."""

    content = content.strip()

    for q in _QUOTE_WRAPPERS:
        if (
            content.startswith(q)
            and content.endswith(q)
            and len(content) >= 2 * len(q)
        ):
            return content[len(q):-len(q)]

    return content


def _extract_final_call(text: str) -> Optional[str]:
    """Extract content of the LAST FINAL(...) call appearing on its own line.

    Strategy: find the last `FINAL(` opener, then take everything from there to
    the LAST ')' in the message. This is robust against:
      - Nested parens inside the content like "(74% of total)"
      - Numbered list items like "1) ...", "2) ..." that fooled the previous
        depth-counting parser into closing early
      - Triple-quoted, single-quoted, double-quoted, or bare-text content
      - Multi-line content

    The model's API response always ends with the FINAL terminator (or is cut
    off entirely), so the last ')' after the FINAL( opener is the right close.
    If the output was truncated mid-FINAL with no closing ')', return everything
    from FINAL( onward as a best-effort recovery.
    """

    opens = list(_FINAL_OPEN_RE.finditer(text))
    if not opens:
        return None

    m = opens[-1]          # use the last FINAL() the model wrote
    start = m.end()

    end = text.rfind(')', start)
    if end == -1:
        # Output truncated before close — recover everything we have
        return _strip_string_wrapper(text[start:].rstrip())

    return _strip_string_wrapper(text[start:end])

def _extract_final_var(text: str) -> Optional[str]:
    """Extract the variable name from FINAL_VAR(<name>). Returns None if absent.

    Variable names are simple identifiers (no parens, no numbered lists), so a
    light parse is enough — find the LAST ')' after the opener and take what's
    between as the candidate, then match a leading identifier.
    """

    opens = list(_FINAL_VAR_OPEN_RE.finditer(text))
    if not opens:
        return None

    m = opens[-1]
    start = m.end()
    end = text.rfind(')', start)

    inner = text[start:end] if end != -1 else text[start:].rstrip().rstrip(')')
    inner = _strip_string_wrapper(inner).strip()

    name_match = re.match(r'\w+', inner)
    return name_match.group(0) if name_match else None


# --- System prompt (mirrors Appendix C.1) ------------------------------------

def build_system_prompt(
    context_total_length: int,
    context_lengths: list,
    depth: int,
) -> str:
    """
    Constructs the RLM system prompt.
    - Always exposes `llm_query` (prompt 1a from Appendix C.1)
    - When depth + 1 <= MAX_DEPTH, also documents `rlm_query` (diff 1c).
    """

    can_recurse = (depth + 1) <= MAX_DEPTH

    rlm_query_block = ""
    rlm_example = ""
    choosing_block = ""
    final_item_num = 3

    if can_recurse:
        final_item_num = 4

        rlm_query_block = (
            "3. An `rlm_query(context, query)` function for COMPLEX sub-tasks that benefit "
            "from iterative, multi-step reasoning. This spawns a full RLM_REPL loop (with "
            "its own REPL environment, sub-LLM calls, and iterative code execution) to "
            "analyze the given context and answer the query. Use this when a sub-task is too "
            "difficult for a single `llm_query` call — for example, when the sub-task itself "
            "requires chunking, aggregation, or multi-step analysis. Note: if the maximum "
            "recursion depth is reached, `rlm_query` automatically falls back to `llm_query`.\n"
        )

        rlm_example = dedent("""
            For a truly complex sub-task, delegate it to a full RLM_REPL loop:
                ```repl
                sub_context = "\\n".join(context_chunks[500:1000])  # a large sub-section
                answer = rlm_query(sub_context, "What are the key themes across these documents?")
                print(f"Deep analysis result: {answer}")
                ```
        """).strip()

        choosing_block = dedent("""
            **Choosing between `llm_query` and `rlm_query`:**
            - Use `llm_query(prompt)` for simple sub-tasks: summarize a chunk, extract a fact,
              answer a direct question. Single LLM call, fast and cheap.
            - Use `rlm_query(context, query)` when a sub-task is itself complex enough to need
              iterative reasoning with code execution — e.g., analyzing a very large sub-context
              that needs its own chunking, or a multi-step reasoning chain. Slower and more
              expensive, but more powerful.
        """).strip()

    prompt = dedent(f"""
        You are tasked with answering a query with associated context. You can access, transform,
        and analyze this context interactively in a REPL environment that can recursively query
        sub-LLMs, which you are strongly encouraged to use as much as possible. You will be
        queried iteratively until you provide a final answer.

        Your context is a document with {context_total_length} total characters, and is broken up
        into chunks of char lengths: {context_lengths}.

        The REPL environment is initialized with:
        1. A `context` variable that contains extremely important information about your query.
           Check the content of `context` to understand what you are working with. Look through
           it sufficiently as you answer your query.
        2. An `llm_query(prompt)` function that allows you to query an LLM (that can handle around
           500K chars) inside your REPL environment. Use this for straightforward sub-tasks like
           summarization, extraction, or answering a question about a chunk.
        {rlm_query_block}{final_item_num}. The ability to use `print()` statements to view the
           output of your REPL code and continue your reasoning.

        You will only see truncated outputs from the REPL environment, so use the query function
        on variables you want to analyze, especially for semantic analysis. Use REPL variables as
        buffers to build up your final answer.

        Look through the entire context in the REPL before answering. An example strategy: first
        inspect the context and figure out a chunking strategy, then break the context into smart
        chunks, query an LLM per chunk with a focused question and save answers to a buffer, then
        query an LLM with the buffer to produce your final answer.

        Sub-LLMs are powerful — they can fit around 500K characters in their context window, so
        don't be afraid to put a lot of context into them. A viable strategy is feeding 10
        documents per sub-LLM query. Analyze the input data and see if it fits in just a few
        sub-LLM calls!

        {choosing_block}

        When you want to execute Python code in the REPL environment, wrap it in triple backticks
        with the `repl` language identifier. Example — search for a specific fact in a long
        context by chunking:

        ```repl
        chunk = context[:10000]
        answer = llm_query(f"What is the magic number in the context? Chunk: {{chunk}}")
        print(answer)
        ```

        Iterative chunked aggregation example:

        ```repl
        query = "How many jobs did the man behind 'The Great Gatsby' have?"
        chunk_size = len(context) // 10
        answers = []

        for i in range(10):
            s = i * chunk_size
            e = (i + 1) * chunk_size if i < 9 else len(context)
            chunk_str = context[s:e]

            answer = llm_query(
                f"Try to answer: {{query}}. Documents:\\n{{chunk_str}}. "
                f"Only answer if confident based on the evidence."
            )

            answers.append(answer)

        final_answer = llm_query(
            f"Aggregate all per-chunk answers and answer the original "
            f"query: {{query}}\\n\\nAnswers:\\n" + "\\n".join(answers)
        )

        ```

        Header-based chunking example:

        ```repl
        import re

        sections = re.split(r'### (.+)', context)
        buffers = []

        for i in range(1, len(sections), 2):
            header, info = sections[i], sections[i + 1]

            summary = llm_query(
                f"Summarize this {{header}} section: {{info}}"
            )
            buffers.append(f"{{header}}: {{summary}}")

        final_answer = llm_query(
            f"Based on these summaries, answer the original query: "
            f"{{query}}\\n\\nSummaries:\\n" + "\\n".join(buffers)
        )
        ```

        {rlm_example}

        In the next step, you can return FINAL_VAR(final_answer).

        IMPORTANT: When you are done with the iterative process, you MUST provide a final answer
        inside a FINAL function — NOT in code. Do not use these tags unless you have completed
        your task. You have two options:
        1. Use FINAL(your final answer here) to provide the answer directly.
        2. Use FINAL_VAR(variable_name) to return a REPL variable as your final output.

        Think step by step carefully, plan, and execute that plan immediately in your response —
        do not just say "I will do this" or "I will do that". Output to the REPL environment and
        recursive LLMs as much as possible. Remember to explicitly answer the original query in
        your final answer.

        ------------------------------------------------------------------------
        EVIDENCE RULES (apply to every FINAL you produce):
        ------------------------------------------------------------------------

        • VERIFY every number, name, date, and quote against what the REPL has
          actually printed for THIS document. Do NOT use training-data memory.
          Well-known entities (companies, people, places) have stale facts in
          your memory; trust ONLY the REPL output.
        • Anything inside quotation marks in your FINAL must be VERBATIM from
          REPL output — same words, same order. If you cannot quote it
          verbatim, paraphrase without quotation marks instead.
        • If you cite a section number or heading (e.g. "Section 6.4"), you
          MUST also quote the section's heading line verbatim from the REPL.
          Do not invent or paraphrase section titles.
        • If a fact is not in the REPL output, do not include it. "Not found
          in this document" is a valid finding when supported by searches.

        ------------------------------------------------------------------------
        SEARCH STRATEGY — avoid the rigid-heading trap:
        ------------------------------------------------------------------------

        Documents extracted from PDFs often have unusual whitespace and
        layout. A heading like "CONSOLIDATED STATEMENTS OF OPERATIONS" may
        appear in the source with spaces between every letter, line breaks
        in the middle, or vertical separators. Strict substring searches
        for canonical headings WILL OFTEN FAIL even when the section is
        clearly present.

        Strategy when the first 1-2 heading searches return zero matches:
        1. STOP repeating heading-variant searches. Each additional "Maybe
           it's spelled CONSOLIDATED STATEMENT (singular) instead" attempt
           burns an iteration without adding information.
        2. PIVOT to content-based regex on the answer terms themselves. For
           a question about "energy generation and storage revenue", search
           directly for the regex pattern `energy\\s+generation\\s+and\\s+storage`
           (case-insensitive, whitespace-flexible). The actual numbers you
           need are next to that phrase, not next to the heading.
        3. Use the position of a content hit, then slice 2,000–4,000 chars
           around it and PRINT THE SLICE — that surrounding text usually
           contains the table the question is asking about.

        Sub-revenue and breakdown questions specifically:
        Questions about *segment revenue*, *product-line revenue*, or any
        revenue *broken down* by category (Energy vs Automotive, Sales vs
        Leasing, Subscription vs Services, by Geography) typically live in
        a different table than the top-line "Total revenue" line. The
        breakdown table is usually:
            • In the MD&A "Results of Operations" section, OR
            • In a "Segment Information" note in the audited financial
              statements

        Both ARE audited and ARE the correct source for these questions.
        Do not waste iterations trying to find them under the top-line
        Consolidated Statement of Profit/Loss heading; the breakdown is
        usually elsewhere in the document. Use content-based regex on the
        category name (e.g. "energy generation and storage") and slice
        around the hit.

        Iteration budget awareness:
        If you have run 5+ iterations and still haven't located the
        specific table the question needs, your search strategy is wrong —
        change approach (different regex, broader slice, different
        landmark) rather than running yet another variant of the same
        failing search.
    """).strip()

    return prompt

# --- RLM main loop (Algorithm 1) ---------------------------------------------

def rlm(
    context: str,
    query: str,
    client: AzureOpenAI,
    depth: int = 0,
    verbose: bool = True,
    stage_override: Optional[str] = None,
) -> str:
    """Recursive Language Model — faithful implementation of Algorithm 1.

    stage_override (optional): when set, the root LM iterations are recorded
    under this stage name instead of "root_lm". Used by cross_doc fan-out to
    bucket each per-doc sub-RLM separately (e.g. "fanout:adobe").
    """

    indent = "  " * depth

    if verbose:
        print(f"\n{Fore.MAGENTA}{indent}{'=' * 60}")
        print(
            f"{indent}RLM (depth={depth}/{MAX_DEPTH}) | context={len(context)} chars "
            f"| query={query[:80]}"
        )
        print(f"{indent}{'=' * 60}{Style.RESET_ALL}")

    call_count = [0]

    # --- leaf sub-LM call: always a single non-recursive LM call -------------

    def llm_query(sub_prompt: str) -> str:
        call_count[0] += 1
        label = f"sub-LM#{call_count[0]} d={depth}"
        return llm_call(
            sub_prompt,
            client,
            label=label,
            deployment=cfg(client).sub_deployment,
        )

    # --- sub-RLM call: REAL recursion when depth permits, else fallback ------

    def rlm_query(sub_context: str, sub_query: str) -> str:
        call_count[0] += 1

        if depth + 1 <= MAX_DEPTH:
            label = f"sub-RLM#{call_count[0]} d={depth}→d={depth+1}"

            if verbose:
                print(
                    f"{Fore.YELLOW}{indent}  [{label}] spawning recursive RLM "
                    f"({len(sub_context)} chars){Style.RESET_ALL}"
                )

            return rlm(
                sub_context,
                sub_query,
                client,
                depth=depth + 1,
                verbose=verbose,
            )

        # Paper: "if max depth is reached, rlm_query automatically falls back to llm_query"

        label = f"sub-LM(fallback)#{call_count[0]} d={depth}"

        full_prompt = (
            f"Context:\n{sub_context}\n\n"
            f"Query: {sub_query}\n\n"
            f"Answer the query based only on the context above."
        )

        return llm_call(
            full_prompt,
            client,
            label=label,
            deployment=cfg(client).sub_deployment,
        )

    # --- state ← InitREPL(prompt=context); state ← AddFunction(sub_RLM) ------

    repl = REPLEnvironment(context, llm_query, rlm_query)

    # --- hist ← [Metadata(state)] --------------------------------------------

    n_chunks = max(
        1,
        (len(context) + CHUNK_SIZE_CHARS - 1) // CHUNK_SIZE_CHARS,
    )

    chunk_lens = [
        min(
            CHUNK_SIZE_CHARS,
            len(context) - i * CHUNK_SIZE_CHARS
        )
        for i in range(n_chunks)
    ]

    system_prompt = build_system_prompt(
        len(context),
        chunk_lens,
        depth,
    )

    user_metadata = dedent(f"""
        Document metadata:
        - Total characters: {len(context)}
        - Chunks of ~{CHUNK_SIZE_CHARS} chars: {n_chunks}
        - Chunk sizes: {chunk_lens}
        - First 400 chars preview:
        ---

        {context[:400]}

        ---

        QUERY: {query}
    """).strip()

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_metadata},
    ]
    
    # --- while True: ---------------------------------------------------------

    for iteration in range(MAX_ROOT_ITERS):
        if verbose:
            print(
                f"\n{Fore.YELLOW}{indent}[Iter {iteration + 1}] Root LM "
                f"({cfg(client).root_deployment})…{Style.RESET_ALL}"
            )

        # code ← LLM_M(hist)
        try:
            resp = _create_completion(
                client,
                deployment=cfg(client).root_deployment,
                messages=history,
                reasoning_effort=cfg(client).root_reasoning_effort,
                stage=stage_override or "root_lm",
            )
            lm_output = resp.choices[0].message.content or ""
        except Exception as e:
            lm_output = f"[ROOT LM ERROR: {e}]"

        if verbose:
            preview = lm_output[:300].replace("\n", " ")
            print(
                f"{Fore.WHITE}{indent}Root LM: {preview}…{Style.RESET_ALL}"
            )

        # hist ← hist || code (the assistant turn carries the code blocks verbatim)
        history.append(
            {
                "role": "assistant",
                "content": lm_output,
            }
        )

        # (state, stdout) ← REPL(state, code)
        # Run code FIRST so that any variables the LM intends to expose via
        # FINAL_VAR exist before we look them up. The previous order checked
        # inline FINAL_VAR first, which terminated the loop with a
        # [var '...' not found] error when the LM wrote a code block AND an
        # inline FINAL_VAR for a variable that block was meant to define.

        code_blocks = re.findall(
            r"```(?:repl|python)[ \t]*\r?\n(.*?)```",
            lm_output,
            re.DOTALL,
        )

        all_stdout = ""

        for code in code_blocks:
            if verbose:
                code_preview = code[:150].replace("\n", " ")
                print(
                    f"{Fore.CYAN}{indent}  [REPL exec]: "
                    f"{code_preview}…{Style.RESET_ALL}"
                )

            stdout = repl.run(code)
            all_stdout += stdout

            if verbose and stdout:
                print(
                    f"{Fore.WHITE}{indent}  [REPL out]: "
                    f"{stdout[:300]}{Style.RESET_ALL}"
                )

        # if state[Final] is set: return state[Final]
        # (code called FINAL/FINAL_VAR)

        if repl.is_done():
            if verbose:
                print(
                    f"{Fore.GREEN}{indent}[FINAL from REPL]: "
                    f"{str(repl.final_value)[:200]}{Style.RESET_ALL}"
                )

            return str(repl.final_value)

        # Defensive: inline FINAL_VAR(...) outside a code block.
        # Now that code has already run, any variable the model intended
        # to expose exists.

        final_var_match = _extract_final_var(lm_output)

        if final_var_match is not None:
            if final_var_match in repl._globals:
                answer = repl._globals[final_var_match]

                if verbose:
                    print(
                        f"{Fore.GREEN}{indent}[FINAL_VAR inline '{final_var_match}']: "
                        f"{str(answer)[:200]}{Style.RESET_ALL}"
                    )

                return str(answer)

            # Variable still doesn't exist — model called FINAL_VAR prematurely.
            # Don't terminate; surface the error to the LM and continue.

            all_stdout += (
                f"\n[system: FINAL_VAR('{final_var_match}') was called "
                f"but '{final_var_match}' is not defined in the REPL. "
                f"Define it via code first, or use FINAL(...) directly.]"
            )

            if verbose:
                print(
                    f"{Fore.YELLOW}{indent}[FINAL_VAR '{final_var_match}' missing - "
                    f"continuing loop]{Style.RESET_ALL}"
                )

        # Defensive: inline FINAL(...) outside a code block.
        # Uses last-')' parser — robust against numbered lists and nested parens.

        final_content = _extract_final_call(lm_output)

        if final_content is not None:
            if verbose:
                print(
                    f"{Fore.GREEN}{indent}[FINAL inline]: "
                    f"{final_content[:200]}{Style.RESET_ALL}"
                )

            return final_content

        # hist ← hist || Metadata(stdout) (truncated to constant size)

        if len(all_stdout) > STDOUT_PREVIEW_LEN:
            stdout_meta = (
                all_stdout[:STDOUT_PREVIEW_LEN]
                + f"\n...[truncated, total {len(all_stdout)} chars]"
            )
        else:
            stdout_meta = all_stdout or "(no output)"

        user_feedback = (
            f"REPL stdout (truncated to {STDOUT_PREVIEW_LEN} chars):\n"
            f"---\n{stdout_meta}\n---\n"
            f"Current REPL variables: {repl.get_state_info()}\n"
            f"Sub-LM/RLM calls so far at this depth: {call_count[0]}\n"
            f"Continue your analysis. When complete, call FINAL(answer) or "
            f"FINAL_VAR('var_name') — not inside a code block."
        )

        history.append(
            {
                "role": "user",
                "content": user_feedback,
            }
        )

    # --- Fallback when MAX_ROOT_ITERS hit
    # (paper has no explicit fallback) ---

    if verbose:
        print(
            f"{Fore.RED}{indent}[MAX ITERS REACHED] "
            f"forcing final answer{Style.RESET_ALL}"
        )

    fallback_prompt = (
        f"You have reached the maximum iterations. "
        f"Based on everything computed so far, "
        f"provide your best answer to:\n{query}\n"
        f"REPL variables available: {repl.get_state_info()}"
    )

    history.append(
        {
            "role": "user",
            "content": fallback_prompt,
        }
    )

    try:
        resp = _create_completion(
            client,
            deployment=cfg(client).root_deployment,
            messages=history,
            reasoning_effort=cfg(client).root_reasoning_effort,
            stage=(
                f"{stage_override}_fallback"
                if stage_override
                else "fallback"
            ),
        )

        return resp.choices[0].message.content or "(no answer)"

    except Exception as e:
        return f"[FALLBACK ERROR: {e}]"