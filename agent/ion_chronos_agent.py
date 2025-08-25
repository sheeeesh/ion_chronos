# agent/ion_chronos_agent.py
"""
Ion Chronos Agent — tool-calling assistant with persistent memory and a clean Markdown style.
"""

from __future__ import annotations

import os
from typing import Optional

from tools.io_paths import WORKSPACE

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

# ---------- Project tools (keep the tools. prefix) ----------
from tools.astro_dataset import build_astro_dataset
from tools.backtest import backtest_signal
from tools.rl_train import train_rl_agent
from tools.web_search import web_search
from tools.file_manager import write_file, read_file
from tools.fs_access import ls, read_text, write_text
from tools.os_exec import exec_cmd


# Optional: one-shot pipeline if present
try:
    from tools.pipeline import run_rl_astro_pipeline, PipelineArgs  # type: ignore
    HAS_PIPE = True
except Exception:
    HAS_PIPE = False

DELIM = "\n---\n"


def _wrap_write_file(inp: str) -> str:
    if DELIM not in inp:
        return "Usage:\nwrite_file relative/path.ext\n---\ncontent..."
    path, content = inp.split(DELIM, 1)
    return write_file(path.strip(), content)


def _wrap_read_file(inp: str) -> str:
    path = inp.strip()
    if not path:
        return "Usage: read_file relative/path.ext"
    return read_file(path)


# ---------- Arg schemas ----------

class BuildAstroArgs(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., BTC-USD or SPY")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD (optional)")
    timeframe: str = Field("1d", description="Bar interval, e.g., '1d', '5m', '15m', '1h'")
    lat: float = Field(40.7128, description="Latitude for Placidus houses (default NYC)")
    lon: float = Field(-74.0060, description="Longitude for Placidus houses (default NYC)")
    orb_deg: float = Field(3.0, description="Aspect orb in degrees")
    cache_parquet: Optional[str] = Field(None, description="Optional parquet cache path for astro features")


class BacktestArgs(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="End date YYYY-MM-DD (optional)")
    strategy: str = Field("ma_cross", description="Strategy name (default: ma_cross)")


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Number of results to fetch (default 5)")


class RLTrainArgs(BaseModel):
    ticker: str = Field(..., description="Ticker to train on (e.g., BTC-USD)")
    total_timesteps: int = Field(200_000, description="PPO training steps")
    window_size: int = Field(30, description="Observation window length")
    cost: float = Field(0.0005, description="Transaction cost per position change")
    slippage: float = Field(0.0005, description="Slippage cost per position change")
    continuous: bool = Field(False, description="Use continuous action space [-1,1]")
    n_envs: int = Field(1, description="Number of vectorized envs")
    use_vecnorm: bool = Field(True, description="Enable VecNormalize for obs/reward")
    lr: float = Field(3e-4, description="Learning rate (start of linear schedule)")
    ent_coef: float = Field(0.01, description="Entropy bonus coefficient")
    gamma: float = Field(0.99, description="Discount factor")
    seed: int = Field(42, description="Random seed")
    max_trades_per_day: int = Field(30, description="Daily trade cap")
    day_stop_dd: float = Field(0.02, description="Daily stop after 2% drawdown")
    # Stronger control & iterative training options
    train_split: float = Field(0.70, ge=0.5, le=0.95, description="Train/OOS split fraction")
    extra_timesteps_after: int = Field(0, description="Extra PPO steps to continue after main training")
    resume_from: Optional[str] = Field(None, description="Path to an existing model.zip to resume training")


# ---------- Memory (persistent chat history) ----------

_MEMORY_DIR = os.path.join(WORKSPACE, ".memory")
os.makedirs(_MEMORY_DIR, exist_ok=True)


def _get_history(session_id: str):
    """
    Persistent chat history bound to session_id (file-based).
    Uses the path-style constructor to avoid version-specific kwargs.
    """
    file_path = os.path.join(_MEMORY_DIR, f"{session_id}.json")
    return FileChatMessageHistory(file_path=file_path)


# ---------- Agent factory ----------

def create_agent():
    # Model note: some deployments only support temperature=1 for this model
    llm = ChatOpenAI(model="gpt-5-mini", temperature=1, timeout=60, max_retries=2)

    def web_search_bound(query: str, max_results: int = 5) -> str:
        try:
            return web_search(query, max_results=max_results)
        except TypeError:
            # fallback if your tool expects an llm
            return web_search(query, llm=llm, max_results=max_results)

    tools = [
        StructuredTool.from_function(
            func=build_astro_dataset,
            name="build_astro_dataset",
            description=(
                "Build an astro-enhanced dataset with Swiss Ephemeris features "
                "(tropical/geocentric, Placidus houses) for a symbol/date range. "
                "Supports timeframe, lat/lon, orb, and optional cache parquet."
            ),
            args_schema=BuildAstroArgs,
        ),
        StructuredTool.from_function(
            func=backtest_signal,
            name="backtest_signal",
            description="Backtest a strategy on a ticker over a date range (default strategy=ma_cross).",
            args_schema=BacktestArgs,
        ),
        StructuredTool.from_function(
            func=train_rl_agent,
            name="rl_train",
            description="Train a PPO RL agent with hyperparameters and realism knobs (with OOS eval, resume/extra steps).",
            args_schema=RLTrainArgs,
        ),
        StructuredTool.from_function(
            func=web_search_bound,
            name="web_search",
            description="Search the web and summarize results.",
            args_schema=WebSearchArgs,
        ),
        Tool(name="write_file", func=_wrap_write_file, description="Write file. Input: 'path\\n---\\ncontent...'"),
        Tool(name="read_file",  func=_wrap_read_file,  description="Read file. Input: 'path'"),
    ]

    if HAS_PIPE:
        tools.append(
            StructuredTool.from_function(
                func=run_rl_astro_pipeline,
                name="run_rl_astro_pipeline",
                description="One-shot pipeline: build dataset → train RL → (optional) baseline + eval.",
                args_schema=PipelineArgs,
            )
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are Ion Chronos, an expert AI trading assistant.\n"
             "Style rules (always):\n"
             "• Use concise, plain English.\n"
             "• Format with Markdown: short intro, then bullets or small sections.\n"
             "• Put file paths in `code`, code in fenced blocks.\n"
             "• No tool-call traces. End with 1–2 crisp next steps."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=16,
    )

    return RunnableWithMessageHistory(
        executor,
        _get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


if __name__ == "__main__":
    print(ls("~"))  # expands, checks allowlist
    print(read_text("~/Documents/notes.txt"))
    print(exec_cmd("ls -la", workdir="~"))
