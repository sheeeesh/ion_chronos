# main.py
import os
import sys
import argparse
from uuid import uuid4
from datetime import datetime
from dotenv import load_dotenv
from pyfiglet import Figlet


# Pretty terminal rendering (Markdown + styled input). Falls back to plain print if not installed.
_RICH = False
console = None
try:
    from rich import box
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    from rich.theme import Theme
    from rich.traceback import install as rich_traceback

    THEME = Theme({
        # accents & chrome
        "accent": "bold red",
        "border": "red",
        "sep": "grey50",
        "warn": "#FFC107",  # amber
        "err": "bold bright_red",
        "session": "italic dim",
        # user / assistant cards
        "user.border": "#008080",  # teal
        "user.title": "bold #008080",
        "user.text": "#008080",
        "assistant.border": "#FF00FF",  # magenta
        "assistant.title": "bold #FF00FF",
        "assistant.text": "#FF00FF",
    })
    console = Console(theme=THEME)
    rich_traceback(show_locals=False)
    _RICH = True
except Exception:
    pass

# Ensure project root is importable BEFORE importing the agent
ROOT = os.path.dirname(os.path.abspath(__file__))
# Lock CWD to project root so relative paths are stable
try:
    os.chdir(ROOT)
except Exception:
    pass
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv()  # load OPENAI_API_KEY and any other vars from .env

# Import the factory that returns a RunnableWithMessageHistory agent
try:
    from agent.ion_chronos_agent import create_agent
except ImportError as e:
    if _RICH and console is not None:
        console.print(Panel.fit(Text(
            "Failed to import create_agent from agent/ion_chronos_agent.py.\n"
            "Make sure the project layout matches 'agent/ion_chronos_agent.py'.",
            style="err"
        ), border_style="red", box=box.ROUNDED))
    else:
        print("Failed to import create_agent from agent.ion_chronos_agent.py.")
        print("Make sure the project layout matches 'agent/ion_chronos_agent.py'.")
    raise e


BANNER_TEXT = """\
Ion Chronos AI Trading Assistant is now running.
Type your queries or commands.

Quick commands:
  help                 Show this help
  doctor               Run a self-check (imports + workspace permissions)
  validate <TICKER>    Validate recent backtest execution
  list validated       List tickers with validated executions
  new session          Start a fresh memory session (new session_id)
  use session <name>   Switch to a named session_id (persistent)
  exit / quit          Leave the assistant
"""

HELP = """\
[b]Commands[/b]
  • [bold]help[/bold] — Show this help and some tips.
  • [bold]doctor[/bold] — Verify imports and workspace R/W in one go.
  • [bold]validate <TICKER>[/bold] — Validate recent backtest execution for a ticker.
  • [bold]list validated[/bold] — List all tickers with recent validated executions.
  • [bold]new session[/bold] — Start a brand-new memory session with a random session_id.
  • [bold]use session <name>[/bold] — Switch to or create a named session (e.g. research-btc).
  • [bold]exit[/bold] / [bold]quit[/bold] — Leave the assistant.

[i]Notes[/i]
- Memory depends on a stable session_id across turns (this REPL always passes it).
- You can also set [code]ION_SESSION_ID[/code] in the environment to pick the initial session.
- Use [code]validate SPY[/code] to check if recent backtest results are authentic.
"""

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ion Chronos CLI")
    p.add_argument(
        "--session",
        default=os.environ.get("ION_SESSION_ID", "default"),
        help="Initial session_id to use for memory (default: %(default)s)",
    )
    p.add_argument(
        "--noninteractive",
        metavar="PROMPT",
        help="Run a single prompt non-interactively, print the assistant's reply, then exit.",
    )
    return p.parse_args()


def format_error(e: Exception) -> str:
    """Tidy error strings for the REPL."""
    msg = str(e)
    if "Missing keys ['session_id']" in msg:
        msg += "  (Hint: ensure we pass {'configurable': {'session_id': '<id>'}} in invoke/stream.)"
    return msg


def _banner(session_id: str) -> None:
    if _RICH and console is not None:
        console.print(
            Panel.fit(
                Text("⚡ Ion Chronos — AI Trading Assistant", style="assistant.title"),
                subtitle="Type help • doctor • new session • use session <name> • exit",
                subtitle_align="left",
                border_style="assistant.border",
                box=box.ROUNDED,
            )
        )
        console.print(f"[session]session:[/session] [accent]{session_id}[/accent]\n")
        console.print(Markdown(BANNER_TEXT))
    else:
        print(BANNER_TEXT)
        print(f"[session: {session_id}]")


def render_reply(text: str) -> None:
    """Render assistant text; prefer Markdown when Rich is available."""
    if _RICH and console is not None:
        console.print(Rule(style="sep"))
        try:
            md = Markdown(text)
            body = md
        except Exception:
            body = Text(str(text), style="assistant.text")
        console.print(
            Panel(
                body,
                title="[assistant.title]⚡ Ion Chronos[/assistant.title]",
                border_style="assistant.border",
                padding=(0, 1),
                box=box.ROUNDED,
            )
        )
        return
    print(f"Assistant: {text}")


def read_user_input(session_id: str) -> str:
    """Prompt for input and reprint it in a styled panel so it stands out."""
    if _RICH and console is not None:
        ts = datetime.now().strftime("%H:%M:%S")
        raw = console.input(f"[accent]{ts}[/accent] [user.title]You ({session_id})[/user.title] › ").strip()
        if raw:
            console.print(
                Panel(
                    Text(raw, style="user.text"),
                    title="[user.title]🧑‍💻 You[/user.title]",
                    title_align="left",
                    border_style="user.border",
                    padding=(0, 1),
                    box=box.ROUNDED,
                )
            )
        return raw
    # Fallback without Rich
    return input("\nUser: ").strip()


def _doctor() -> str:
    # 1) Imports
    import importlib
    mods = [
        "agent.ion_chronos_agent",
        "tools.file_manager",
        "tools.rl_train",
        "tools.astro_dataset",
        "tools.astro_features",
        "tools.backtest",
        "tools.classifier_backtest",
        "tools.validation",
        "tools.result_checker",
        "tools.safe_classifier_backtest",
        "tools.pipeline",
        "tools.web_search",
        "tools.os_exec",
        "tools.fs_access",
        "tools.io_paths",
        "validation_utils",
    ]
    imported_ok, errors = [], []
    for m in mods:
        try:
            importlib.import_module(m)
            imported_ok.append(m)
        except Exception as e:
            errors.append(f"{m}: {e}")

    # 2) Workspace R/W
    ws_report = []
    ws_root = "(unknown)"
    try:
        from tools.file_manager import mkdir, write_file, read_file, remove_path, list_dir, workspace_path
        mkdir("_doctor", exist_ok=True)
        ws_report.append("mkdir OK")
        write_file("_doctor/ping.txt", "pong")
        ws_report.append("write_file OK")
        content = read_file("_doctor/ping.txt").strip()
        ws_report.append(f"read_file OK ({content})")
        listing = list_dir("_doctor")
        ws_report.append("list_dir OK" if "ping.txt" in listing else "list_dir (?)")
        remove_path("_doctor", recursive=True)
        ws_report.append("remove_path OK")
        ws_root = workspace_path()
    except Exception as e:
        ws_report.append(f"workspace ops error: {e}")

    api_warn = "" if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY not set (LLM calls may fail)."

    return (
        "=== Ion Doctor ===\n"
        f"Project root: {os.path.abspath(os.getcwd())}\n"
        f"Workspace:    {ws_root}\n\n"
        "Imports OK:\n  - " + (", ".join(imported_ok) if imported_ok else "(none)") + "\n\n"
        "Import errors:\n  - " + (", ".join(errors) if errors else "(none)") + "\n\n"
        "Workspace ops:\n  - " + ("; ".join(ws_report)) + "\n\n"
        "Env:\n  - " + (api_warn or "OPENAI_API_KEY present")
    )


def run_once(agent, session_id: str, user_input: str) -> str:
    """Invoke the agent once and return the assistant's final text output."""
    if _RICH and console is not None:
        with console.status("[accent]Analyzing...[/accent]", spinner="earth"):
            resp = agent.invoke(
                {"input": user_input},
                {"configurable": {"session_id": session_id}},
            )
    else:
        resp = agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": session_id}},
        )
    # RunnableWithMessageHistory typically returns {'output': '...'}
    if isinstance(resp, dict) and "output" in resp:
        return str(resp["output"])
    return str(resp)


def run_repl(initial_session: str) -> None:
    agent = create_agent()
    session_id = initial_session or "default"

    # Basic environment sanity
    if not os.getenv("OPENAI_API_KEY"):
        if _RICH and console is not None:
            console.print(Panel.fit(Text(
                "Warning: OPENAI_API_KEY not set. The assistant may not be able to call the LLM.",
                style="warn"
            ), border_style="border", box=box.ROUNDED))
        else:
            print("Warning: OPENAI_API_KEY not set. The assistant may not be able to call the LLM.\n")

    _banner(session_id)

    while True:
        try:
            user = read_user_input(session_id)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Ion Chronos assistant.")
            break

        if not user:
            continue

        low = user.lower()

        # --- Built-in REPL commands ---
        if low in {"exit", "quit"}:
            print("Goodbye!")
            break

        if low == "help":
            render_reply(HELP)
            continue

        if low == "doctor":
            try:
                report = _doctor()
                render_reply(f"```\n{report}\n```")
            except Exception as e:
                render_reply(f"Doctor failed: {e}")
            continue

        if low.startswith("validate "):
            ticker = low.replace("validate ", "").strip().upper()
            if ticker:
                try:
                    from validation_utils import quick_validate
                    result = quick_validate(ticker)
                    if result.get("execution_verified"):
                        render_reply(f"✅ **{ticker} Validation Passed**\n\n{result['summary']}")
                    else:
                        render_reply(f"❌ **{ticker} Validation Failed**\n\n{result['execution_details']['reason']}")
                except Exception as e:
                    render_reply(f"Validation error: {e}")
            else:
                render_reply("Usage: validate <TICKER>")
            continue

        if low == "list validated":
            try:
                from validation_utils import list_validated_tickers
                tickers = list_validated_tickers()
                if tickers:
                    render_reply(f"✅ **Validated Tickers**: {', '.join(tickers)}")
                else:
                    render_reply("❌ No validated tickers found")
            except Exception as e:
                render_reply(f"Error listing validated tickers: {e}")
            continue

        if low == "new session":
            session_id = str(uuid4())
            msg = f"[info] Started a new session: {session_id}"
            if _RICH and console is not None:
                console.print(Panel.fit(Text(msg, style="accent"), border_style="border", box=box.ROUNDED))
            else:
                print(msg)
            continue

        if low.startswith("use session "):
            parts = user.split(" ", 2)  # allow spaces in session name
            if len(parts) < 3 or not parts[2].strip():
                render_reply("Usage: `use session <name>`")
                continue
            session_id = parts[2].strip()
            msg = f"[info] Switched to session: {session_id}"
            if _RICH and console is not None:
                console.print(Panel.fit(Text(msg, style="accent"), border_style="border", box=box.ROUNDED))
            else:
                print(msg)
            continue

        # --- Normal agent turn ---
        try:
            reply = run_once(agent, session_id, user)
            render_reply(reply)
        except Exception as e:
            err = format_error(e)
            if _RICH and console is not None:
                console.print(Panel.fit(Text(f"Error during processing: {err}", style="err"), border_style="red", box=box.ROUNDED))
            else:
                print(f"Error during processing: {err}")


def main():
    args = parse_args()

    # Non-interactive single-shot mode (useful for scripting)
    if args.noninteractive:
        agent = create_agent()
        try:
            out = run_once(agent, args.session or "default", args.noninteractive)
            print(out)
        except Exception as e:
            print(f"Error during processing: {format_error(e)}")
        return

    # Interactive REPL
    run_repl(args.session or "default")


if __name__ == "__main__":
    main()
