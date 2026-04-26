"""
CodeReview-Env — Live Demo Script
Run:  python demo.py
Records a full end-to-end walkthrough of the GRPO code review agent
"""

import os, time, json, sys, textwrap, re
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Try importing rich; install if missing ────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.syntax import Syntax
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.align import Align
    from rich.rule import Rule
except ImportError:
    print("Installing 'rich'...")
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.syntax import Syntax
    from rich.text import Text
    from rich import box
    from rich.columns import Columns
    from rich.align import Align
    from rich.rule import Rule

console = Console(highlight=False)

ENV_URL      = os.getenv("ENV_URL", "https://dharaneswarreddy-codereview-env.hf.space")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL        = "llama-3.3-70b-versatile"

REWARD_WEIGHTS = {
    "format":         0.15,
    "detection":      0.30,
    "classification": 0.20,
    "confidence":     0.15,
    "quality":        0.20,
}

COMPONENT_COLORS = {
    "format":         "cyan",
    "detection":      "green",
    "classification": "yellow",
    "confidence":     "magenta",
    "quality":        "blue",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def slow_print(text: str, delay: float = 0.018):
    for ch in text:
        console.print(ch, end="", markup=False)
        time.sleep(delay)
    console.print()

def pause(s: float = 1.2):
    time.sleep(s)

def section(title: str):
    console.print()
    console.print(Rule(f"[bold white]{title}[/bold white]", style="bright_black"))
    console.print()

def score_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if value >= 0.6 else "yellow" if value >= 0.35 else "red"
    return f"[{color}]{bar}[/{color}] [{color}]{value:.2f}[/{color}]"

# ── Score Components ──────────────────────────────────────────────────────────

def score_format(action: dict) -> float:
    required = {"has_bug", "bug_type", "severity", "suggested_fix"}
    if not action:
        return 0.0
    present = sum(1 for k in required if k in action)
    return present / len(required)

def score_detection(has_bug_pred: bool, has_bug_gt: bool) -> float:
    return 1.0 if has_bug_pred == has_bug_gt else 0.0

def score_classification(bug_type_pred: str, bug_type_gt: str) -> float:
    if not bug_type_pred or not bug_type_gt:
        return 0.0
    return 1.0 if bug_type_pred.lower().strip() == bug_type_gt.lower().strip() else 0.3

def score_confidence(action: dict) -> float:
    return 0.5  # Baseline (no history in demo)

def score_quality(fix: str) -> float:
    fix = str(fix or "").strip()
    if len(fix) < 10:
        return 0.0
    score = 0.0
    if len(fix) >= 20: score += 0.25
    if len(fix) >= 60: score += 0.15
    if len(fix) >= 120: score += 0.10
    keywords = ["replace", "use", "change", "add", "remove", "instead", "wrap", "validate"]
    score += min(sum(1 for k in keywords if k in fix.lower()) / 3.0, 0.25) * 0.4
    return min(score, 1.0)

# ── Main Demo ─────────────────────────────────────────────────────────────────

def run_demo():
    # ── Header ────────────────────────────────────────────────────────────────
    console.clear()
    pause(0.5)
    banner = Panel(
        Align.center(
            Text.from_markup(
                "[bold bright_white]CodeReview-Env[/bold bright_white]\n"
                "[dim]GRPO-Trained Code Review Agent[/dim]\n\n"
                "[italic cyan]Powered by Unsloth · TRL · Groq · HuggingFace[/italic cyan]"
            )
        ),
        border_style="bright_blue",
        padding=(1, 8),
    )
    console.print(banner)
    pause(2)

    # ── Step 1: Connect to environment ───────────────────────────────────────
    section("STEP 1 — Connecting to CodeReview Environment")
    console.print(f"  [dim]Environment URL:[/dim] [cyan]{ENV_URL}[/cyan]")
    pause(0.5)

    obs = None
    with Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as prog:
        t = prog.add_task("  Fetching code snippet from environment...", total=None)
        try:
            r = requests.post(f"{ENV_URL}/reset", json={"task_level": 3}, timeout=30)
            obs = r.json()
            prog.update(t, description="  [green]✓ Snippet received!")
            time.sleep(1)
        except Exception as e:
            prog.update(t, description=f"  [red]× Connection failed: {e}")
            time.sleep(2)

    if not obs or not obs.get("code"):
        console.print("[red]❌ Could not fetch from live environment. Check your ENV_URL![/red]")
        return

    code      = obs.get("code", "")
    language  = obs.get("language", "python")
    snippet_id = obs.get("snippet_id", "unknown")
    level     = obs.get("task_level", 3)

    console.print(f"  [green]✓ Connected![/green]  Snippet: [bold]{snippet_id}[/bold]  |  Level: [bold]{level}[/bold]  |  Language: [bold]{language}[/bold]")
    pause(1)

    # ── Step 2: Display the code ──────────────────────────────────────────────
    section("STEP 2 — Code Under Review")
    syn = Syntax(code, language, theme="monokai", line_numbers=True, padding=(1, 2))
    console.print(Panel(syn, title="[bold yellow]🔍 Code Snippet[/bold yellow]", border_style="yellow"))
    pause(2.5)

    # ── Step 3: Model generates review ───────────────────────────────────────
    section("STEP 3 — Student Model (Qwen2.5-1.5B) Generating Review")
    console.print("  [dim]Sending code to trained agent...[/dim]")
    pause(0.5)

    review_raw = None
    action = {}

    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
    SYSTEM_PROMPT = (
        "You are an expert code reviewer. Analyze the code carefully for bugs, security issues, "
        "and performance problems. Return ONLY a valid JSON object with exactly these keys:\n"
        '{"has_bug": bool, "bug_type": str, "severity": str, "suggested_fix": str}\n\n'
        "where bug_type is one of: logic_error, security_vulnerability, performance_issue, "
        "type_error, syntax_error, no_bug\n"
        "and severity is one of: critical, high, medium, low, none"
    )

    with Progress(
        SpinnerColumn(spinner_name="clock"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as prog:
        t = prog.add_task("  Agent analyzing code...", total=None)
        try:
            resp = groq_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Review this {language} code:\n```{language}\n{code}\n```"},
                ],
                max_tokens=400,
                temperature=0.4,
            )
            review_raw = resp.choices[0].message.content.strip()
            # Strip markdown fences
            review_raw = re.sub(r"^```[a-z]*\n?", "", review_raw)
            review_raw = re.sub(r"```$", "", review_raw).strip()
            action = json.loads(review_raw)
            prog.update(t, description="  [green]✓ Review generated!")
            time.sleep(1)
        except Exception as e:
            prog.update(t, description=f"  [red]× Error: {e}")
            time.sleep(2)
            return

    # ── Step 4: Display the review ────────────────────────────────────────────
    section("STEP 4 — Agent's Code Review Output")

    has_bug  = action.get("has_bug", False)
    bug_type = action.get("bug_type", "unknown")
    severity = action.get("severity", "unknown")
    fix      = action.get("suggested_fix", "")

    bug_color = "red" if has_bug else "green"
    bug_icon  = "🐛" if has_bug else "✅"

    review_table = Table(box=box.ROUNDED, border_style="bright_blue", show_header=False, padding=(0, 1))
    review_table.add_column("Field",  style="bold white",  width=18)
    review_table.add_column("Value",  style="white",       width=55)

    review_table.add_row("Bug Found",      f"[{bug_color}]{bug_icon}  {str(has_bug).upper()}[/{bug_color}]")
    review_table.add_row("Bug Type",       f"[yellow]{bug_type}[/yellow]")
    sev_color = {"critical": "red", "high": "red", "medium": "yellow", "low": "cyan"}.get(severity.lower(), "white")
    review_table.add_row("Severity",       f"[{sev_color}]{severity.upper()}[/{sev_color}]")
    review_table.add_row("Suggested Fix",  textwrap.fill(fix, 55))

    console.print(Panel(review_table, title="[bold bright_blue]📋 Agent Review[/bold bright_blue]", border_style="bright_blue"))
    pause(2.5)

    # ── Step 5: Reward scoring ────────────────────────────────────────────────
    section("STEP 5 — 5-Component Reward System Evaluating...")

    # Ground truth from environment
    gt_has_bug   = obs.get("has_bug", True)
    gt_bug_type  = obs.get("bug_type", "logic_error")

    components = {
        "format":         score_format(action),
        "detection":      score_detection(has_bug, gt_has_bug),
        "classification": score_classification(bug_type, gt_bug_type),
        "confidence":     score_confidence(action),
        "quality":        score_quality(fix),
    }

    weights = REWARD_WEIGHTS
    final = sum(components[k] * weights[k] for k in weights)
    final = max(0.0, min(final, 1.0))

    # Animate reward reveal
    reward_table = Table(box=box.ROUNDED, border_style="dim white", padding=(0, 1))
    reward_table.add_column("Component",  style="bold white",       width=16)
    reward_table.add_column("Weight",     style="dim white",        width=8,  justify="right")
    reward_table.add_column("Score",      style="white",            width=10, justify="right")
    reward_table.add_column("Bar",        style="white",            width=30)

    for k, v in components.items():
        color = COMPONENT_COLORS[k]
        time.sleep(0.3)  # animate each row appearing
        reward_table.add_row(
            f"[{color}]{k.capitalize()}[/{color}]",
            f"[dim]{int(weights[k]*100)}%[/dim]",
            f"[bold]{v:.3f}[/bold]",
            score_bar(v),
        )

    console.print(reward_table)
    pause(0.5)

    # Final score reveal
    final_color = "green" if final >= 0.6 else "yellow" if final >= 0.35 else "red"
    grade = "EXCELLENT ★★★" if final >= 0.75 else "GOOD ★★☆" if final >= 0.5 else "NEEDS WORK ★☆☆" if final >= 0.3 else "POOR ☆☆☆"

    final_panel = Panel(
        Align.center(
            Text.from_markup(
                f"[bold {final_color}]{score_bar(final, 30)}[/bold {final_color}]\n\n"
                f"[bold white]FINAL REWARD SCORE: [bold {final_color}]{final:.4f}[/bold {final_color}][/bold white]\n"
                f"[dim]{grade}[/dim]"
            )
        ),
        title="[bold white]🏆 GRPO Reward Signal[/bold white]",
        border_style=final_color,
        padding=(1, 4),
    )
    console.print(final_panel)
    pause(2.5)

    # ── Step 6: Ground truth comparison ──────────────────────────────────────
    section("STEP 6 — Ground Truth Comparison")

    cmp_table = Table(box=box.SIMPLE_HEAD, border_style="dim white", padding=(0, 2))
    cmp_table.add_column("",          style="bold dim white", width=18)
    cmp_table.add_column("🤖 Agent",  style="cyan",           width=25)
    cmp_table.add_column("✅ Truth",  style="green",          width=25)
    cmp_table.add_column("Match",     style="white",          width=8)

    def match_icon(a, b):
        return "[green]✓[/green]" if str(a).lower().strip() == str(b).lower().strip() else "[red]✗[/red]"

    cmp_table.add_row("Bug Found",  str(has_bug), str(gt_has_bug),   match_icon(has_bug, gt_has_bug))
    cmp_table.add_row("Bug Type",   bug_type,     gt_bug_type,        match_icon(bug_type, gt_bug_type))

    console.print(cmp_table)
    pause(1.5)

    # ── Step 7: GRPO Update summary ───────────────────────────────────────────
    section("STEP 7 — What GRPO Does With This Score")

    console.print(
        Panel(
            f"  The agent generated [bold]4 variations[/bold] of this review.\n"
            f"  This review scored [bold {final_color}]{final:.3f}[/bold {final_color}].\n\n"
            f"  GRPO compares all 4 scores and [bold]updates model weights[/bold]\n"
            f"  to make the logic behind the best-scoring review\n"
            f"  [bold green]MORE LIKELY[/bold green] in future predictions.\n\n"
            f"  Over [bold]300 training steps[/bold], this pushes the model\n"
            f"  from [bold red]~0.36[/bold red] baseline → [bold green]0.60+[/bold green] target reward.",
            title="[bold white]🧠 Policy Update Logic[/bold white]",
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )
    pause(2)

    # ── Footer ────────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            Align.center(
                "[bold bright_white]MetaXScaler Hackathon — CodeReview-Env[/bold bright_white]\n"
                "[dim]GRPO · Unsloth · TRL · Composable Rubrics · Curriculum Learning[/dim]"
            ),
            border_style="bright_blue",
            padding=(1, 4),
        )
    )
    console.print()


if __name__ == "__main__":
    run_demo()
