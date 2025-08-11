# FILENAME: agent.py
import os
import sys
import ollama
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# --- Configuration ---
MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini-128k")
USER_STORY = "As a user, I want to log in with my email and password so I can access my account."
SYSTEM_PROMPT = """
You are an elite QA Engineer and Business Analyst from a top-tier consulting firm.
Your task is to take a high-level user story and generate a comprehensive BDD test plan in Gherkin format.
The plan must be clear enough for a CEO to understand and detailed enough for a developer to implement.
CRITICALLY, you must identify not just the "happy path" but also negative paths and security-related edge cases.
Structure your output as a single Markdown block.
"""

# --- Rich Console Initialization ---
console = Console()

def ensure_model_available(model_name: str):
    """
    Checks if the model is available locally in Ollama. If not, it pulls the model.
    """
    try:
        local_models = [m["name"] for m in ollama.list()["models"]]
        if model_name in local_models:
            console.print(f"[green]Model '{model_name}' is available locally.[/green]")
            return

        console.print(f"[yellow]Model '{model_name}' not found. Pulling from Ollama Hub...[/yellow]")
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                transient=True,
        ) as progress:
            pull_task = progress.add_task(f"[cyan]Downloading {model_name}", total=100)
            current_digest = ""
            for status in ollama.pull(model_name, stream=True):
                if status.get("digest"):
                    current_digest = status["digest"]
                if "total" in status and "completed" in status:
                    progress.update(
                        pull_task,
                        completed=status["completed"],
                        total=status["total"],
                    )
        console.print(f"[green]Successfully pulled model '{model_name}'. Digest: {current_digest}[/green]")

    except Exception as e:
        console.print(f"[bold red]Failed to pull model '{model_name}'.[/bold red]")
        console.print(f"Error details: {e}")
        sys.exit(1)


# Tenacity will retry on network-related issues, which is its strength.
@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(3),
    retry=retry_if_exception_type(ollama.ResponseError),
    reraise=True,
)
def generate_test_plan(prompt: str, model: str):
    """
    Connects to the Ollama service to generate the test plan, with retries on failure.
    """
    with console.status("[bold green]Agent is thinking... Analyzing requirements and edge cases...", spinner="dots"):
        response = ollama.generate(
            model=model, prompt=prompt, system=SYSTEM_PROMPT, stream=False
        )
        return response["response"]

def main():
    """Main execution function."""
    try:
        console.rule("[bold blue]AI QA Agent Initializing[/]", style="blue")
        console.print(f"\n[bold]Objective Received:[/] [italic]'{USER_STORY}'[/]")
        console.print(f"[bold]Model Specified:[/]  [italic]{MODEL}[/]\n")

        # Step 1: Ensure the language model is ready.
        ensure_model_available(MODEL)

        # Step 2: Generate the test plan.
        console.rule("[bold blue]Generating Test Plan[/]", style="blue")
        generated_plan = generate_test_plan(USER_STORY, MODEL)

        # Step 3: Display the result.
        console.rule("[bold green]Mission Complete[/]", style="green")
        console.print(Markdown(generated_plan))

    except ollama.ResponseError as e:
        console.print(f"[bold red]Error during generation: Could not connect to the Ollama server.[/bold red]")
        if e.error:
            console.print(f"[red]Details: {e.error}[/red]")
    except Exception as e:
        console.print(f"[bold red]An unexpected error occurred:[/bold red]\n{e}")

if __name__ == "__main__":
    main()