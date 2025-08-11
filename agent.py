import os
import sys
import ollama
import structlog
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
log = structlog.get_logger()

MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
USER_STORY = "As a user, I want to log in with my email and password so I can access my account."
SYSTEM_PROMPT = """
You are an elite QA Engineer. Your task is to generate a complete BDD feature file in Gherkin syntax based on a user story.

The output MUST be a single Markdown code block containing the Gherkin text.
It MUST include scenarios for the "happy path," "negative paths," and critical "security/edge cases."
Use Gherkin keywords (`Feature`, `Scenario`, `Given`, `When`, `Then`, `And`) correctly.
Tag scenarios with appropriate labels like `@happy-path`, `@negative-path`, `@security`.

Here is an example of the required format to guide the AI (this is a "one-shot" learning example):

```gherkin
Feature: User Authentication
  As a user, I want to log in with my credentials to access my account securely.

  @happy-path
  Scenario: Successful login with valid credentials
    Given the user is on the login page
    And the user enters a valid email and password
    When the user clicks the "Login" button
    Then the user should be redirected to their account dashboard
    And a success message "Welcome back!" should be displayed

  @negative-path
  Scenario: Unsuccessful login with an invalid password
    Given the user is on the login page
    And the user enters a valid email
    And the user enters an invalid password
    When the user clicks the "Login" button
    Then the user should remain on the login page
    And an error message "Invalid credentials. Please try again." should be displayed

  @security @edge-case
  Scenario: Attempted login with a locked account
    Given a user's account has been locked due to multiple failed login attempts
    And the user is on the login page
    When the user enters the correct credentials for the locked account
    Then an error message "Your account is locked. Please contact support." should be displayed
```
"""

class QAAgent:
    def __init__(self, model: str, user_story: str):
        self.model = model
        self.user_story = user_story
        self.log = structlog.get_logger(agent=self.__class__.__name__)

    def run(self):
        try:
            self.log.info("=" * 20 + " AI QA Agent Initializing " + "=" * 20)
            self.log.info("Objective Received", story=self.user_story)
            self.log.info("Model Specified", model=self.model)

            self._ensure_model_available()

            self.log.info("=" * 20 + " Generating Test Plan " + "=" * 20)
            generated_plan = self._generate_test_plan()

            self.log.info("=" * 20 + " Mission Complete " + "=" * 20)
            print(generated_plan)

        except ollama.ResponseError as e:
            self.log.error(
                "Could not connect to the Ollama server.",
                error=e.error,
                status_code=e.status_code
            )
            sys.exit(1)
        except Exception:
            self.log.exception("An unexpected error occurred.")
            sys.exit(1)

    def _ensure_model_available(self):
        try:
            local_models = [m["name"] for m in ollama.list()["models"]]
            if self.model in local_models:
                self.log.info("Model is available locally.", model=self.model)
                return

            self.log.warning("Model not found locally. Pulling from Ollama Hub...", model=self.model)

            current_digest = ""
            for status in ollama.pull(self.model, stream=True):
                if status.get("digest"):
                    current_digest = status["digest"]

            self.log.info("Successfully pulled model.", model=self.model, digest=current_digest)

        except Exception:
            self.log.exception("Failed to pull model.", model=self.model)
            raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(ollama.ResponseError),
        reraise=True,
        before_sleep=lambda retry_state: log.warning(
            "Retrying API call...",
            attempt=retry_state.attempt_number,
            wait=retry_state.next_action.sleep,
        ),
    )
    def _generate_test_plan(self) -> str:
        self.log.info("Agent is thinking... Crafting a professional-grade test plan...")
        response = ollama.generate(
            model=self.model,
            prompt=self.user_story,
            system=SYSTEM_PROMPT,
            stream=False,
        )
        self.log.info("Agent has finished generating the plan.")

        response_text = response.get("response", "")
        if "```gherkin" in response_text:
            return response_text.split("```gherkin")[1].split("```")[0].strip()
        elif "```" in response_text:
            return response_text.split("```")[1].strip()
        else:
            return response_text

def main():
    agent = QAAgent(model=MODEL, user_story=USER_STORY)
    agent.run()

if __name__ == "__main__":
    main()