"""
RunHook for injecting complete task definitions into containers for ask_user tool.

This hook writes the full task definition (including underspecified version and removed
segments) to a file in the container so the ask_user tool can access it.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.run.hooks.abstract import RunHook

logger = logging.getLogger(__name__)


class TaskDefinitionInjectionHook(RunHook):
    """
    Inject complete task definitions into container for ask_user tool.

    Writes task definition to /tmp/task_definition.json in the container, which
    the ask_user tool reads to provide accurate clarifications.
    """

    def __init__(
        self,
        task_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
        task_definitions_file: Optional[Path] = None,
    ):
        """
        Initialize hook with task definitions.

        Args:
            task_definitions: Dict mapping instance_id to task definition, OR
            task_definitions_file: Path to JSON file with task definitions

        Task definition should contain:
            - primary_task: Complete task description
            - underspecified_task: Partial task given to agent
            - removed_segments: List of removed segments
            - expected_questions: Expected clarification questions
        """
        super().__init__()
        self.task_definitions = task_definitions
        self.task_definitions_file = task_definitions_file

    def _load_task_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load task definitions from file or return cached dict."""
        if self.task_definitions is not None:
            return self.task_definitions

        if self.task_definitions_file and self.task_definitions_file.exists():
            with open(self.task_definitions_file) as f:
                return json.load(f)

        return {}

    def on_instance_start(
        self,
        *,
        index: int,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
    ) -> None:
        """
        Inject task definition into container before agent starts.

        Called after environment is ready but before agent.setup().
        """
        instance_id = problem_statement.id

        # Load task definitions
        task_definitions = self._load_task_definitions()

        # Check if we have a task definition for this instance
        if instance_id not in task_definitions:
            logger.debug(f"No task definition found for instance {instance_id}, skipping injection")
            return

        task_def = task_definitions[instance_id]

        # Write task definition to container
        task_def_path = "/tmp/task_definition.json"
        task_def_json = json.dumps(task_def, indent=2)

        try:
            # Write file to container using swerex
            logger.info(f"Injecting task definition for {instance_id} to {task_def_path}")

            # Create a temporary file write command
            command = f"cat > {task_def_path} << 'TASK_DEFINITION_EOF'\n{task_def_json}\nTASK_DEFINITION_EOF"
            env.communicate(command, check="raise")

            # Set environment variables for the ask_user tool
            env_vars = {
                "TASK_DEFINITION_PATH": task_def_path,
                "HAS_TASK_DEFINITION": "true",
            }

            # Pass through API credentials for the user simulator LLM
            api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
            simulator_model = os.environ.get("USER_SIMULATOR_MODEL")

            if api_key:
                env_vars["OPENAI_API_KEY"] = api_key
                env_vars["LLM_API_KEY"] = api_key
            if base_url:
                env_vars["OPENAI_BASE_URL"] = base_url
                env_vars["LLM_BASE_URL"] = base_url
            if simulator_model:
                env_vars["USER_SIMULATOR_MODEL"] = simulator_model

            env.set_env_variables(env_vars)

            logger.info(f"Successfully injected task definition for {instance_id}")
        except Exception as e:
            logger.error(f"Failed to inject task definition for {instance_id}: {e}")
            # Don't raise - let the run continue even if injection fails
