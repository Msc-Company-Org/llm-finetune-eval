"""llm-finetune-eval — benchmark a fine-tuned model against a frontier baseline."""

from .config import TaskConfig, load_task
from .runner import run_eval

__version__ = "0.1.0"
__all__ = ["TaskConfig", "load_task", "run_eval", "__version__"]
