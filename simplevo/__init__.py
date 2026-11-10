"""SimplEvo: direct control by an MLLM with a task memory and a working memory it writes itself."""
from .agent import EpisodeRecord, adapt, deploy, run_episode
from .annotate import annotate
from .chain import AdaptationChain, validate_fixed
from .demonstration import Demonstration
from .task import Embodiment, Environment, Outcome, TaskSpec
from .task_memory import TaskMemory

__version__ = "1.0.0"
__all__ = ["AdaptationChain", "Demonstration", "Embodiment", "Environment", "EpisodeRecord", "Outcome", "TaskMemory",
           "TaskSpec", "adapt", "annotate", "deploy", "run_episode", "validate_fixed"]
