from pathlib import Path
from .utils import RecursiveNamespace

DATA_PATH = Path(__file__).parent.parent / "data"
USE_MODIN = False

from dagster import ExperimentalWarning
import warnings

warnings.filterwarnings("ignore", category=ExperimentalWarning)
