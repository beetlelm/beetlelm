"""
Miscellaneous logging utilities.
"""
# src/training/utils/logging.py
import logging
import os

from io import StringIO

import yaml
from lightning.fabric.utilities.rank_zero import rank_zero_only
from rich.console import Console
from rich.panel import Panel


@rank_zero_only
def pretty_print_yaml_config(logger, config: dict) -> None:
    """
    Pretty print config with rich formatting. Assumes that the config is already saved as a
    dictionary - this can be done by calling `asdict` on the dataclass or loading in the config
    from a yaml file.

    NOTE: this function is only called on rank 0.

    Args:
        logger: Logger object to log the formatted output to.
        config: Dictionary containing the config to pretty print.
    """
    # Create string buffer
    output = StringIO()
    console = Console(file=output, force_terminal=False)

    # Convert to YAML string first
    yaml_str = yaml.dump(
        config, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper
    )

    # Create formatted panel
    panel = Panel(
        yaml_str,
        border_style="blue",
        padding=(0, 1),  # Reduced padding
        expand=False,  # Don't expand to terminal width
    )

    # Print to buffer
    console.print(panel)

    # Log the formatted output
    for line in output.getvalue().splitlines():
        logger.info(line)



def initialize_logging(log_dir=None, log_level=logging.INFO):
    """
    Initialize Python logging for training.
    Optionally saves logs to a directory.
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Remove any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info("Logging initialized")
    return logger
