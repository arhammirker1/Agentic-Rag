import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)

class TrailLogger:
    """
    Utility for high-visibility logging of AI 'trails' (thoughts, inputs, outputs).
    """
    def __init__(self, log_dir="pageindex_data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.trail_file = os.path.join(log_dir, "trail.log")

    def step(self, title: str, content: str = "", data: any = None, quiet: bool = False):
        """Log a specific step in the pipeline."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        header = f"\n{'='*20} {title} ({timestamp}) {'='*20}"
        
        # Format for console (only if not quiet)
        if not quiet:
            print(f"\n[trail] >> {title}")
        
        # Always write to file
        with open(self.trail_file, "a", encoding="utf-8") as f:
            f.write(header + "\n")
            if content:
                f.write(f"INFO: {content}\n")
            if data:
                import json
                if isinstance(data, (dict, list)):
                    f.write("DATA:\n" + json.dumps(data, indent=2) + "\n")
                else:
                    f.write(f"DATA: {data}\n")
            f.write("-" * len(header) + "\n")

    def decision(self, agent: str, thought: str, action: str):
        """Log an AI decision/thought process."""
        self.step(f"DECISION: {agent}", f"THOUGHT: {thought}\nACTION: {action}")

trail = TrailLogger()
