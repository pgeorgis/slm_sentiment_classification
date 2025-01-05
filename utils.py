"""General auxiliary functions."""

import datetime
import subprocess

def get_git_commit_hash():
    """Get latest git commit hash of current repository."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', "--short", 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError as e:
        return None

def create_timestamp():
    """Create time stamp with current date and time."""
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime
