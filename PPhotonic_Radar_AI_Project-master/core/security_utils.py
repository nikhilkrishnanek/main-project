import os

def safe_path(path):
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    base = os.path.abspath("results")
    
    # If path is just a filename, join it with results
    if not os.path.isabs(path) and "results" not in path:
        full = os.path.abspath(os.path.join("results", path))
    else:
        full = os.path.abspath(path)
        
    if not full.startswith(base):
        # Allow reading from project root if it's explicitly allowed files, 
        # but for now let's stick to results for safety as intended.
        raise PermissionError(f"Unauthorized access to {full}")
    return full
