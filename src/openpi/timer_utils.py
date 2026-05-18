# timer_utils.py
import time
import functools
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    """Generic timer usable as a context manager or decorator."""
    
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        if self.verbose:
            #print(f"[TIMER] {self.name}: {self.elapsed:.4f} seconds")
            pass
        return False
    
    def reset(self):
        self.elapsed = 0
        self.start_time = None

def timed_function(name=None):
    """Function timing decorator."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            with Timer(timer_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator