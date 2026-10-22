import time


class FPSTimer:
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.calls += 1
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        if average:
            return self.total_time / self.calls
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.


import time

class Timer:
    def __init__(self, duration):
        self.duration = duration
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def is_time_up(self):
        if self.start_time is None:
            return False
        diff_time = time.time() - self.start_time
        if (diff_time) < self.duration:
            return False
        return True
        
    def get_remaining_time(self):
        """返回剩餘的倒數時間"""
        if self.start_time is None:
            return self.duration
        elapsed_time = time.time() - self.start_time
        remaining_time = self.duration - elapsed_time
        return max(0, int(remaining_time)) 
    
    def reset(self):
        self.start_time = None

