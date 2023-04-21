from datetime import datetime

class FPS:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.num_frames = 0
    
    def start(self):
        self.start_time = datetime.now()
    
    def stop(self):
        self.end_time = datetime.now()

    def update(self):
        self.num_frames += 1
    
    def elapsed(self):
        return (self.end_time - self.start_time).total_seconds()
    
    def fps(self):
        return self.num_frames / self.elapsed()