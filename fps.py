# importing necessary packages
import datetime

class FPS:
    """
    Manages the recording of the number of frames per second, as the application is running.
    """
    def __init__(self):
        """
        Store the start and end times, as well as the total number of frames
        examined between those times.
        """
        self._start = None
        self._end = None
        self._numFrames = 0
    
    def start(self):
        """
        Starting the timer
        """
        self._start = datetime.datetime.now()
        return self
    
    def stop(self):
        """
        Stopping the timer
        """
        self._end = datetime.datetime.now()

    def update(self):
        """
        Increment the total number of frames examined during the start and end
        interval
        """
        self._numFrames += 1
    
    def elapsed(self):
        """
        Returns:
            The total number of seconds between the start end end interval
        """
        return (self._end - self._start).total_seconds()

    def end_to_end_fps(self):
        """
        Compute the approximate FPS
        """
        return self._numFrames / self.elapsed()

    def current_frame_number(self):
        """
        Returns:
            The number of frames processed thus far. 
        """
        return self._numFrames

