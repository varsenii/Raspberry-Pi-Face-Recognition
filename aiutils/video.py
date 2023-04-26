from datetime import datetime
import cv2
from threading import Thread

class FPS:
    """A class to measure the frames per second of a video stream.

    Attributes:
        start_time: A datetime object that represents the start time of the measurement.
        end_time: A datetime object that represents the end time of the measurement.
        num_frames: An integer that counts the number of frames processed.
    """

    def __init__(self):
        """Initialize the attributes with None or zero values."""
        self.start_time = None
        self.end_time = None
        self.num_frames = 0
    
    def start(self):
        """Start the timer and return the instance of the class."""
        self.start_time = datetime.now()
        return self
    
    def stop(self):
        """Stop the timer and set the end time attribute."""
        self.end_time = datetime.now()

    def update(self):
        """Increment the number of frames attribute by one."""
        self.num_frames += 1
        self.end_time = datetime.now()
    
    def elapsed(self):
        """Return the elapsed time in seconds between the start and end times.

        Returns:
            A float that represents the elapsed time in seconds.
        """
        return (self.end_time - self.start_time).total_seconds()
    
    def fps(self):
        """Return the average frames per second.

        Returns:
            A float that represents the average frames per second.
        """
        if self.elapsed() > 0.0:
            return self.num_frames / self.elapsed()
        return self.num_frames


class WebcamVideoStream:
    """A class to capture video frames from a webcam using a separate thread.

    Attributes:
        stream: A cv2.VideoCapture object that represents the video source.
        grabbed: A boolean that indicates if the frame was successfully grabbed.
        frame: A numpy array that contains the current frame.
        stopped: A boolean that indicates if the stream is stopped.
    """

    def __init__(self, src=0):
        """Initialize the video capture object with the source.

        Args:
            src: An integer or a string that specifies the video source. Default is 0 (the first webcam).
        """
        self.stream = cv2.VideoCapture(src)
        # Grab the first frame and store it as an attribute (strictly for initialization puposes)
        (self.grabbed, self.frame) = self.stream.read()
        # Initialize a flag to indicate if the stream is stopped
        self.stopped = False
    
    def start(self):
        """Start a thread to update the frames in the background.

        Returns:
            The instance of the class.
        """
        # Start a thread to update the frames in the background
        Thread(target=self.update, args=()).start()
        # Return the instance of the class
        return self

    def update(self):
        """Loop until the stream is stopped and grab the next frame."""
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        """Return the current frame.

        Returns:
            A numpy array that contains the current frame.
        """
        return self.frame
    
    def stop(self):
        """Set the flag to stop the stream."""
        self.stopped = True