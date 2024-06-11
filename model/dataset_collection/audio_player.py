import threading
import wave
from typing import Mapping

import pyaudio


class AudioPlayer:
    """ A class for playing audio files. """

    def __init__(self, audio_file_path):
        """
        Initializes the AudioPlayer with the given audio file.

        Args:
            audio_file_path (str): Path to the audio file to be played.

        Raises:
            Exception: If the audio file cannot be opened.
        """
        self._paused = False
        self._stop = False
        self._lock = threading.Lock()

        try:
            self._wf = wave.open(audio_file_path, "rb")
        except IOError:
            raise Exception(f"Could not open audio file: {audio_file_path}")

        self._p = pyaudio.PyAudio()
        self._stream = self._p.open(
            format=self._p.get_format_from_width(self._wf.getsampwidth()),
            channels=self._wf.getnchannels(),
            rate=self._wf.getframerate(),
            output=True,
            stream_callback=self._stream_callback,
        )

    def _stream_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for audio stream.

        Args:
            in_data (bytes | None): Input data.
            frame_count (int): Number of frames.
            time_info (Mapping[str, float]): Time information.
            status (int): Status of the stream.

        Returns:
            tuple[bytes, int]: Audio data and continue flag.
        """
        with self._lock:
            if self._paused or self._stop:
                return b"\x00" * frame_count * self._wf.getsampwidth(), pyaudio.paContinue

            data = self._wf.readframes(frame_count)
            if len(data) == 0:
                return None, pyaudio.paComplete
            return data, pyaudio.paContinue

    def close(self):
        """
        Closes the audio stream and releases resources.
        """
        with self._lock:
            self._stop = True
        self._stream.stop_stream()
        self._stream.close()
        self._p.terminate()
        self._wf.close()

    def pause(self):
        """
        Pauses the audio playback.
        """
        with self._lock:
            self._paused = True

    def resume(self):
        """
        Resumes the audio playback.
        """
        with self._lock:
            self._paused = False

    def get_status(self):
        """
        Gets the current status of the audio playback.

        Returns:
            dict: Dictionary containing paused state, current position, and duration.
        """
        with self._lock:
            return {
                "paused": self._paused,
                "position": self._wf.tell() / self._wf.getframerate(),
                "duration": self._wf.getnframes() / self._wf.getframerate()
            }

    def seek(self, position):
        """
        Seeks to a specific position in the audio file.

        Args:
            position (float): Position in seconds to seek to.
        """
        with self._lock:
            frame = int(position * self._wf.getframerate())
            self._wf.setpos(frame)

    def __enter__(self):
        """
        Enters the runtime context related to this object.

        Returns:
            AudioPlayer: The current instance of the AudioPlayer.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exits the runtime context related to this object and closes resources.

        Args:
            exc_type: The exception type.
            exc_value: The exception value.
            traceback: The traceback object.
        """
        self.close()
