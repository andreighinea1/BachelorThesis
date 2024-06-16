import os
import random
import shutil
import subprocess
import sys
import time

import cv2
import moviepy.editor as mp
import numpy as np
import screeninfo
import ujson

from dataset_collection.audio_player import AudioPlayer
from dataset_collection.custom_video_capture import CustomVideoCapture

PAUSE_KEY = ord(" ")
QUIT_KEY = ord("q")
SKIP_KEY = ord("s")  # Used to skip messages only


# TODO: Show the chosen number when scoring, and don't close the scoring screen, keep it open for the whole duration.


class EEGEmotionExperiment:
    EXPERIMENT_WINDOW_NAME = "EEG Emotion Recognition Experiment"
    # BASE_PATH = "./dataset_collection"
    BASE_PATH = "."
    VIDEOS_DIR_PATH = f"{BASE_PATH}/videos"
    SCORES_FILE_PATH = f"{BASE_PATH}/output/scores.json"

    EMOTION_COLORS = {
        "positive": (120, 200, 80),  # Emerald Green in BGR
        "neutral": (128, 128, 128),  # Gray
        "negative": (43, 43, 210),  # Cadmium Red in BGR
    }
    PAUSED_COLOR = (0, 0, 255)  # Red color in BGR
    PAUSED_TEXT = "PAUSED"

    VIDEO_FORMAT = "mp4"
    AUDIO_FORMAT = "wav"

    VERDICTS_DICT = {
        "positive": 1,
        "neutral": 0,
        "negative": -1,
    }
    scoring_dict = {
        ord(str(i)): i
        for i in range(6)  # Scores from 0 to 5 inclusive
    }

    def __init__(
            self,
            desired_segment_duration=2 * 60,  # 2 minutes
            segment_eps=0.9,  # Accept segments of length at least `segment_duration * segment_eps`
            overwrite_concatenated_videos=False,
            ignore_existing_files=False,
            msg_font_scale=2, msg_font_thickness=4,
            save_audio_separately=True,
    ):
        if not (0 <= segment_eps <= 1):
            raise Exception("segment_eps must be between 0 and 1")

        self.desired_segment_duration = desired_segment_duration
        self.segment_eps = segment_eps
        self.overwrite_concatenated_videos = overwrite_concatenated_videos
        self.ignore_existing_files = ignore_existing_files
        self.msg_font_scale = msg_font_scale
        self.msg_font_thickness = msg_font_thickness
        self.save_audio_separately = save_audio_separately

        self.scores = dict()

        self.video_dirs = {
            emotion: emotion_path
            for emotion in self.VERDICTS_DICT.keys()
            if (emotion_path := os.path.join(self.VIDEOS_DIR_PATH, emotion))
            if os.path.exists(emotion_path) and os.listdir(emotion_path)
        }
        self.concatenated_videos = {
            emotion: os.path.join(emotion_path, f"_concatenated.{self.VIDEO_FORMAT}")
            for emotion, emotion_path in self.video_dirs.items()
        }
        self.segmented_dirs = {
            emotion: os.path.join(emotion_path, "_segments")
            for emotion, emotion_path in self.video_dirs.items()
        }
        for path in self.segmented_dirs.values():
            os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.dirname(self.SCORES_FILE_PATH), exist_ok=True)

        screen = screeninfo.get_monitors()[0]
        self.screen_size_x, self.screen_size_y = screen.width, screen.height

    def _pause_experiment(self, img):
        cv2.putText(
            img, self.PAUSED_TEXT, (self.screen_size_x - 200, self.screen_size_y - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, self.PAUSED_COLOR, 2, cv2.LINE_AA
        )
        cv2.imshow(self.EXPERIMENT_WINDOW_NAME, img)

        # Listen for keys
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == PAUSE_KEY:
                return
            elif key == QUIT_KEY:
                cv2.destroyAllWindows()
                sys.exit()

    # 1
    @staticmethod
    def _concatenate_videos(video_files, output_path):
        # If only one video file, simply copy it to the output path
        if len(video_files) == 1:
            shutil.copyfile(video_files[0], output_path)
            return

        final_clip = mp.concatenate_videoclips(
            clips=[mp.VideoFileClip(file) for file in video_files],
            method="compose",  # "chain"
        )
        final_clip.write_videofile(output_path, codec="libx264")
        return

    def prepare_videos(self):
        for emotion, dir_path in self.video_dirs.items():
            concatenated_path = self.concatenated_videos[emotion]
            if os.path.isfile(concatenated_path) and not self.overwrite_concatenated_videos:
                str_msg = f"{emotion} video already concatenated"
                if self.ignore_existing_files:
                    print(f"{str_msg}, ignoring existing videos")
                    continue
                raise Exception(f"{str_msg}, and did not allow overwriting it")

            video_files = [
                os.path.join(dir_path, file_name)
                for file_name in os.listdir(dir_path)
                if file_name.endswith(f".{self.VIDEO_FORMAT}") and "_concatenated" not in file_name
            ]
            self._concatenate_videos(video_files, concatenated_path)
        return

    # 2
    def create_segments(self):
        for emotion in self.video_dirs.keys():
            concatenated_path = self.concatenated_videos[emotion]
            segment_base_path = self.segmented_dirs[emotion]

            if os.listdir(segment_base_path):
                if self.ignore_existing_files:
                    print(f"Segmenting for {emotion} canceled, as segments already exist in {segment_base_path}")
                    continue
                raise Exception(f"To proceed, remove existing segments from {segment_base_path}")

            duration = mp.VideoFileClip(concatenated_path).duration
            for i in range(0, int(duration), self.desired_segment_duration):
                segment_end = min(i + self.desired_segment_duration, duration)
                actual_segment_duration = segment_end - i

                # Take only segments at least close in size with the desired segment duration
                if actual_segment_duration < self.desired_segment_duration * self.segment_eps:
                    break

                segment_name = f"segment_{i // self.desired_segment_duration}"

                # Save the video segment using ffmpeg
                video_segment_output_path = os.path.join(segment_base_path, f"{segment_name}.{self.VIDEO_FORMAT}")
                ffmpeg_command = [
                    "ffmpeg",
                    "-i", concatenated_path,
                    "-ss", str(i),
                    "-to", str(segment_end),
                    "-c", "copy",
                    video_segment_output_path
                ]
                if self.save_audio_separately:
                    ffmpeg_command.insert(-1, "-an")  # Disable audio in the output video
                subprocess.run(ffmpeg_command, check=True)

                if self.save_audio_separately:
                    # Save the corresponding audio segment using ffmpeg
                    audio_segment_output_path = os.path.join(segment_base_path, f"{segment_name}.{self.AUDIO_FORMAT}")
                    ffmpeg_audio_command = [
                        "ffmpeg",
                        "-i", concatenated_path,
                        "-ss", str(i),
                        "-to", str(segment_end),
                        "-c", "copy",
                        "-vn",  # No video
                        audio_segment_output_path
                    ]
                    subprocess.run(ffmpeg_audio_command, check=True)
        return

    # 3
    def _save_scores(self):
        with open(self.SCORES_FILE_PATH, "w", encoding="utf-8") as f:
            ujson.dump(
                self.scores, f,
                escape_forward_slashes=False, indent=2, sort_keys=True, ensure_ascii=False
            )
        return

    def _play_video(self, video_path, audio_path):
        with CustomVideoCapture(video_path) as cap, AudioPlayer(audio_path) as audio_player:
            paused = False
            start_time = time.time()
            pause_start_time = None

            def re_sync_audio():
                current_video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                audio_player.seek(current_video_time)

            # printed_desync_time = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                video_position = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                audio_position = audio_player.get_position()
                desync = abs(audio_position - video_position)
                # if time.time() - printed_desync_time > 1:
                #     print(f"desync: {desync:.4f}")
                #     printed_desync_time = time.time()
                if desync > 0.25:  # More than 250ms desync
                    print(f"Video desynchronized with audio by {desync:.4f}! Resynchronizing...", file=sys.stderr)
                    re_sync_audio()

                if paused:
                    if not pause_start_time:
                        pause_start_time = time.time()
                    audio_player.pause()
                    self._pause_experiment(img=frame)
                    paused = False
                    start_time += time.time() - pause_start_time  # Adjust `start_time` to account for paused duration
                    pause_start_time = None

                    # Resynchronize audio to the current position of the video
                    re_sync_audio()
                else:
                    cv2.imshow(self.EXPERIMENT_WINDOW_NAME, frame)
                    audio_player.resume()

                    # Find the amount needed to sleep in order to synchronize the playing video to the current time
                    elapsed = int((time.time() - start_time) * 1000)  # ms
                    play_time = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    sleep = max(1, play_time - elapsed)

                    key = cv2.waitKey(sleep) & 0xFF
                    if key == PAUSE_KEY:
                        paused = True
                    elif key == QUIT_KEY:
                        cv2.destroyAllWindows()
                        sys.exit()
                    elif key == SKIP_KEY:  # TODO: Make more complex system to skip videos to not skip them by mistake
                        break
        return

    def _show_message(self, message, duration, *, emotion=None, segment_path=None, emotion_msg_scale=2.75):
        # Prepare the main message
        text_size = cv2.getTextSize(
            message, cv2.FONT_HERSHEY_SIMPLEX, self.msg_font_scale, self.msg_font_thickness
        )[0]
        text_x = (self.screen_size_x - text_size[0]) // 2
        text_y = (self.screen_size_y + text_size[1]) // 2

        # Prepare the emotion msg
        # Place emotion msg below the main message
        emotion_font_scale = self.msg_font_scale * emotion_msg_scale
        emotion_font_thickness = int(self.msg_font_thickness * emotion_msg_scale)
        emotion_message = emotion.capitalize() if emotion in self.VERDICTS_DICT else ""
        emotion_color = self.EMOTION_COLORS.get(emotion, (0, 0, 0))
        emotion_text_size = cv2.getTextSize(
            emotion_message, cv2.FONT_HERSHEY_SIMPLEX, emotion_font_scale, emotion_font_thickness
        )[0]
        emotion_text_x = (self.screen_size_x - emotion_text_size[0]) // 2
        emotion_text_y = text_y + text_size[1] + emotion_text_size[1]

        # Center the messages
        if emotion_message:
            text_y -= emotion_text_size[1] // 2
            emotion_text_y -= emotion_text_size[1] // 2

        elapsed_time = 0
        last_paused = True
        last_time = None
        while elapsed_time < duration:
            # Increase the `elapsed_time` like a delta-time only after the first loop. Discard time while paused.
            current_time = time.time()
            if last_paused:
                last_paused = False
            else:
                elapsed_time += current_time - last_time
            last_time = current_time

            # Show the messages
            img = 255 * np.ones((self.screen_size_y, self.screen_size_x, 3), dtype=np.uint8)
            cv2.putText(
                img, message, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, self.msg_font_scale, (15, 15, 15), self.msg_font_thickness, cv2.LINE_AA
            )
            if emotion_message:
                cv2.putText(
                    img, emotion_message, (emotion_text_x, emotion_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, emotion_font_scale, emotion_color, emotion_font_thickness, cv2.LINE_AA
                )
            cv2.imshow(self.EXPERIMENT_WINDOW_NAME, img)

            # Listen for keys
            key = cv2.waitKey(1) & 0xFF
            if key == PAUSE_KEY:
                last_paused = True
                self._pause_experiment(img=img)
            elif key == QUIT_KEY:
                cv2.destroyAllWindows()
                sys.exit()
            elif key == SKIP_KEY:
                return
            elif segment_path and key != 255:
                if key in self.scoring_dict:
                    # Save this score for this video in a dictionary
                    self.scores[segment_path.replace("\\", "/")] = self.scoring_dict[key]
                    self._save_scores()
                    return

        if segment_path:
            # TODO: Make this into a message on screen to not close the experiment for this
            raise Exception("We are scoring, but no score was provided")

        return

    def run_experiment(self):
        # Get the list of segmented videos
        segment_files = {
            emotion: sorted([
                os.path.join(segmented_dir, file)
                for file in os.listdir(segmented_dir)
                if file.endswith(f".{self.VIDEO_FORMAT}")
            ])
            for emotion, segmented_dir in self.segmented_dirs.items()
        }

        # Generate the dynamic order of segments
        emotions = list(segment_files.keys())
        assert set(emotions) == {"positive", "negative", "neutral"}
        random.shuffle(emotions)

        order = []
        while all(segment_files.values()):
            for emotion in emotions:
                order.append((emotion, segment_files[emotion].pop(0)))

        # Define the experiment protocol
        protocol = [("Hint of start", 5), ("Movie clip", None), ("Self-scoring", 45), ("Rest", 15)]

        # --- Start of opencv stuff ---
        cv2.namedWindow(self.EXPERIMENT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.EXPERIMENT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Give 1 minute to prepare for the start of the whole experiment
        self._show_message("Prepare for experiment", 60)

        # Main experiment loop
        for emotion, segment_video_path in order:
            # TODO:
            #  Check overall time in experiment until now.
            #  If > 50 minutes, pause the experiment automatically and wait for manual start again.
            for step, duration in protocol:
                if step == "Hint of start":
                    self._show_message(
                        "Starting with emotion:", duration,
                        emotion=emotion,
                    )
                elif step == "Movie clip":
                    start_video_time = time.time()
                    self._play_video(
                        video_path=segment_video_path,
                        audio_path=segment_video_path.replace(
                            f".{self.VIDEO_FORMAT}",
                            f".{self.AUDIO_FORMAT}"
                        ),
                    )
                    time_taken = time.time() - start_video_time
                    if time_taken < self.desired_segment_duration:
                        print(f"Probably skipped the video. Video time taken: {time_taken:.4f}", file=sys.stderr)
                    else:
                        print(f"Video time taken: {time_taken:.4f}")
                elif step == "Self-scoring":
                    self._show_message(
                        "Please score your emotions", duration,
                        segment_path=segment_video_path,  # TODO: Don't use `segment_video_path` directly here
                    )
                elif step == "Rest":
                    self._show_message(
                        "Rest", duration,
                    )

        cv2.destroyAllWindows()
        return
