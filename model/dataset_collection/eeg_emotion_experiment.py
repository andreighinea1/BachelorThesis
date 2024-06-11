import os
import random
import sys
import time

import cv2
import moviepy.editor as mp
import numpy as np
import ujson

PAUSE_KEY = ord(" ")
QUIT_KEY = ord("q")
SKIP_KEY = ord("s")  # Used to skip messages only


def _pause_experiment():
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == PAUSE_KEY:
            return
        elif key == QUIT_KEY:
            cv2.destroyAllWindows()
            sys.exit()


class EEGEmotionExperiment:
    EXPERIMENT_WINDOW_NAME = "EEG Emotion Recognition Experiment"
    BASE_PATH = "./dataset_collection"
    VIDEOS_DIR_PATH = f"{BASE_PATH}/videos"
    SCORES_FILE_PATH = f"{BASE_PATH}/scores.json"

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
            desired_segment_duration=4 * 60,  # 4 minutes
            segment_eps=0.9,  # Accept segments of length at least `segment_duration * segment_eps`
            overwrite_concatenated_videos=False,
            ignore_existing_files=False,
    ):
        if not (0 <= segment_eps <= 1):
            raise Exception("segment_eps must be between 0 and 1")

        self.desired_segment_duration = desired_segment_duration
        self.segment_eps = segment_eps
        self.overwrite_concatenated_videos = overwrite_concatenated_videos
        self.ignore_existing_files = ignore_existing_files

        self.scores = dict()

        self.video_dirs = {
            emotion: emotion_path
            for emotion in self.VERDICTS_DICT.keys()
            if (emotion_path := os.path.join(self.VIDEOS_DIR_PATH, emotion))
            if os.path.exists(emotion_path) and os.listdir(emotion_path)
        }
        self.concatenated_videos = {
            emotion: os.path.join(emotion_path, f"_concatenated.mp4")
            for emotion, emotion_path in self.video_dirs.items()
        }
        self.segmented_dirs = {
            emotion: os.path.join(emotion_path, "_segments")
            for emotion, emotion_path in self.video_dirs.items()
        }
        for path in self.segmented_dirs.values():
            os.makedirs(path, exist_ok=True)

    # 1
    @staticmethod
    def _concatenate_videos(video_files, output_path):
        clips = [mp.VideoFileClip(file) for file in video_files]
        final_clip = mp.concatenate_videoclips(clips)
        final_clip.write_videofile(output_path, codec="libx264")

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
                if file_name.endswith(".mp4") and "_concatenated" not in file_name
            ]
            self._concatenate_videos(video_files, concatenated_path)

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

            video = mp.VideoFileClip(concatenated_path)
            duration = video.duration
            for i in range(0, int(duration), self.desired_segment_duration):
                segment_end = min(i + self.desired_segment_duration, duration)
                actual_segment_duration = segment_end - i

                # Take only segments at least close in size with the desired segment duration
                if actual_segment_duration < self.desired_segment_duration * self.segment_eps:
                    break

                segment = video.subclip(i, segment_end)
                segment_file_name = f"segment_{i // self.desired_segment_duration}.mp4"
                segment_output_file_path = os.path.join(segment_base_path, segment_file_name)
                segment.write_videofile(segment_output_file_path, codec="libx264")

    # 3
    def _save_scores(self):
        with open(self.SCORES_FILE_PATH, "w") as f:
            ujson.dump(
                self.scores, f,
                escape_forward_slashes=False, indent=2, sort_keys=True, ensure_ascii=False
            )

    def _play_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(self.EXPERIMENT_WINDOW_NAME, frame)
            key = cv2.waitKey(25) & 0xFF
            if key == PAUSE_KEY:
                _pause_experiment()
            elif key == QUIT_KEY:
                cv2.destroyAllWindows()
                sys.exit()
            elif key == SKIP_KEY:  # TODO: Make more complex system to skip videos to not skip them by mistake
                return
        cap.release()

    def _show_message(
            self, message, duration, *,
            font_scale=2, font_thickness=3, screen_size_x=1920, screen_size_y=1080,
            scoring=False, segment_path=None,
    ):
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = (screen_size_x - text_size[0]) // 2
        text_y = (screen_size_y + text_size[1]) // 2

        start_time = time.time()
        while time.time() - start_time < duration:
            img = 255 * np.ones((screen_size_y, screen_size_x, 3), dtype=np.uint8)
            cv2.putText(
                img, message, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA
            )
            cv2.imshow(self.EXPERIMENT_WINDOW_NAME, img)
            key = cv2.waitKey(1) & 0xFF
            if key == PAUSE_KEY:
                _pause_experiment()
            elif key == QUIT_KEY:
                cv2.destroyAllWindows()
                sys.exit()
            elif key == SKIP_KEY:
                return
            elif scoring and key != 255:
                if key in self.scoring_dict:
                    # Save this score for this video in a dictionary
                    self.scores[segment_path] = self.scoring_dict[key]
                    self._save_scores()
                    return

    def run_experiment(self):
        cv2.namedWindow(self.EXPERIMENT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.EXPERIMENT_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Get the list of segmented videos
        segment_files = {
            emotion: sorted([
                os.path.join(segmented_dir, file)
                for file in os.listdir(segmented_dir)
                if file.endswith(".mp4")
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
                order.append(segment_files[emotion].pop(0))

        # Define the experiment protocol
        protocol = [("Hint of start", 5), ("Movie clip", None), ("Self-scoring", 45), ("Rest", 15)]

        # Give 1 minute to prepare for the start of the whole experiment
        self._show_message("Prepare for experiment", 60)

        # Main experiment loop
        for segment_path in order:
            for step, duration in protocol:
                if step == "Hint of start":
                    self._show_message("Hint of start", duration)
                elif step == "Movie clip":
                    self._play_video(segment_path)
                elif step == "Self-scoring":
                    self._show_message(
                        "Please score your emotions", duration,
                        scoring=True, segment_path=segment_path,
                    )
                elif step == "Rest":
                    self._show_message("Rest", duration)

        cv2.destroyAllWindows()
        return
