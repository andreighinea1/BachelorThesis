import os
import time

import moviepy.editor as mp
import pygame


class EEGEmotionExperiment:
    VIDEOS_PATH = "./dataset_collection/videos"
    VERDICTS_DICT = {
        "positive": 1,
        "neutral": 0,
        "negative": -1,
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

        self.video_dirs = {
            emotion: emotion_path
            for emotion in self.VERDICTS_DICT.keys()
            if (emotion_path := os.path.join(self.VIDEOS_PATH, emotion))
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
                raise Exception(f"{str_msg}, and didn't allow overwriting it")

            video_files = [
                os.path.join(dir_path, file_name)
                for file_name in os.listdir(dir_path)
                if file_name.endswith('.mp4') and "_concatenated" not in file_name
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

    def run_experiment(self):
        def play_video(video_path):
            clip = mp.VideoFileClip(video_path)
            clip.preview(fullscreen=True)

        # Get the list of segmented videos
        segment_files = {
            emotion: sorted([
                os.path.join(segmented_dir, file)
                for file in os.listdir(segmented_dir)
                if file.endswith('.mp4')
            ])
            for emotion, segmented_dir in self.segmented_dirs.items()
        }

        # Generate the dynamic order of segments
        order = []
        while all(segment_files.values()):
            for emotion in ["positive", "negative", "neutral"]:
                order.append(segment_files[emotion].pop(0))
        print("order:", order)
        import sys
        sys.exit()

        # Define the experiment protocol
        protocol = [("Hint of start", 5), ("Movie clip", 240), ("Self-assessment", 45), ("Rest", 15)]

        # Start pygame - or..?
        # TODO: Maybe don't use pygame?
        #  Or idk, but it doesn't play the video at all.
        #  And it shouldn't fix the video playing to 240s,
        #  it should just play the video for however much time it is
        #  (the video's already segmented after all)
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption('EEG Emotion Recognition Experiment')

        def show_message(message, duration):
            font = pygame.font.Font(None, 74)
            text = font.render(message, True, (255, 255, 255))
            screen.fill((0, 0, 0))
            screen.blit(
                text,
                (
                    (screen.get_width() - text.get_width()) // 2,
                    (screen.get_height() - text.get_height()) // 2
                )
            )
            pygame.display.flip()
            time.sleep(duration)

        # Main experiment loop
        for segment_path in order:
            for step, duration in protocol:
                if step == "Hint of start":
                    show_message("Hint of start", duration)
                elif step == "Movie clip":
                    play_video(segment_path)
                elif step == "Self-assessment":
                    show_message("Please assess your emotions", duration)
                elif step == "Rest":
                    show_message("Rest", duration)

        # Clean up
        pygame.quit()
