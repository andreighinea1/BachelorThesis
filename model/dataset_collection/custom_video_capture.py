import cv2


class CustomVideoCapture(cv2.VideoCapture):
    """ A custom `cv2.VideoCapture` that calls self.release() automatically for sure. """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
