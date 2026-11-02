"""GUI adapter for deterministic preview tests without opening a window."""

import cv2


class PreviewWindowStub:
    def __init__(self, title):
        self.title = title

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def show(self, frame):
        cv2.imshow(self.title, frame)

    def poll_key(self):
        return cv2.waitKey(1) & 0xFF

    def is_open(self):
        return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
