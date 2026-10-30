"""Test head movement with the local Reachy Mini simulator running."""

import time

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


def main():
    # Camera and audio are unnecessary for this movement-only test.
    with ReachyMini(
        connection_mode="localhost_only",
        media_backend="no_media",
    ) as mini:
        for yaw in (0, 20, -20, 0):
            print(f"Moving head to yaw {yaw} degrees...")
            mini.goto_target(
                head=create_head_pose(yaw=yaw, degrees=True),
                duration=1.0,
            )
            time.sleep(0.5)
    print("Head movement test finished.")


if __name__ == "__main__":
    main()
