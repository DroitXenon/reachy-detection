"""GPU-backed camera preview with nonblocking event processing.

GLFW and OpenGL are loaded only when the window context starts. All window and
OpenGL operations belong to the main thread; pacing belongs to the application.
"""

from collections import deque
from threading import get_ident, main_thread

import numpy as np


def _load_gui():
    import glfw
    from OpenGL import GL
    return glfw, GL


class PreviewWindow:
    """Upload BGR frames to a reusable texture and draw an aspect-fitted quad."""

    def __init__(self, title):
        self.title = title
        self._keys = deque(maxlen=32)
        self._glfw = self._gl = self._window = self._texture = None
        self._owner = None
        self._started = self._open = self._initialized = self._closing = False
        self._source_size = self._texture_size = None

    def __enter__(self):
        if self._started:
            raise RuntimeError('A preview window instance can only be started once')
        if get_ident() != main_thread().ident:
            raise RuntimeError('The preview window must run on the main thread')
        self._started, self._owner = True, get_ident()
        try:
            self._glfw, self._gl = _load_gui()
            glfw, gl = self._glfw, self._gl
            if not glfw.init():
                raise RuntimeError('GLFW could not initialize a graphical display')
            self._initialized = True
            self._available = self._available_size()
            glfw.default_window_hints()
            # macOS supports legacy OpenGL 2.1 without the core-profile flags
            # that would remove immediate-mode quads.
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.FALSE)
            self._window = glfw.create_window(min(960, self._available[0]),
                                              min(720, self._available[1]),
                                              self.title, None, None)
            if not self._window:
                raise RuntimeError('GLFW could not create an OpenGL preview window')
            glfw.make_context_current(self._window)
            # The app owns its 60 Hz deadline. Vsync here would add a second wait.
            glfw.swap_interval(0)
            glfw.set_key_callback(self._window, self._on_key)
            glfw.set_window_close_callback(self._window, self._on_close)
            self._texture = int(gl.glGenTextures(1))
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
            for parameter in (gl.GL_TEXTURE_MIN_FILTER, gl.GL_TEXTURE_MAG_FILTER):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, gl.GL_LINEAR)
            for parameter in (gl.GL_TEXTURE_WRAP_S, gl.GL_TEXTURE_WRAP_T):
                gl.glTexParameteri(gl.GL_TEXTURE_2D, parameter, gl.GL_CLAMP_TO_EDGE)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            gl.glPixelStorei(gl.GL_UNPACK_ROW_LENGTH, 0)
            gl.glClearColor(.09, .09, .09, 1.0)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            self._open = True
        except Exception as error:
            self.close()
            raise RuntimeError(f'Cannot open preview window: {error}') from error
        return self

    def _available_size(self):
        glfw = self._glfw
        monitor = glfw.get_primary_monitor()
        width, height = 1280, 800
        if monitor:
            mode = glfw.get_video_mode(monitor)
            if mode is not None:
                width, height = mode.size.width, mode.size.height
            # GLFW workarea accounts for the menu/Dock and logical display
            # coordinates on Retina screens. Older GLFW versions can omit it.
            workarea = getattr(glfw, 'get_monitor_workarea', None)
            if callable(workarea):
                area = workarea(monitor)
                if area is not None and area[2] > 0 and area[3] > 0:
                    width, height = area[2], area[3]
        return max(100, width - 80), max(100, height - 100)

    def _check_owner(self):
        if self._owner is not None and self._owner != get_ident():
            raise RuntimeError('Preview window operations must run on its main thread')

    def _on_close(self, window):
        self._closing = True

    def _on_key(self, window, key, scancode, action, modifiers):
        glfw = self._glfw
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_ESCAPE:
            self._keys.append(27)
        elif glfw.KEY_A <= key <= glfw.KEY_Z:
            self._keys.append(ord(chr(key).lower()))
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            plus = key == glfw.KEY_KP_ADD or modifiers & glfw.MOD_SHIFT
            self._keys.append(ord('+' if plus else '='))
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self._keys.append(ord('_' if modifiers & glfw.MOD_SHIFT else '-'))

    def show(self, image):
        """Upload a uint8 BGR image without CPU color conversion or resizing."""
        self._check_owner()
        if not self._open or self._window is None:
            raise RuntimeError('Preview window is closed')
        if (not isinstance(image, np.ndarray) or image.ndim != 3
                or image.shape[2] != 3 or image.dtype != np.uint8
                or image.shape[0] == 0 or image.shape[1] == 0):
            raise ValueError('Preview needs a nonempty uint8 BGR image')
        height, width = image.shape[:2]
        size = width, height
        glfw, gl = self._glfw, self._gl
        glfw.make_context_current(self._window)
        if size != self._source_size:
            scale = min(1.0, self._available[0] / width, self._available[1] / height)
            glfw.set_window_size(self._window, max(1, round(width * scale)),
                                 max(1, round(height * scale)))
            self._source_size = size
        pixels = image if image.flags.c_contiguous else np.ascontiguousarray(image)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        if self._texture_size != size:
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
                            gl.GL_BGR, gl.GL_UNSIGNED_BYTE, pixels)
            self._texture_size = size
        else:
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, width, height,
                               gl.GL_BGR, gl.GL_UNSIGNED_BYTE, pixels)
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self._window)
        if framebuffer_width <= 0 or framebuffer_height <= 0:
            return  # A minimized window has no drawable framebuffer.
        gl.glViewport(0, 0, framebuffer_width, framebuffer_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        scale = min(framebuffer_width / width, framebuffer_height / height)
        draw_width, draw_height = max(1, round(width * scale)), max(1, round(height * scale))
        gl.glViewport((framebuffer_width - draw_width) // 2,
                      (framebuffer_height - draw_height) // 2, draw_width, draw_height)
        gl.glColor4f(1.0, 1.0, 1.0, 1.0)
        gl.glBegin(gl.GL_QUADS)
        # Row zero is the top of a NumPy image. Flip texture coordinates rather
        # than copying/flipping the full frame on the CPU.
        for texture, vertex in (((0, 1), (-1, -1)), ((1, 1), (1, -1)),
                                ((1, 0), (1, 1)), ((0, 0), (-1, 1))):
            gl.glTexCoord2f(*texture)
            gl.glVertex2f(*vertex)
        gl.glEnd()
        glfw.swap_buffers(self._window)

    def poll_key(self):
        """Pump pending input immediately; return one key or -1."""
        self._check_owner()
        if not self._open:
            return -1
        self._glfw.poll_events()
        return self._keys.popleft() if self._keys else -1

    def is_open(self):
        return bool(self._open and not self._closing and self._window is not None
                    and not self._glfw.window_should_close(self._window))

    def close(self):
        self._check_owner()
        window, texture = self._window, self._texture
        self._window = self._texture = None
        self._open = False
        self._keys.clear()
        try:
            if window is not None:
                try:
                    self._glfw.make_context_current(window)
                    if texture is not None:
                        self._gl.glDeleteTextures(1, [texture])
                except Exception:
                    pass  # Cleanup still destroys the window after context loss.
                finally:
                    self._glfw.destroy_window(window)
        finally:
            if self._initialized:
                self._initialized = False
                self._glfw.terminate()

    def __exit__(self, *exc):
        self.close()
