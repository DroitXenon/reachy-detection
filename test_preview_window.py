"""GPU-preview lifecycle and uploads without opening a native window."""

from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from preview_window import PreviewWindow


def fake_glfw():
    values = dict(CONTEXT_VERSION_MAJOR=1, CONTEXT_VERSION_MINOR=2,
                  OPENGL_FORWARD_COMPAT=3, FALSE=0, PRESS=1, REPEAT=2,
                  KEY_ESCAPE=256, KEY_A=65, KEY_Z=90, KEY_EQUAL=61,
                  KEY_KP_ADD=334, KEY_MINUS=45, KEY_KP_SUBTRACT=333, MOD_SHIFT=1)
    functions = ('init', 'get_primary_monitor', 'get_video_mode', 'get_monitor_workarea',
                 'default_window_hints', 'window_hint', 'create_window',
                 'make_context_current', 'swap_interval', 'set_key_callback',
                 'set_window_close_callback', 'set_window_size', 'get_framebuffer_size',
                 'swap_buffers', 'poll_events', 'window_should_close', 'destroy_window',
                 'terminate')
    values.update({name: Mock() for name in functions})
    glfw = SimpleNamespace(**values)
    glfw.init.return_value = True
    glfw.get_primary_monitor.return_value = 'monitor'
    glfw.get_video_mode.return_value = SimpleNamespace(size=SimpleNamespace(width=1280, height=900))
    glfw.get_monitor_workarea.return_value = (0, 0, 1280, 900)
    glfw.create_window.return_value = 'window'
    glfw.get_framebuffer_size.return_value = (960, 720)
    glfw.window_should_close.return_value = False
    return glfw


def fake_gl():
    constants = ('GL_TEXTURE_2D', 'GL_TEXTURE_MIN_FILTER', 'GL_TEXTURE_MAG_FILTER',
                 'GL_LINEAR', 'GL_TEXTURE_WRAP_S', 'GL_TEXTURE_WRAP_T',
                 'GL_CLAMP_TO_EDGE', 'GL_UNPACK_ALIGNMENT', 'GL_UNPACK_ROW_LENGTH',
                 'GL_DEPTH_TEST', 'GL_PROJECTION', 'GL_MODELVIEW', 'GL_RGB', 'GL_BGR',
                 'GL_UNSIGNED_BYTE', 'GL_COLOR_BUFFER_BIT', 'GL_QUADS')
    functions = ('glGenTextures', 'glBindTexture', 'glTexParameteri', 'glPixelStorei',
                 'glClearColor', 'glDisable', 'glEnable', 'glMatrixMode',
                 'glLoadIdentity', 'glTexImage2D', 'glTexSubImage2D', 'glViewport',
                 'glClear', 'glColor4f', 'glBegin', 'glTexCoord2f', 'glVertex2f',
                 'glEnd', 'glDeleteTextures')
    values = {name: index for index, name in enumerate(constants, 1)}
    values.update({name: Mock() for name in functions})
    gl = SimpleNamespace(**values)
    gl.glGenTextures.return_value = 7
    return gl


class PreviewWindowTests(unittest.TestCase):
    def gui(self):
        glfw, gl = fake_glfw(), fake_gl()
        loader = self.enterContext(patch('preview_window._load_gui', return_value=(glfw, gl)))
        return glfw, gl, loader

    def test_import_and_construction_do_not_initialize_gui(self):
        with patch('preview_window._load_gui') as loader:
            window = PreviewWindow('test')
            self.assertFalse(window.is_open())
            window.close()
        loader.assert_not_called()

    def test_reuses_texture_and_uploads_native_bgr_without_modifying_pixels(self):
        glfw, gl, _ = self.gui()
        pixels = np.full((12, 20, 3), [3, 2, 1], np.uint8)
        original = pixels.copy()
        with PreviewWindow('Reachy test') as window:
            self.assertTrue(window.is_open())
            glfw.swap_interval.assert_called_once_with(0)
            window.show(pixels)
            first = gl.glTexImage2D.call_args.args
            self.assertEqual(first[3:5], (20, 12))
            self.assertEqual(first[6:8], (gl.GL_BGR, gl.GL_UNSIGNED_BYTE))
            self.assertIs(first[8], pixels)
            window.show(pixels)
            gl.glTexImage2D.assert_called_once()
            gl.glTexSubImage2D.assert_called_once()
            second = gl.glTexSubImage2D.call_args.args
            self.assertEqual(second[6:8], (gl.GL_BGR, gl.GL_UNSIGNED_BYTE))
            self.assertIs(second[8], pixels)
            self.assertEqual(gl.glTexCoord2f.call_args_list[:4],
                             [call(0, 1), call(1, 1), call(1, 0), call(0, 0)])
            self.assertEqual(gl.glVertex2f.call_args_list[:4],
                             [call(-1, -1), call(1, -1), call(1, 1), call(-1, 1)])
            self.assertEqual(window.poll_key(), -1)
            glfw.poll_events.assert_called_once_with()
        glfw.destroy_window.assert_called_once_with('window')
        glfw.terminate.assert_called_once_with()
        gl.glDeleteTextures.assert_called_once_with(1, [7])
        np.testing.assert_array_equal(pixels, original)

    def test_window_fits_screen_and_gpu_letterboxes_without_cpu_resize(self):
        glfw, gl, _ = self.gui()
        glfw.get_monitor_workarea.return_value = (0, 0, 1000, 800)
        glfw.get_framebuffer_size.return_value = (1000, 1000)
        with PreviewWindow('test') as window:
            window.show(np.zeros((1000, 2000, 3), np.uint8))
            glfw.set_window_size.assert_called_once_with('window', 920, 460)
            self.assertEqual(gl.glTexImage2D.call_args.args[3:5], (2000, 1000))
            self.assertEqual(gl.glViewport.call_args.args, (0, 250, 1000, 500))
            window.show(np.zeros((100, 100, 3), np.uint8))
            self.assertEqual(gl.glTexImage2D.call_count, 2)
            self.assertEqual(gl.glViewport.call_args.args, (0, 0, 1000, 1000))

    def test_strided_images_are_contiguous_and_minimized_window_does_not_swap(self):
        glfw, gl, _ = self.gui()
        pixels = np.full((12, 20, 3), [3, 2, 1], np.uint8)[:, ::-1]
        glfw.get_framebuffer_size.return_value = (0, 0)
        with PreviewWindow('test') as window:
            window.show(pixels)
            upload = gl.glTexImage2D.call_args.args[8]
            self.assertTrue(upload.flags.c_contiguous)
            np.testing.assert_array_equal(upload, pixels)
            glfw.swap_buffers.assert_not_called()

    def test_keys_are_bounded_normalized_and_escape_is_supported(self):
        glfw, _, _ = self.gui()
        with PreviewWindow('test') as window:
            handler = glfw.set_key_callback.call_args.args[1]
            for key in (ord('C'), ord('R'), ord('Q'), ord('M'), ord('D')):
                handler('window', key, 0, glfw.PRESS, 0)
            handler('window', glfw.KEY_EQUAL, 0, glfw.PRESS, glfw.MOD_SHIFT)
            handler('window', glfw.KEY_MINUS, 0, glfw.REPEAT, 0)
            handler('window', glfw.KEY_EQUAL, 0, glfw.PRESS, 0)
            handler('window', glfw.KEY_ESCAPE, 0, glfw.PRESS, 0)
            self.assertEqual([window.poll_key() for _ in range(9)],
                             [ord(character) for character in 'crqmd+-='] + [27])
            handler('window', ord('C'), 0, 0, 0)  # Releases do not repeat controls.
            self.assertEqual(window.poll_key(), -1)
            for number in range(50):
                handler('window', ord('A') + number % 26, 0, glfw.PRESS, 0)
            self.assertEqual([window.poll_key() for _ in range(32)],
                             [ord('a') + number % 26 for number in range(18, 50)])
            self.assertEqual(window.poll_key(), -1)

    def test_window_close_and_repeated_cleanup_are_safe(self):
        glfw, _, _ = self.gui()
        window = PreviewWindow('test').__enter__()
        glfw.set_window_close_callback.call_args.args[1]('window')
        self.assertFalse(window.is_open())
        window.close()
        self.assertEqual(window.poll_key(), -1)
        window.close()
        glfw.destroy_window.assert_called_once_with('window')
        glfw.terminate.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            window.show(np.zeros((2, 2, 3), np.uint8))

    def test_creation_failure_and_context_loss_release_resources(self):
        glfw, gl, _ = self.gui()
        glfw.create_window.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'create an OpenGL'):
            PreviewWindow('test').__enter__()
        glfw.terminate.assert_called_once_with()
        glfw.destroy_window.assert_not_called()
        glfw.create_window.return_value = 'window'
        with PreviewWindow('test'):
            gl.glDeleteTextures.side_effect = RuntimeError('context lost')
        glfw.destroy_window.assert_called_once_with('window')
        self.assertEqual(glfw.terminate.call_count, 2)

    def test_gui_operations_require_main_thread_and_bad_images_are_rejected(self):
        glfw, _, _ = self.gui()
        errors = []
        def start_in_thread():
            try:
                PreviewWindow('thread').__enter__()
            except RuntimeError as error:
                errors.append(str(error))
        thread = Thread(target=start_in_thread)
        thread.start()
        thread.join(1)
        self.assertIn('main thread', errors[0])
        with PreviewWindow('test') as window:
            for invalid in (None, np.zeros((2, 2), np.uint8), np.zeros((2, 2, 3), float)):
                with self.assertRaises(ValueError):
                    window.show(invalid)
            glfw.window_should_close.return_value = True
            self.assertFalse(window.is_open())


if __name__ == '__main__':
    unittest.main()
