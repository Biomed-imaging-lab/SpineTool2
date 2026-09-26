import unittest
from unittest.mock import patch

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QKeyEvent

from utils.shortcuts import Shortcut, ShortcutsHandler, format_shortcut


class TestPlatformShortcuts(unittest.TestCase):
    def test_native_labels_preserve_arrow_delete_and_plus_keys(self):
        with patch('utils.shortcuts.sys.platform', 'darwin'):
            self.assertEqual(format_shortcut('Ctrl+Shift+Z'), '⌘+⇧+Z')
            self.assertEqual(format_shortcut('Ctrl+Alt+→'), '⌘+⌥+→')
            self.assertEqual(format_shortcut('Ctrl+⌫'), '⌘+⌫')
            self.assertEqual(format_shortcut('⊞+A'), '⌃+A')
            self.assertEqual(format_shortcut('+'), '+')
        with patch('utils.shortcuts.sys.platform', 'win32'):
            self.assertEqual(format_shortcut('Ctrl+Shift+Z'), 'Ctrl+Shift+Z')
            self.assertEqual(format_shortcut('Alt+→'), 'Alt+→')

    def test_display_does_not_change_shortcut_dispatch(self):
        calls = []
        handler = ShortcutsHandler(active_groups=['project'])
        handler.register_shortcuts('project', [Shortcut('Ctrl+S', lambda: calls.append('save'))])
        with patch('utils.shortcuts.sys.platform', 'darwin'):
            self.assertEqual(handler.shortcuts_info[0][2], '⌘+S')
            self.assertTrue(handler.on_key_pressed(QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.ControlModifier)))
            self.assertFalse(handler.on_key_pressed(QKeyEvent(QEvent.KeyPress, Qt.Key_S, Qt.MetaModifier)))
        self.assertEqual(calls, ['save'])

    def test_preferences_comma_and_redo_dispatch(self):
        calls = []
        handler = ShortcutsHandler(active_groups=['project'])
        handler.register_shortcuts('project', [
            Shortcut('Ctrl+,', lambda: calls.append('preferences')),
            Shortcut('Ctrl+Shift+Z', lambda: calls.append('redo')),
        ])
        self.assertTrue(handler.on_key_pressed(QKeyEvent(QEvent.KeyPress, Qt.Key_Comma, Qt.ControlModifier)))
        self.assertTrue(handler.on_key_pressed(QKeyEvent(QEvent.KeyPress, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)))
        self.assertEqual(calls, ['preferences', 'redo'])


if __name__ == '__main__':
    unittest.main()
