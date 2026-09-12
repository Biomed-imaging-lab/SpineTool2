from typing import Callable, Dict, List, Optional, OrderedDict, Tuple

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeyEvent

MODIFIER_KEYS = [Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Shift, Qt.Key.Key_Meta]

MODIFIER_FLAGS = OrderedDict(
    [
        (Qt.KeyboardModifier.ControlModifier, Qt.Key.Key_Control),
        (Qt.KeyboardModifier.AltModifier, Qt.Key.Key_Alt),
        (Qt.KeyboardModifier.ShiftModifier, Qt.Key.Key_Shift),
        (Qt.KeyboardModifier.MetaModifier, Qt.Key.Key_Meta),
    ]
)

KEY_SYMBOLS = {
    Qt.Key.Key_Control: "Ctrl",
    Qt.Key.Key_Shift: "Shift",
    Qt.Key.Key_Alt: "Alt",
    Qt.Key.Key_Meta: "⊞",
    Qt.Key.Key_Left: "←",
    Qt.Key.Key_Right: "→",
    Qt.Key.Key_Up: "↑",
    Qt.Key.Key_Down: "↓",
    Qt.Key.Key_Backspace: "⌫",
    Qt.Key.Key_Delete: "⌦",
    Qt.Key.Key_Tab: "↹",
    Qt.Key.Key_Escape: "Esc",
    Qt.Key.Key_Enter: "↵",
    Qt.Key.Key_Space: "␣",
    Qt.Key.Key_Slash: "/",
    Qt.Key.Key_Plus: "+",
    Qt.Key.Key_Minus: "-",
    Qt.Key.Key_F1: "F1",
    Qt.Key.Key_F2: "F2",
    Qt.Key.Key_F3: "F3",
    Qt.Key.Key_F4: "F4",
    Qt.Key.Key_F5: "F5",
    Qt.Key.Key_F6: "F6",
    Qt.Key.Key_F7: "F7",
    Qt.Key.Key_F8: "F8",
    Qt.Key.Key_F9: "F9",
    Qt.Key.Key_F10: "F10",
    Qt.Key.Key_F11: "F11",
    Qt.Key.Key_F12: "F12",
    Qt.Key.Key_Home: "Home",
    Qt.Key.Key_End: "End",
    Qt.Key.Key_PageUp: "PgUp",
    Qt.Key.Key_PageDown: "PgDown",
    Qt.Key.Key_Insert: "Insert",
    Qt.Key.Key_A: "A",
    Qt.Key.Key_B: "B",
    Qt.Key.Key_C: "C",
    Qt.Key.Key_D: "D",
    Qt.Key.Key_E: "E",
    Qt.Key.Key_F: "F",
    Qt.Key.Key_G: "G",
    Qt.Key.Key_H: "H",
    Qt.Key.Key_I: "I",
    Qt.Key.Key_J: "J",
    Qt.Key.Key_K: "K",
    Qt.Key.Key_L: "L",
    Qt.Key.Key_M: "M",
    Qt.Key.Key_N: "N",
    Qt.Key.Key_O: "O",
    Qt.Key.Key_P: "P",
    Qt.Key.Key_Q: "Q",
    Qt.Key.Key_R: "R",
    Qt.Key.Key_S: "S",
    Qt.Key.Key_T: "T",
    Qt.Key.Key_U: "U",
    Qt.Key.Key_V: "V",
    Qt.Key.Key_W: "W",
    Qt.Key.Key_X: "X",
    Qt.Key.Key_Y: "Y",
    Qt.Key.Key_Z: "Z",
    Qt.Key.Key_0: "0",
    Qt.Key.Key_1: "1",
    Qt.Key.Key_2: "2",
    Qt.Key.Key_3: "3",
    Qt.Key.Key_4: "4",
    Qt.Key.Key_5: "5",
    Qt.Key.Key_6: "6",
    Qt.Key.Key_7: "7",
    Qt.Key.Key_8: "8",
    Qt.Key.Key_9: "9",
}


class Shortcut:
    def __init__(
        self,
        key_combo: str,
        callback: Optional[Callable] = None,
        on_release_callback: Optional[Callable] = None,
        time_interval: Optional[int] = None,
        description: str = "",
        ignore_auto_repeat: bool = False,
    ):
        self._key_combo: str = key_combo
        self._callback: Optional[Callable] = callback
        self._on_release_callback: Optional[Callable] = on_release_callback
        self._time_interval: Optional[int] = time_interval
        self._timer: Optional[QTimer] = None
        self._description: str = description
        self.ignore_auto_repeat: bool = ignore_auto_repeat

    @property
    def key_combo(self) -> str:
        return self._key_combo

    @property
    def description(self) -> str:
        return self._description

    @property
    def without_action(self) -> bool:
        return self._callback is None and self._on_release_callback is None

    def pressed(self) -> None:
        if self._callback is None:
            return
        if self._time_interval is None:
            self._callback()
        else:
            self._timer = QTimer()
            self._timer.timeout.connect(self._callback)
            self._timer.setInterval(self._time_interval)

    def released(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._on_release_callback is not None:
            self._on_release_callback()


def create_key_combo_string(
    key: Qt.Key, modifiers: List[Qt.Key]
) -> Optional[str]:  # None for invalid key
    if key not in KEY_SYMBOLS:
        return None

    applied_modifiers = []
    for modifier in MODIFIER_KEYS:  # invalid modifiers are simply ignored
        if (
            modifier in modifiers
            and modifier != key
            and modifier not in applied_modifiers
        ):
            applied_modifiers.append(modifier)

    values = []
    for modifier in modifiers:
        values.append(KEY_SYMBOLS[modifier])
    values.append(KEY_SYMBOLS[key])
    return "+".join(values)


def create_key_combo_string(event: QKeyEvent) -> Optional[str]:  # None for invalid key
    key = event.key()
    if key not in KEY_SYMBOLS:
        key = event.nativeVirtualKey()
    if key not in KEY_SYMBOLS:
        return None

    applied_modifiers = []
    for flag, key_ in MODIFIER_FLAGS.items():
        if event.modifiers() & flag and key_ != key:
            applied_modifiers.append(key_)

    values = []
    for modifier in applied_modifiers:
        values.append(KEY_SYMBOLS[modifier])
    values.append(KEY_SYMBOLS[key])
    return "+".join(values)


Shortcuts = Dict[str, Shortcut]


class ShortcutsHandler:
    def __init__(
        self,
        shortcut_groups: Optional[Dict[str, Shortcuts]] = None,
        active_groups: Optional[List[str]] = None,
    ):
        self._shortcut_groups: Dict[str, Shortcuts] = (
            {} if shortcut_groups is None else shortcut_groups
        )
        self._active_groups: List[str] = [] if active_groups is None else active_groups

    @property
    def shortcuts_info(self) -> List[Tuple[str, str, str]]:
        info = []
        for group, shortcuts in self._shortcut_groups.items():
            for shortcut in shortcuts.values():
                info.append((shortcut.description, group, shortcut.key_combo))
        return info

    @property
    def active_shortcuts(self) -> List[Shortcuts]:
        active = []
        for group in self._active_groups:
            active.append(self._shortcut_groups[group])
        return active

    @active_shortcuts.setter
    def active_shortcuts(self, active_groups: List[str]) -> None:
        self._active_groups = active_groups

    def toggle_active(self, groups: List[str]) -> None:
        for group in groups:
            if group in self._shortcut_groups:
                if group not in self._active_groups:
                    self._active_groups.append(group)
                else:
                    self._active_groups.remove(group)

    def register_shortcuts(
        self, group: str, shortcuts: List[Shortcut] = []
    ) -> List[Shortcut]:  # return list of unregistered key combos
        unregistered = []
        if group not in self._shortcut_groups:
            new_group = {}
            for shortcut in shortcuts:
                new_group[shortcut.key_combo] = shortcut
            self._shortcut_groups[group] = new_group
        else:
            for shortcut in shortcuts:
                if shortcut.key_combo in self._shortcut_groups[group]:
                    unregistered.append(shortcut)
                else:
                    self._shortcut_groups[group][shortcut.key_combo] = shortcut
        return unregistered

    def on_key_pressed(self, event: QKeyEvent) -> bool:
        shortcut = self._find_shortcut(event)
        if shortcut is not None and (
            (not shortcut.ignore_auto_repeat and event.isAutoRepeat())
            or not event.isAutoRepeat()
        ):
            shortcut.pressed()
            return True
        return False

    def on_key_released(self, event: QKeyEvent) -> bool:
        shortcut = self._find_shortcut(event)
        if shortcut is not None and (
            (not shortcut.ignore_auto_repeat and event.isAutoRepeat())
            or not event.isAutoRepeat()
        ):
            shortcut.released()
            return True
        return False

    def _find_shortcut(self, event: QKeyEvent) -> Optional[Shortcut]:
        key_combo = create_key_combo_string(event)
        if key_combo is None:
            return None
        for active_group in self.active_shortcuts:
            if key_combo not in active_group:
                continue
            else:
                shortcut = active_group[key_combo]
                if shortcut.without_action:
                    return None
                else:
                    return shortcut
        return None
