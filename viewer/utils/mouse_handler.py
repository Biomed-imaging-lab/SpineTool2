import contextlib
import inspect
from typing import Any, Callable, Dict, Generator, List


class MouseHandler:
    def __init__(self):
        self.mouse_move_callbacks: List[Callable] = []
        self.mouse_double_click_callbacks: List[Callable] = []
        self.mouse_wheel_callbacks: List[Callable] = []
        self.mouse_drag_callbacks: List[Callable] = []
        self._persisted_mouse_event: Dict[Generator, Any] = {}
        self._mouse_drag_gen: Dict[Callable, Generator] = {}
        self._mouse_wheel_gen: Dict[Callable, Generator] = {}

    def on_mouse_double_clicked(self, event) -> None:
        for mouse_click_func in self.mouse_double_click_callbacks:
            if inspect.isgeneratorfunction(mouse_click_func):
                raise ValueError("Double-click actions can't be generators.")
            mouse_click_func(self, event)

    def on_mouse_pressed(self, event) -> None:
        for mouse_drag_func in self.mouse_drag_callbacks:
            gen = mouse_drag_func(self, event)
            if inspect.isgenerator(gen):
                try:
                    next(gen)
                    self._mouse_drag_gen[mouse_drag_func] = gen
                    self._persisted_mouse_event[gen] = event
                except StopIteration:
                    pass

    def on_mouse_moved(self, event) -> None:
        if not event.is_dragging:
            for mouse_move_func in self.mouse_move_callbacks:
                mouse_move_func(self, event)

        for func, gen in tuple(self._mouse_drag_gen.items()):
            self._persisted_mouse_event[gen].__wrapped__ = event
            try:
                next(gen)
            except StopIteration:
                del self._mouse_drag_gen[func]
                del self._persisted_mouse_event[gen]

    def on_mouse_wheel_scrolled(self, event) -> None:
        for mouse_wheel_func in self.mouse_wheel_callbacks:
            gen = mouse_wheel_func(self, event)
            if inspect.isgenerator(gen):
                try:
                    next(gen)
                    self._mouse_wheel_gen[mouse_wheel_func] = gen
                    self._persisted_mouse_event[gen] = event
                except StopIteration:
                    pass

    def on_mouse_released(self, event) -> None:
        for func, gen in tuple(self._mouse_drag_gen.items()):
            self._persisted_mouse_event[gen].__wrapped__ = event
            with contextlib.suppress(StopIteration):
                next(gen)
            del self._mouse_drag_gen[func]
            del self._persisted_mouse_event[gen]
