from json import dump, load
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal


class Settings(QObject):
    FILE_EXTENSION = ".json"  # works only with json

    updated = pyqtSignal(object)

    def __init__(
        self,
        schema: Dict[str, Any],
        ui_schema: Dict[str, Any],
        default_state: Dict[str, Any],
        description: str,
        path: str,
        not_displayed: List[str] = [],
    ):
        super().__init__()
        self._schema: Dict[str, Any] = schema
        self._ui_schema: Dict[str, Any] = ui_schema
        self._state: Dict[str, Any] = default_state.copy()
        self._default_state: Dict[str, Any] = default_state.copy()
        self._description = description
        self._path: str = path
        self._not_displayed: List[str] = not_displayed
        if not self._path.endswith(self.FILE_EXTENSION):
            self._path += self.FILE_EXTENSION

    @property
    def schema(self) -> Dict[str, Any]:
        return self._schema

    @property
    def ui_schema(self) -> Dict[str, Any]:
        return self._ui_schema

    @property
    def state(self) -> Dict[str, Any]:
        return self._state

    @property
    def description(self) -> str:
        return self._description

    @property
    def not_displayed(self) -> str:
        return self._not_displayed

    def update(self, state: Optional[Dict[str, Any]] = None) -> None:
        if state is None:
            new_state = self._default_state.copy()
        else:
            new_state = state

        updated_values = {}
        for key, value in new_state.items():
            if key in self._state and self._state[key] != value:
                self._state[key] = value
                updated_values[key] = value

        if len(updated_values) > 0:
            self.updated.emit(updated_values)

    def reset(self) -> None:
        self.update()

    def load(self) -> None:
        try:
            file = open(self._path)
            state = load(file)
            self.update(state)
            file.close()
        except:
            return

    def save(self) -> None:
        file = open(self._path, "w")
        dump(self._state, file)
        file.close()
