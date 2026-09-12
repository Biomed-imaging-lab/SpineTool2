from vispy.scene import SceneCanvas, Widget

from utils.colormaps.standardize_color import transform_color
from viewer.vispy.utils.gl import get_max_texture_sizes


class VispyCanvas(SceneCanvas):
    def __init__(self, *args, **kwargs) -> None:
        # Since the base class is frozen we must create this attribute
        # before calling super().__init__().
        self.max_texture_sizes = None
        self._last_theme_color = None
        self._background_color_override = None
        super().__init__(*args, **kwargs)

        # Call get_max_texture_sizes() here so that we query OpenGL right
        # now while we know a Canvas exists. Later calls to
        # get_max_texture_sizes() will return the same results because it's
        # using an lru_cache.
        self.max_texture_sizes = get_max_texture_sizes()

        self.events.ignore_callback_errors = False
        self.context.set_depth_func("lequal")

    @property
    def destroyed(self):
        return self._backend.destroyed

    @property
    def background_color_override(self):
        return self._background_color_override

    @background_color_override.setter
    def background_color_override(self, value):
        self._background_color_override = value
        self.bgcolor = value or self._last_theme_color

    def _on_theme_change(self, theme):
        from utils.themes import color_as_hex, get_theme

        self._last_theme_color = transform_color(
            color_as_hex(get_theme(theme, False).canvas)
        )[0]
        self.bgcolor = self._last_theme_color

    @property
    def bgcolor(self):
        SceneCanvas.bgcolor.fget(self)

    @bgcolor.setter
    def bgcolor(self, value):
        _value = self._background_color_override or value
        SceneCanvas.bgcolor.fset(self, _value)

    @property
    def central_widget(self):
        if self._central_widget is None:
            self._central_widget = Widget(
                size=self.size, parent=self.scene, border_width=0
            )
        return self._central_widget

    def _process_mouse_event(self, event):
        if event.type == "mouse_wheel" and len(event.modifiers) > 0:
            return
        if event.handled:
            return
        super()._process_mouse_event(event)
