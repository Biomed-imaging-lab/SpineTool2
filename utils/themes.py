import re
from ast import literal_eval
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

from utils._icons import write_colorized_svgs
from utils.appdirs import data_dir

Color = str


class Theme:
    """Theme model.

    Attributes
    ----------
    id : str
        id of the theme and name of the virtual folder where icons
        will be saved to.
    label : str
        Name of the theme as it should be shown in the ui.
    syntax_style : str
        Name of the console style.
        See for more details: https://pygments.org/docs/styles/
    canvas : Color
        Background color of the canvas.
    background : Color
        Color of the application background.
    foreground : Color
        Color to contrast with the background.
    primary : Color
        Color used to make part of a widget more visible.
    secondary : Color
        Alternative color used to make part of a widget more visible.
    highlight : Color
        Color used to highlight visual element.
    text : Color
        Color used to display text.
    warning : Color
        Color used to indicate something needs attention.
    error : Color
        Color used to indicate something is wrong or could stop functionality.
    current : Color
        Color used to highlight Qt widget.
    """

    def __init__(
        self,
        id: str,
        label: str,
        syntax_style: str,
        canvas: Color,
        console: Color,
        background: Color,
        foreground: Color,
        primary: Color,
        secondary: Color,
        highlight: Color,
        text: Color,
        icon: Color,
        warning: Color,
        error: Color,
        current: Color,
    ):
        self.id = id
        self.label = label
        self.syntax_style = syntax_style
        self.canvas = canvas
        self.console = console
        self.background = background
        self.foreground = foreground
        self.primary = primary
        self.secondary = secondary
        self.highlight = highlight
        self.text = text
        self.icon = icon
        self.warning = warning
        self.error = error
        self.current = current

    def copy(self) -> "Theme":
        return Theme(
            self.id,
            self.label,
            self.syntax_style,
            self.canvas,
            self.console,
            self.background,
            self.foreground,
            self.primary,
            self.secondary,
            self.highlight,
            self.text,
            self.icon,
            self.warning,
            self.error,
            self.current,
        )

    def dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "syntax_style": self.syntax_style,
            "canvas": self.canvas,
            "console": self.console,
            "background": self.background,
            "foreground": self.foreground,
            "primary": self.primary,
            "secondary": self.secondary,
            "highlight": self.highlight,
            "text": self.text,
            "icon": self.icon,
            "warning": self.warning,
            "error": self.error,
            "current": self.current,
        }


gradient_pattern = re.compile(r"([vh])gradient\((.+)\)")
darken_pattern = re.compile(r"{{\s?darken\((\w+),?\s?([-\d]+)?\)\s?}}")
lighten_pattern = re.compile(r"{{\s?lighten\((\w+),?\s?([-\d]+)?\)\s?}}")
opacity_pattern = re.compile(r"{{\s?opacity\((\w+),?\s?([-\d]+)?\)\s?}}")


def darken(color: Color, percentage=10):
    color = literal_eval(color.lstrip("rgb(").rstrip(")"))
    ratio = 1 - float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red * ratio), 0), 255)
    green = min(max(int(green * ratio), 0), 255)
    blue = min(max(int(blue * ratio), 0), 255)
    return f"rgb({red}, {green}, {blue})"


def lighten(color: Color, percentage=10):
    color = literal_eval(color.lstrip("rgb(").rstrip(")"))
    ratio = float(percentage) / 100
    red, green, blue = color
    red = min(max(int(red + (255 - red) * ratio), 0), 255)
    green = min(max(int(green + (255 - green) * ratio), 0), 255)
    blue = min(max(int(blue + (255 - blue) * ratio), 0), 255)
    return f"rgb({red}, {green}, {blue})"


def opacity(color: Color, value=255):
    color = literal_eval(color.lstrip("rgb(").rstrip(")"))
    red, green, blue = color
    return f"rgba({red}, {green}, {blue}, {max(min(int(value), 255), 0)})"


def gradient(stops, horizontal=True):
    if horizontal:
        grad = "qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, "
    else:
        grad = "qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, "

    _stops = [f"stop: {n} {stop}" for n, stop in enumerate(stops)]
    grad += ", ".join(_stops) + ")"

    return grad


def template(css: str, **theme):
    def darken_match(matchobj):
        color, percentage = matchobj.groups()
        return darken(theme[color], percentage)

    def lighten_match(matchobj):
        color, percentage = matchobj.groups()
        return lighten(theme[color], percentage)

    def opacity_match(matchobj):
        color, percentage = matchobj.groups()
        return opacity(theme[color], percentage)

    def gradient_match(matchobj):
        horizontal = matchobj.groups()[1] == "h"
        stops = [i.strip() for i in matchobj.groups()[1].split("-")]
        return gradient(stops, horizontal)

    for k, v in theme.items():
        css = gradient_pattern.sub(gradient_match, css)
        css = darken_pattern.sub(darken_match, css)
        css = lighten_pattern.sub(lighten_match, css)
        css = opacity_pattern.sub(opacity_match, css)
        css = css.replace("{{ %s }}" % k, v)
    return css


DARK = Theme(
    id="dark",
    label="Default Dark",
    background="rgb(38, 41, 48)",
    foreground="rgb(65, 72, 81)",
    primary="rgb(90, 98, 108)",
    secondary="rgb(134, 142, 147)",
    highlight="rgb(106, 115, 128)",
    text="rgb(240, 241, 242)",
    icon="rgb(209, 210, 212)",
    warning="rgb(227, 182, 23)",
    error="rgb(153, 18, 31)",
    current="rgb(0, 122, 204)",
    syntax_style="native",
    console="rgb(18, 18, 18)",
    canvas="rgb(0, 0, 0)",
)
LIGHT = Theme(
    id="light",
    label="Default Light",
    background="rgb(239, 235, 233)",
    foreground="rgb(214, 208, 206)",
    primary="rgb(188, 184, 181)",
    secondary="rgb(150, 146, 144)",
    highlight="rgb(163, 158, 156)",
    text="rgb(59, 58, 57)",
    icon="rgb(107, 105, 103)",
    warning="rgb(227, 182, 23)",
    error="rgb(255, 18, 31)",
    current="rgb(253, 240, 148)",
    syntax_style="default",
    console="rgb(255, 255, 255)",
    canvas="rgb(255, 255, 255)",
)


THEMES: OrderedDict[str, Theme] = {DARK.id: DARK, LIGHT.id: LIGHT}


def color_as_hex(color: str) -> str:
    color = color[4:-1]
    rgb = [int(c) for c in color.split(",")]
    as_hex = "".join(f"{v:02x}" for v in rgb)
    return f"#{as_hex}"


def get_theme(theme_id: str = "", as_dict: bool = False):
    if theme_id not in THEMES:
        theme = DARK
    else:
        theme = THEMES[theme_id]
    _theme = theme.copy()
    if as_dict:
        _theme = _theme.dict()
        _theme = {k: v for (k, v) in _theme.items()}
        return _theme
    return _theme


def theme_path(theme_id: str) -> Path:
    return Path(data_dir()) / "themes" / theme_id


def build_theme_svgs(theme_id: str, svg_paths: List[str]) -> str:
    out = theme_path(theme_id)
    write_colorized_svgs(
        out,
        svg_paths=svg_paths,
        colors=[(theme_id, "icon")],
        opacities=(0.5, 1),
        theme_override={
            "warning": "warning",
            "error": "error",
            "logo_silhouette": "background",
        },
    )
    return str(out)


def get_stylesheet(theme_id: str, files: List[str]) -> str:
    stylesheet = ""
    for file in files:
        with open(file) as f:
            stylesheet += f.read()
    return template(stylesheet, **get_theme(theme_id, as_dict=True))


class Style:
    def __init__(self, stylesheets_paths: List[str], icons_paths: List[str]):
        self._stylesheets_paths = stylesheets_paths
        self._icons_paths = icons_paths

    @property
    def stylesheets_paths(
        self,
    ) -> List[
        str
    ]:  # return paths to qss templates for generating styles for current theme
        # {{ theme param name }} - places to substitute values from theme
        return self._stylesheets_paths

    @property
    def icons_paths(
        self,
    ) -> List[str]:  # return paths to svg icons for generating styles for current theme
        return self._icons_paths
