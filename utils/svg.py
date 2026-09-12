"""
A Class for generating QIcons from SVGs with arbitrary colors at runtime.
"""

from typing import Optional, Union

from PyQt5.QtCore import QByteArray, QPoint, QRect, QRectF, Qt
from PyQt5.QtGui import QIcon, QIconEngine, QImage, QPainter, QPixmap
from PyQt5.QtSvg import QSvgRenderer


class QColoredSVGIcon(QIcon):
    """A QIcon class that specializes in colorizing SVG files.

    Parameters
    ----------
    path_or_xml : str
        Raw SVG XML or a path to an existing svg file.  (Will raise error on
        ``__init__`` if a non-existent file is provided.)
    color : str, optional
        A valid CSS color string, used to colorize the SVG. by default None.
    opacity : float, optional
        Fill opacity for the icon (0-1).  By default 1 (opaque).
    """

    def __init__(
        self, path_or_xml: str, color: Optional[str] = None, opacity: float = 1.0
    ):
        from utils._icons import get_colorized_svg

        self._svg = path_or_xml
        colorized = get_colorized_svg(path_or_xml, color, opacity)
        super().__init__(SVGBufferIconEngine(colorized))

    def colored(
        self,
        color: Optional[str] = None,
        opacity: float = 1.0,
        theme: Optional[str] = None,
        theme_key: str = "icon",
    ) -> "QColoredSVGIcon":
        """Return a new colorized QIcon instance.

        Parameters
        ----------
        color : str, optional
            A valid CSS color string, used to colorize the SVG.  If provided,
            will take precedence over ``theme``, by default None.
        opacity : float, optional
            Fill opacity for the icon (0-1).  By default 1 (opaque).
        theme : str, optional
            Name of the theme to from which to get `theme_key` color.
            ``color`` argument takes precedence.
        theme_key : str, optional
            If using a theme, key in the theme dict to use, by default 'icon'

        Returns
        -------
        QColoredSVGIcon
            A pre-colored QColoredSVGIcon (which may still be recolored)
        """
        if not color and theme:
            from utils.themes import color_as_hex, get_theme

            color = color_as_hex(getattr(get_theme(theme, False), theme_key))

        return QColoredSVGIcon(self._svg, color, opacity)


class SVGBufferIconEngine(QIconEngine):
    def __init__(self, xml: Union[str, bytes]) -> None:
        if isinstance(xml, str):
            xml = xml.encode("utf-8")
        self.data = QByteArray(xml)
        super().__init__()

    def paint(self, painter: QPainter, rect, mode, state):
        """Paint the icon int ``rect`` using ``painter``."""
        renderer = QSvgRenderer(self.data)
        renderer.render(painter, QRectF(rect))

    def clone(self):
        """Required to subclass abstract QIconEngine."""
        return SVGBufferIconEngine(self.data)

    def pixmap(self, size, mode, state):
        """Return the icon as a pixmap with requested size, mode, and state."""
        img = QImage(size, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        pixmap = QPixmap.fromImage(img, Qt.NoFormatConversion)
        painter = QPainter(pixmap)
        self.paint(painter, QRect(QPoint(0, 0), size), mode, state)
        return pixmap
