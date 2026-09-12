from utils.misc import StringEnum


class CursorStyle(StringEnum):
    """CursorStyle: Style on the cursor.

    Sets the style of the cursor
            * square: A square
            * circle: A circle
            * cross: A cross
            * forbidden: A forbidden symbol
            * pointing: A finger for pointing
            * standard: The standard cursor
            # crosshair: A crosshair
    """

    SQUARE = "square"
    CIRCLE = "circle"
    CROSS = "cross"
    FORBIDDEN = "forbidden"
    POINTING = "pointing"
    STANDARD = "standard"
    CROSSHAIR = "crosshair"
