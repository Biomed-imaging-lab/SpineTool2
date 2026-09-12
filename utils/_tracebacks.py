import re
import sys
from types import TracebackType
from typing import Callable, Dict, Generator, Tuple, Type, Union

import numpy as np

ExcInfo = Union[
    Tuple[Type[BaseException], BaseException, TracebackType],
    Tuple[None, None, None],
]


def get_tb_formatter() -> Callable[[ExcInfo, bool, str], str]:
    import traceback

    if sys.version_info < (3, 11):
        import cgitb

        def cgitb_chain(exc: Exception) -> Generator[str, None, None]:
            if exc.__cause__:
                yield from cgitb_chain(exc.__cause__)
                yield (
                    '<br><br><font color="#51B432">The above exception was '
                    "the direct cause of the following exception:</font><br>"
                )
            elif exc.__context__:
                yield from cgitb_chain(exc.__context__)
                yield (
                    '<br><br><font color="#51B432">During handling of the '
                    "above exception, another exception occurred:</font><br>"
                )
            yield cgitb_html(exc)

        def cgitb_html(exc: Exception) -> str:
            info = (type(exc), exc, exc.__traceback__)
            return cgitb.html(info)

        def format_exc_info(info: ExcInfo, as_html: bool, color=None) -> str:
            np.set_printoptions(
                formatter={"all": lambda arr: f"{type(arr)} {arr.shape} {arr.dtype}"}
            )
            if as_html:
                html = "\n".join(cgitb_chain(info[1]))
                html = re.sub('bgcolor="#.*"', "", html)
                html = html.replace("<br>\n", "\n")
                html = re.sub(r"(<tr><td><small.*</tr>)", "<br>\\1<br>", html)
                html = html.replace(
                    "<p>A problem occurred in a Python script.  "
                    "Here is the sequence of",
                    "",
                )
                html = html.replace(
                    "function calls leading up to the error, "
                    "in the order they occurred.</p>",
                    "<br>",
                )
                html = html.replace('face="helvetica, arial"', "")
                html = (
                    "<span style='font-family: monaco,courier,monospace;'>"
                    + html
                    + "</span>"
                )
                tb_text = html
            else:
                tb_text = "".join(traceback.format_exception(*info))
            np.set_printoptions()
            return tb_text

    else:

        def format_exc_info(info: ExcInfo, as_html: bool, color=None) -> str:
            np.set_printoptions(
                formatter={"all": lambda arr: f"{type(arr)} {arr.shape} {arr.dtype}"}
            )
            tb_text = "".join(traceback.format_exception(*info))
            if as_html:
                tb_text = "<pre>" + tb_text + "</pre>"
            np.set_printoptions()
            return tb_text

    return format_exc_info


ANSI_STYLES = {
    1: {"font_weight": "bold"},
    2: {"font_weight": "lighter"},
    3: {"font_weight": "italic"},
    4: {"text_decoration": "underline"},
    5: {"text_decoration": "blink"},
    6: {"text_decoration": "blink"},
    8: {"visibility": "hidden"},
    9: {"text_decoration": "line-through"},
    30: {"color": "black"},
    31: {"color": "red"},
    32: {"color": "green"},
    33: {"color": "yellow"},
    34: {"color": "blue"},
    35: {"color": "magenta"},
    36: {"color": "cyan"},
    37: {"color": "white"},
}


def ansi2html(
    ansi_string: str, styles: Dict[int, Dict[str, str]] = ANSI_STYLES
) -> Generator[str, None, None]:
    previous_end = 0
    in_span = False
    ansi_codes = []
    ansi_finder = re.compile("\033\\[([\\d;]*)([a-zA-Z])")
    for match in ansi_finder.finditer(ansi_string):
        yield ansi_string[previous_end : match.start()]
        previous_end = match.end()
        params, command = match.groups()

        if command not in "mM":
            continue

        try:
            params = [int(p) for p in params.split(";")]
        except ValueError:
            params = [0]

        for i, v in enumerate(params):
            if v == 0:
                params = params[i + 1 :]
                if in_span:
                    in_span = False
                    yield "</span>"
                ansi_codes = []
                if not params:
                    continue

        ansi_codes.extend(params)
        if in_span:
            yield "</span>"
            in_span = False

        if not ansi_codes:
            continue

        style = [
            "; ".join([f"{k}: {v}" for k, v in styles[k].items()]).strip()
            for k in ansi_codes
            if k in styles
        ]
        yield '<span style="%s">' % "; ".join(style)

        in_span = True

    yield ansi_string[previous_end:]
    if in_span:
        yield "</span>"
        in_span = False
