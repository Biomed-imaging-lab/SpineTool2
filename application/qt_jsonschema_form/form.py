from copy import deepcopy

from application.qt_jsonschema_form import widgets
from application.qt_jsonschema_form.defaults import compute_defaults


def get_widget_state(schema, state=None):
    if state is None:
        return compute_defaults(schema)
    return state


def get_schema_type(schema: dict) -> str:
    return schema["type"]


class WidgetBuilder:
    default_widget_map = {
        "boolean": {
            "checkbox": widgets.CheckboxSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
        "object": {
            "object": widgets.ObjectSchemaWidget,
            "horizontal_object": widgets.HorizontalObjectSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
            "shortcuts": widgets.ShortcutsWidget,
        },
        "number": {
            "spin": widgets.SpinDoubleSchemaWidget,
            "text": widgets.TextSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
        "string": {
            "textarea": widgets.TextAreaSchemaWidget,
            "text": widgets.TextSchemaWidget,
            "password": widgets.PasswordWidget,
            "filepath": widgets.FilepathSchemaWidget,
            "colour": widgets.ColorSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
        "integer": {
            "spin": widgets.SpinSchemaWidget,
            "text": widgets.TextSchemaWidget,
            "range": widgets.IntegerRangeSchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
        "array": {
            "array": widgets.ArraySchemaWidget,
            "enum": widgets.EnumSchemaWidget,
        },
    }

    default_widget_variants = {
        "boolean": "checkbox",
        "object": "object",
        "array": "array",
        "number": "spin",
        "integer": "spin",
        "string": "text",
    }

    widget_variant_modifiers = {"string": lambda schema: schema.get("format", "text")}

    def __init__(self):
        self.widget_map = deepcopy(self.default_widget_map)

    def create_form(
        self, schema: dict, ui_schema: dict, state=None
    ) -> widgets.SchemaWidgetMixin:
        schema_widget = self.create_widget(schema, ui_schema, state)

        form = widgets.FormWidget(schema_widget)

        return form

    def create_widget(
        self,
        schema: dict,
        ui_schema: dict,
        state=None,
        description: str = "",
    ) -> widgets.SchemaWidgetMixin:
        schema_type = get_schema_type(schema)

        try:
            default_variant = self.widget_variant_modifiers[schema_type](schema)
        except KeyError:
            default_variant = self.default_widget_variants[schema_type]

        if "enum" in schema:
            default_variant = "enum"

        if schema.get("description"):
            description = schema["description"]

        widget_variant = ui_schema.get("ui:widget", default_variant)
        widget_cls = self.widget_map[schema_type][widget_variant]
        widget = widget_cls(schema, ui_schema, self)
        default_state = get_widget_state(schema, state)
        if default_state is not None:
            widget.state = default_state

        if description:
            widget.setDescription(description)
            widget.setToolTip(description)

        return widget
