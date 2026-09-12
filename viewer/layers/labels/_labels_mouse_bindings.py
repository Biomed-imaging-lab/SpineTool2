from viewer.layers.labels._labels_utils import mouse_event_to_labels_coordinate
from viewer.layers.labels.labels_constants import Mode


def draw(layer, event):
    coordinates = mouse_event_to_labels_coordinate(layer, event)
    if layer._mode == Mode.ERASE:
        new_label = layer.colormap.background_value
    else:
        new_label = layer.selected_label

    # on press
    with layer.block_history():
        layer._draw(new_label, coordinates, coordinates)
        yield

        last_cursor_coord = coordinates
        # on move
        while event.type == "mouse_move":
            coordinates = mouse_event_to_labels_coordinate(layer, event)
            if coordinates is not None or last_cursor_coord is not None:
                layer._draw(new_label, last_cursor_coord, coordinates)
            last_cursor_coord = coordinates
            yield
