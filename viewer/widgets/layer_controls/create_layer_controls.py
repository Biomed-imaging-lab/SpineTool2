from viewer.layers.image.image import Image
from viewer.layers.labels.labels import Labels
from viewer.layers.points.points import Points
from viewer.layers.surface.surface import Surface
from viewer.widgets.layer_controls.qt_image_controls import QtImageControls
from viewer.widgets.layer_controls.qt_labels_controls import QtLabelsControls
from viewer.widgets.layer_controls.qt_points_controls import QtPointsControls
from viewer.widgets.layer_controls.qt_surface_controls import QtSurfaceControls

layer_to_controls = {
    Labels: QtLabelsControls,
    Image: QtImageControls,
    Points: QtPointsControls,
    Surface: QtSurfaceControls,
}


def create_qt_layer_controls(layer, parent=None):
    candidates = []
    for layer_type in layer_to_controls:
        if isinstance(layer, layer_type):
            candidates.append(layer_type)

    if not candidates:
        raise TypeError(
            "Could not find QtControls for layer of type {type_}".format(
                type_=type(layer),
            )
        )

    layer_cls = layer.__class__
    # Sort the list of candidates by 'lineage'
    candidates.sort(key=lambda layer_type: layer_cls.mro().index(layer_type))
    controls = layer_to_controls[candidates[0]]
    return controls(layer, parent)
