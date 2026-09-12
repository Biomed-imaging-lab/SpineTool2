from typing import Type

import numpy as np
from vispy.scene import ArcballCamera, BaseCamera, PanZoomCamera

from viewer.vispy.utils.quaternion import quaternion2euler


class VispyCamera:
    def __init__(self, view, camera, dims) -> None:
        self._view = view
        self._camera = camera
        self._dims = dims

        # Create 2D camera
        self._2D_camera = MouseToggledPanZoomCamera(aspect=1)
        # flip y-axis to have correct alignment
        self._2D_camera.flip = (0, 1, 0)
        self._2D_camera.viewbox_key_event = viewbox_key_event

        # Create 3D camera
        self._3D_camera = MouseToggledArcballCamera(fov=0)
        self._3D_camera.viewbox_key_event = viewbox_key_event

        # Set 2D camera by default
        self._view.camera = self._2D_camera

        self._dims.ndisplay_.connect(self._on_ndisplay_change)

        self._camera.center_.connect(self._on_center_change)
        self._camera.zoom_.connect(self._on_zoom_change)
        self._camera.angles_.connect(self._on_angles_change)
        self._camera.perspective_.connect(self._on_perspective_change)
        self._camera.mouse_pan_.connect(self._on_mouse_toggles_change)
        self._camera.mouse_zoom_.connect(self._on_mouse_toggles_change)

        self._on_ndisplay_change()

    @property
    def angles(self):
        if self._view.camera == self._3D_camera:
            # Do conversion from quaternion representation to euler angles
            angles = quaternion2euler(self._view.camera._quaternion, degrees=True)
        else:
            angles = (0, 0, 90)
        return angles

    @angles.setter
    def angles(self, angles):
        if self.angles == tuple(angles):
            return

        # Only update angles if current camera is 3D camera
        if self._view.camera == self._3D_camera:
            # Create and set quaternion
            quat = self._view.camera._quaternion.create_from_euler_angles(
                *angles,
                degrees=True,
            )
            self._view.camera._quaternion = quat
            self._view.camera.view_changed()

    @property
    def center(self):
        if self._view.camera == self._3D_camera:
            center = tuple(self._view.camera.center)
        else:
            # in 2D, we arbitrarily choose 0.0 as the center in z
            center = (*self._view.camera.center[:2], 0.0)
        # switch from VisPy xyz ordering to NumPy prc ordering
        return center[::-1]

    @center.setter
    def center(self, center):
        if self.center == tuple(center):
            return
        self._view.camera.center = center[::-1]
        self._view.camera.view_changed()

    @property
    def zoom(self):
        canvas_size = np.array(self._view.canvas.size)
        if self._view.camera == self._3D_camera:
            scale = self._view.camera.scale_factor
        else:
            scale = np.array(
                [self._view.camera.rect.width, self._view.camera.rect.height]
            )
            scale[np.isclose(scale, 0)] = 1  # fix for #2875
        zoom = np.min(canvas_size / scale)
        return zoom

    @zoom.setter
    def zoom(self, zoom):
        if self.zoom == zoom:
            return
        scale = np.array(self._view.canvas.size) / zoom
        if self._view.camera == self._3D_camera:
            self._view.camera.scale_factor = np.min(scale)
        else:
            # Set view rectangle, as left, right, width, height
            corner = np.subtract(self._view.camera.center[:2], scale / 2)
            self._view.camera.rect = tuple(corner) + tuple(scale)

    @property
    def perspective(self):
        return self._3D_camera.fov

    @perspective.setter
    def perspective(self, perspective):
        if self.perspective == perspective:
            return
        self._3D_camera.fov = perspective
        self._view.camera.view_changed()

    @property
    def mouse_zoom(self) -> bool:
        return self._view.camera.mouse_zoom

    @mouse_zoom.setter
    def mouse_zoom(self, mouse_zoom: bool):
        self._view.camera.mouse_zoom = mouse_zoom

    @property
    def mouse_pan(self) -> bool:
        return self._view.camera.mouse_pan

    @mouse_pan.setter
    def mouse_pan(self, mouse_pan: bool):
        self._view.camera.mouse_pan = mouse_pan

    def _on_ndisplay_change(self):
        if self._dims.ndisplay == 3:
            self._view.camera = self._3D_camera
        else:
            self._view.camera = self._2D_camera

        self._on_mouse_toggles_change()
        self._on_center_change()
        self._on_zoom_change()
        self._on_angles_change()

    def _on_mouse_toggles_change(self):
        self.mouse_pan = self._camera.mouse_pan
        self.mouse_zoom = self._camera.mouse_zoom

    def _on_center_change(self):
        self.center = self._camera.center[-self._dims.ndisplay :]

    def _on_zoom_change(self):
        self.zoom = self._camera.zoom

    def _on_perspective_change(self):
        self.perspective = self._camera.perspective

    def _on_angles_change(self):
        self.angles = self._camera.angles

    def on_draw(self, _event):
        self._camera.angles_.disconnect(self._on_angles_change)
        self._camera.angles = self.angles
        self._camera.angles_.connect(self._on_angles_change)
        self._camera.center_.disconnect(self._on_center_change)
        self._camera.center = self.center
        self._camera.center_.connect(self._on_center_change)
        self._camera.zoom_.disconnect(self._on_zoom_change)
        self._camera.zoom = self.zoom
        self._camera.zoom_.connect(self._on_zoom_change)
        self._camera.perspective_.disconnect(self._on_perspective_change)
        self._camera.perspective = self.perspective
        self._camera.perspective_.connect(self._on_perspective_change)


def viewbox_key_event(event):
    return


def add_mouse_pan_zoom_toggles(
    vispy_camera_cls: Type[BaseCamera],
) -> Type[BaseCamera]:
    class _vispy_camera_cls(vispy_camera_cls):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mouse_pan = True
            self.mouse_zoom = True

        def viewbox_mouse_event(self, event):
            if (
                self.mouse_zoom
                and event.type == "mouse_wheel"
                or self.mouse_pan
                and event.type in ("mouse_move", "mouse_press", "mouse_release")
            ):
                try:
                    super().viewbox_mouse_event(event)
                except:
                    pass
            else:
                event.handled = False

    return _vispy_camera_cls


MouseToggledPanZoomCamera = add_mouse_pan_zoom_toggles(PanZoomCamera)
MouseToggledArcballCamera = add_mouse_pan_zoom_toggles(ArcballCamera)
