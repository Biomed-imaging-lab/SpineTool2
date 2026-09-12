from vispy.scene.visuals import Mesh, MeshNormals
from vispy.visuals.filters import WireframeFilter

from viewer.vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin


class SurfaceVisual(ClippingPlanesMixin, Mesh):
    def __init__(self, *args, **kwargs) -> None:
        self.wireframe_filter = WireframeFilter(width=1)
        self.face_normals = None
        self.vertex_normals = None
        super().__init__(*args, **kwargs)
        self.face_normals = MeshNormals(primitive="face", parent=self)
        self.vertex_normals = MeshNormals(primitive="vertex", parent=self)
        self.attach(self.wireframe_filter)
