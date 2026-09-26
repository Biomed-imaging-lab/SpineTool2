"""Compatibility tests for the subset of CGAL used by SpineTool 2.

The suite deliberately depends only on the Python standard library and CGAL.
It is intended to be the first check performed against a newly built CGAL
Python package, before installing the GUI and image-processing dependencies.
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path


if sys.version_info < (3, 10):
    raise RuntimeError(
        "SpineTool 2 CGAL tests require Python 3.10 or newer; "
        f"running {sys.version.split()[0]} from {sys.executable}"
    )

try:
    import CGAL
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polygon_mesh_processing import (
        Polylines,
        keep_connected_components,
        remove_connected_components,
        stitch_borders,
        triangulate_and_refine_hole,
        triangulate_hole,
        triangulate_refine_and_fair_hole,
        volume,
    )
    from CGAL.CGAL_Polyhedron_3 import (
        Polyhedron_3,
        Polyhedron_3_Facet_handle,
        Polyhedron_3_Halfedge_around_facet_circulator,
        Polyhedron_3_Halfedge_around_vertex_circulator,
        Polyhedron_3_Halfedge_handle,
        Polyhedron_3_Vertex_handle,
    )
    from CGAL.CGAL_Surface_mesh_skeletonization import (
        surface_mesh_skeletonization,
    )
except (ImportError, OSError) as exc:
    raise RuntimeError(
        "Cannot load the CGAL Python extension. Check that the extension was "
        "built for this Python version/architecture and that all CGAL, GMP, "
        "MPFR and compiler-runtime libraries are available. "
        f"Python: {sys.version.split()[0]} ({sys.executable}). Original error: {exc}"
    ) from exc


TETRAHEDRON_OFF = """OFF
4 4 0
0 0 0
1 0 0
0 1 0
0 0 1
3 0 2 1
3 0 1 3
3 1 2 3
3 2 0 3
"""

TWO_TETRAHEDRA_OFF = """OFF
8 8 0
0 0 0
1 0 0
0 1 0
0 0 1
3 0 0
4 0 0
3 1 0
3 0 1
3 0 2 1
3 0 1 3
3 1 2 3
3 2 0 3
3 4 6 5
3 4 5 7
3 5 6 7
3 6 4 7
"""

# A consistently oriented, triangulated cube.  It is closed and therefore a
# valid input for both volume() and surface_mesh_skeletonization().
TRIANGULATED_CUBE_OFF = """OFF
8 12 0
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
3 0 2 1
3 0 3 2
3 4 5 6
3 4 6 7
3 0 1 5
3 0 5 4
3 1 2 6
3 1 6 5
3 2 3 7
3 2 7 6
3 3 0 4
3 3 4 7
"""

# The same cube without its top two triangles.  The single four-edge border is
# used to verify all three hole-filling entry points used by the application.
OPEN_CUBE_OFF = """OFF
8 10 0
0 0 0
1 0 0
1 1 0
0 1 0
0 0 1
1 0 1
1 1 1
0 1 1
3 0 2 1
3 0 3 2
3 0 1 5
3 0 5 4
3 1 2 6
3 1 6 5
3 2 3 7
3 2 7 6
3 3 0 4
3 3 4 7
"""

# Two triangles have separate handles but coincident border edges.  Stitching
# should turn them into one connected square made from four vertices.
UNSTITCHED_SQUARE_OFF = """OFF
6 2 0
0 0 0
1 0 0
1 1 0
0 0 0
1 1 0
0 1 0
3 0 1 2
3 3 4 5
"""


class CgalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="spinetool-cgal-")
        self.temp_dir = Path(self._temp_dir.name)

    def tearDown(self) -> None:
        self._temp_dir.cleanup()

    def polyhedron(self, contents: str, name: str = "mesh.off") -> Polyhedron_3:
        path = self.temp_dir / name
        path.write_text(contents, encoding="ascii")
        mesh = Polyhedron_3(str(path))
        self.assertTrue(mesh.is_valid(), f"CGAL rejected fixture {name}")
        return mesh

    @staticmethod
    def border_halfedge(mesh: Polyhedron_3):
        for halfedge in mesh.halfedges():
            if halfedge.is_border():
                return halfedge
        raise AssertionError("mesh has no border halfedge")


class TestKernelAndPolyhedron(CgalTestCase):
    def test_package_and_all_imported_binding_types_are_available(self) -> None:
        self.assertTrue(getattr(CGAL, "__version__", ""))
        for binding_type in (
            Polyhedron_3,
            Polyhedron_3_Facet_handle,
            Polyhedron_3_Halfedge_around_facet_circulator,
            Polyhedron_3_Halfedge_around_vertex_circulator,
            Polyhedron_3_Halfedge_handle,
            Polyhedron_3_Vertex_handle,
        ):
            self.assertTrue(callable(binding_type))

    def test_point_coordinates_arithmetic_and_squared_length(self) -> None:
        first = Point_3(1.25, -2.5, 4.0)
        second = Point_3(-0.75, 1.5, 1.0)
        vector = first - second
        self.assertEqual((first.x(), first.y(), first.z()), (1.25, -2.5, 4.0))
        self.assertAlmostEqual(vector.squared_length(), 29.0)

    def test_polyhedron_io_iteration_handles_ids_and_point_mutation(self) -> None:
        mesh = self.polyhedron(TETRAHEDRON_OFF)
        self.assertEqual(mesh.size_of_vertices(), 4)
        self.assertEqual(mesh.size_of_facets(), 4)
        self.assertEqual(len(list(mesh.points())), 4)
        self.assertEqual(len(list(mesh.halfedges())), 12)

        vertices = list(mesh.vertices())
        for index, vertex in enumerate(vertices):
            vertex.set_id(index)
            self.assertEqual(vertex.id(), index)
        original = vertices[0].point()
        vertices[0].set_point(Point_3(original.x() + 0.125, original.y(), original.z()))
        self.assertAlmostEqual(vertices[0].point().x(), original.x() + 0.125)

        facet = next(iter(mesh.facets()))
        facet.set_id(17)
        self.assertEqual(facet.id(), 17)
        self.assertTrue(facet.is_triangle())
        around_facet = facet.facet_begin()
        facet_vertex_ids = []
        while around_facet.hasNext():
            facet_vertex_ids.append(around_facet.next().vertex().id())
        self.assertEqual(len(facet_vertex_ids), 3)

        vertex_halfedge = vertices[1].halfedge()
        around_vertex = vertex_halfedge.vertex_begin()
        neighbours = []
        while around_vertex.hasNext():
            neighbours.append(around_vertex.next().opposite().vertex().point())
        self.assertEqual(len(neighbours), 3)
        self.assertIsNotNone(vertex_halfedge.facet())

        clone = mesh.deepcopy()
        self.assertEqual(clone.size_of_vertices(), mesh.size_of_vertices())
        output = self.temp_dir / "roundtrip.off"
        clone.write_to_file(str(output))
        reloaded = Polyhedron_3(str(output))
        self.assertEqual(reloaded.size_of_facets(), 4)

    def test_polyhedron_erase_fill_hole_and_create_center_vertex(self) -> None:
        mesh = self.polyhedron(TETRAHEDRON_OFF)
        facet = next(iter(mesh.facets()))
        mesh.erase_facet(facet.halfedge())
        self.assertFalse(mesh.is_closed())
        self.assertEqual(mesh.size_of_facets(), 3)
        mesh.fill_hole(self.border_halfedge(mesh))
        self.assertTrue(mesh.is_closed())
        self.assertEqual(mesh.size_of_facets(), 4)

        open_cube = self.polyhedron(OPEN_CUBE_OFF, "open-cube.off")
        open_cube.fill_hole(self.border_halfedge(open_cube))
        non_triangle = next(f for f in open_cube.facets() if not f.is_triangle())
        before = open_cube.size_of_facets()
        open_cube.create_center_vertex(non_triangle.halfedge())
        self.assertTrue(open_cube.is_pure_triangle())
        self.assertGreater(open_cube.size_of_facets(), before)


class TestPolygonMeshProcessing(CgalTestCase):
    def test_volume(self) -> None:
        mesh = self.polyhedron(TRIANGULATED_CUBE_OFF)
        self.assertTrue(mesh.is_closed())
        self.assertTrue(mesh.is_pure_triangle())
        self.assertAlmostEqual(abs(volume(mesh)), 1.0, places=8)

    def test_keep_connected_components(self) -> None:
        mesh = self.polyhedron(TWO_TETRAHEDRA_OFF)
        component_seed = next(iter(mesh.facets()))
        keep_connected_components(mesh, [component_seed])
        self.assertEqual(mesh.size_of_vertices(), 4)
        self.assertEqual(mesh.size_of_facets(), 4)
        self.assertTrue(mesh.is_closed())

    def test_remove_connected_components(self) -> None:
        mesh = self.polyhedron(TWO_TETRAHEDRA_OFF)
        component_seed = next(iter(mesh.facets()))
        remove_connected_components(mesh, [component_seed])
        self.assertEqual(mesh.size_of_vertices(), 4)
        self.assertEqual(mesh.size_of_facets(), 4)
        self.assertTrue(mesh.is_closed())

    def test_stitch_borders(self) -> None:
        mesh = self.polyhedron(UNSTITCHED_SQUARE_OFF)
        before_vertices = mesh.size_of_vertices()
        before_borders = mesh.size_of_border_edges()
        stitch_borders(mesh)
        self.assertLess(mesh.size_of_vertices(), before_vertices)
        self.assertLess(mesh.size_of_border_edges(), before_borders)
        self.assertEqual(mesh.size_of_facets(), 2)
        self.assertTrue(mesh.is_valid())

    def _assert_hole_filler(self, filler) -> None:
        mesh = self.polyhedron(OPEN_CUBE_OFF, f"{filler.__name__}.off")
        facet_output = []
        vertex_output = []
        border = self.border_halfedge(mesh)
        if filler is triangulate_hole:
            filler(mesh, border, facet_output)
        else:
            filler(mesh, border, facet_output, vertex_output)
        self.assertTrue(mesh.is_closed())
        self.assertTrue(mesh.is_pure_triangle())
        self.assertGreaterEqual(len(facet_output), 2)
        self.assertTrue(mesh.is_valid())

    def test_triangulate_hole(self) -> None:
        self._assert_hole_filler(triangulate_hole)

    def test_triangulate_and_refine_hole(self) -> None:
        self._assert_hole_filler(triangulate_and_refine_hole)

    def test_triangulate_refine_and_fair_hole(self) -> None:
        self._assert_hole_filler(triangulate_refine_and_fair_hole)


class TestSurfaceMeshSkeletonization(CgalTestCase):
    def test_surface_mesh_skeletonization_and_polylines(self) -> None:
        mesh = self.polyhedron(TRIANGULATED_CUBE_OFF)
        skeleton = Polylines()
        correspondence = Polylines()
        surface_mesh_skeletonization(mesh, skeleton, correspondence)

        self.assertGreater(len(skeleton), 0)
        self.assertEqual(len(correspondence), mesh.size_of_vertices())
        for line in skeleton:
            self.assertGreaterEqual(len(line), 1)
            for point in line:
                self.assertTrue(
                    all(math.isfinite(value) for value in (point.x(), point.y(), point.z()))
                )
        for line in correspondence:
            self.assertEqual(len(line), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
