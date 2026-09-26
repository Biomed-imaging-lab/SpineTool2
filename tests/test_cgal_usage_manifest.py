"""Keep the CGAL compatibility suite in sync with production imports."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (PROJECT_ROOT / "application", PROJECT_ROOT / "projects")

# Every entry below has a behavioural test in test_cgal_compatibility.py.  If a
# new CGAL symbol is introduced in production, this guard fails until its
# compatibility contract is added to that suite and then to this manifest.
TESTED_CGAL_IMPORTS = {
    "CGAL.CGAL_Kernel.Point_3",
    "CGAL.CGAL_Polygon_mesh_processing.Polylines",
    "CGAL.CGAL_Polygon_mesh_processing.keep_connected_components",
    "CGAL.CGAL_Polygon_mesh_processing.remove_connected_components",
    "CGAL.CGAL_Polygon_mesh_processing.stitch_borders",
    "CGAL.CGAL_Polygon_mesh_processing.triangulate_and_refine_hole",
    "CGAL.CGAL_Polygon_mesh_processing.triangulate_hole",
    "CGAL.CGAL_Polygon_mesh_processing.triangulate_refine_and_fair_hole",
    "CGAL.CGAL_Polygon_mesh_processing.volume",
    "CGAL.CGAL_Polyhedron_3.Polyhedron_3",
    "CGAL.CGAL_Polyhedron_3.Polyhedron_3_Facet_handle",
    "CGAL.CGAL_Polyhedron_3.Polyhedron_3_Halfedge_around_facet_circulator",
    "CGAL.CGAL_Polyhedron_3.Polyhedron_3_Halfedge_around_vertex_circulator",
    "CGAL.CGAL_Polyhedron_3.Polyhedron_3_Halfedge_handle",
    "CGAL.CGAL_Polyhedron_3.Polyhedron_3_Vertex_handle",
    "CGAL.CGAL_Surface_mesh_skeletonization.surface_mesh_skeletonization",
}


def find_production_cgal_imports() -> set[str]:
    imports: set[str] = set()
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "CGAL."
                ):
                    imports.update(f"{node.module}.{alias.name}" for alias in node.names)
                elif isinstance(node, ast.Import):
                    imports.update(
                        alias.name
                        for alias in node.names
                        if alias.name == "CGAL" or alias.name.startswith("CGAL.")
                    )
    return imports


class TestCgalUsageManifest(unittest.TestCase):
    def test_every_production_cgal_import_has_a_compatibility_test(self) -> None:
        self.assertSetEqual(find_production_cgal_imports(), TESTED_CGAL_IMPORTS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
