import json
import os
from pathlib import Path
from queue import Queue
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from tifffile import imwrite

from projects.neural_segmentation.utils.data_processing import plugin_inference
from projects.segmentation.utils.data_processing.result import Result


class PluginPathTests(unittest.TestCase):
    def test_absolute_paths_do_not_require_project_root(self):
        for value in ('/tmp/probability.tif', 'C:\\project\\probability.tif',
                      '\\\\server\\share\\probability.tif'):
            with self.subTest(value=value):
                self.assertEqual(
                    plugin_inference._resolve_project_path({'path': value}, 'path'),
                    os.path.normpath(value),
                )

    def test_relative_path_uses_project_root(self):
        self.assertEqual(
            plugin_inference._resolve_project_path(
                {'project_root': '/tmp/project', 'path': 'artifacts/probability.tif'},
                'path',
            ),
            os.path.abspath('/tmp/project/artifacts/probability.tif'),
        )

    def test_worker_reads_successful_plugin_output(self):
        # Exercise response handling through label generation without rerunning a model.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            volume = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
                              dtype=np.float32).reshape(2, 2, 2)
            probability = root / 'probability.tif'
            imwrite(probability, volume, photometric='minisblack')
            for produced_path in (str(probability), 'probability.tif'):
                with self.subTest(produced_path=produced_path):
                    response = root / 'response.json'
                    response.write_text(json.dumps({
                        'status': 'ok', 'output_probability_path': produced_path,
                    }))
                    output = Queue()
                    with patch.object(plugin_inference, '_run_plugin_process',
                                      return_value=(0, '', False)):
                        plugin_inference.run_ai_spines_inference(
                            'python', 'plugin.py', {'project_root': folder, 'stage': 1},
                            str(response), str(probability), {}, Queue(), output,
                            expected_shape=volume.shape,
                        )
                    results = []
                    while not output.empty():
                        item = output.get_nowait()
                        if isinstance(item, Result):
                            results.append(item)
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0].error, '')
                    self.assertEqual(np.count_nonzero(results[0].data), 4)


if __name__ == '__main__':
    unittest.main()
