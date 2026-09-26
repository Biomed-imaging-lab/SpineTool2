from utils.project_paths import resolve_project_path
import json
import numpy as np

from projects.segmentation.utils.data_processing.result import Result
from projects.segmentation.utils.project_info import FinalSegmentationData


def finalize_spine_preview(
    mesh_v_f_vv, mesh_file, spines_files, spines_file, deleted_spines,
    folder, metadata, queue_in, queue_out,
):
    try:
        with open(resolve_project_path(folder, spines_file), encoding="utf-8") as stream:
            source = json.load(stream)
        vertices, facets, values = mesh_v_f_vv
        values = np.asarray(values).copy()
        kept = {"spines": {}, "pos_to_id": {}}
        output_spines = []
        for old_pos in range(len(source["pos_to_id"])):
            spine_id = source["pos_to_id"][str(old_pos)]
            info = source["spines"][str(spine_id)]
            if old_pos in deleted_spines:
                values[info["indices"]] = 0.0
                continue
            new_pos = len(output_spines)
            info["pos"] = new_pos
            kept["spines"][str(spine_id)] = info
            kept["pos_to_id"][str(new_pos)] = spine_id
            output_spines.append(spines_files[old_pos])
        with open(resolve_project_path(folder, spines_file), "w", encoding="utf-8") as stream:
            json.dump(kept, stream)
        data = FinalSegmentationData(
            mesh_file, (vertices, facets, values), output_spines
        )
        queue_out.put(Result(data, metadata, files={"spines": spines_file}))
    except Exception as error:
        queue_out.put(Result(FinalSegmentationData(), metadata, error=str(error)))
