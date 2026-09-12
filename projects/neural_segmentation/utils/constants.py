TYPE = "neural_segmentation"

LAYERS_PATH = "/layers"
ARTIFACTS_PATH = "/artifacts"
MODEL_RUNS_PATH = ARTIFACTS_PATH + "/model_runs"
CORRECTIONS_PATH = ARTIFACTS_PATH + "/corrections"
PLUGIN_RUNTIME_PATH = ARTIFACTS_PATH + "/plugin_runtime"
RAW_IMAGE_PATH = LAYERS_PATH + "/raw.tif"

STAGE_STATUS_NOT_STARTED = "not_started"
STAGE_STATUS_CONFIGURED = "configured"
STAGE_STATUS_READY = "ready"
STAGE_STATUS_ACCEPTED = "accepted"
STAGE_STATUS_CORRECTED = "corrected"

STAGE_STATUS_ORDER = (
    STAGE_STATUS_NOT_STARTED,
    STAGE_STATUS_CONFIGURED,
    STAGE_STATUS_READY,
    STAGE_STATUS_ACCEPTED,
    STAGE_STATUS_CORRECTED,
)

STAGE_STATUS_LABELS = {
    STAGE_STATUS_NOT_STARTED: "Not started",
    STAGE_STATUS_CONFIGURED: "Configured",
    STAGE_STATUS_READY: "Ready for review",
    STAGE_STATUS_ACCEPTED: "Accepted",
    STAGE_STATUS_CORRECTED: "Corrected",
}

NEURAL_CASCADE_STAGES = (
    {
        "stage_id": 1,
        "key": "stage_1",
        "name": "Stage 1. Dendrite segmentation (standardized XY)",
        "description": (
            "Input: raw image. Preprocess: rescale to standardized XY, patching, channels. "
            "Output: probability volume 0..1."
        ),
        "depends_on": (),
    },
    {
        "stage_id": 2,
        "key": "stage_2",
        "name": "Stage 2. Segmentation refinement for weak neck masks",
        "description": (
            "Input: raw image + stage 1 output. Preprocess: rescale, patching, channels. "
            "Output: probability volume 0..1."
        ),
        "depends_on": (1,),
    },
    {
        "stage_id": 3,
        "key": "stage_3",
        "name": "Stage 3. Return mask to original image resolution",
        "description": (
            "Input: raw image + stage 1 + stage 2 outputs. "
            "Preprocess: scale to original resolution, patching, channels. "
            "Output: probability volume 0..1 at original scale."
        ),
        "depends_on": (1, 2),
    },
    {
        "stage_id": 4,
        "key": "stage_4",
        "name": "Stage 4. Dendritic spines segmentation from binary image",
        "description": (
            "Input: pre-segmentation from external stem/spines tool for each dendrite branch. "
            "Output: probability volume 0..1 at original scale, then thresholds for stem/spines."
        ),
        "depends_on": (3,),
    },
)

DEFAULT_USER_CORRECTIONS_ENTRY = {
    "stage_id": 0,
    "path": "",
    "created_at": "",
}

EMPTY_PROJECT = {
    "name": "",
    "type": TYPE,
    "original_image": "",
    "image_path": RAW_IMAGE_PATH,
    "original_shape": (1, 1, 1),
    "shape": (1, 1, 1),
    "displayed_scale": [1.0, 1.0, 1.0],
    "real_scale": [1.0, 1.0, 1.0],
    "scale": [1.0, 1.0, 1.0],
    "stage_results": [],
    "user_corrections": [],
    "metadata": {},
}
