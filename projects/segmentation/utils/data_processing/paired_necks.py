import heapq
from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


@dataclass
class NeckValidationResult:
    """Результат проверки маски восстановленной шеи."""

    ok: bool
    warnings: list[str]
    touched_labels: list[int]


def _as_zyx_point(point: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Преобразует точку к целочисленным координатам ZYX и проверяет границы.

    Args:
        point: Координаты точки в порядке Z, Y, X.
        shape: Размер трехмерного изображения в порядке Z, Y, X.

    Returns:
        Проверенная точка типа ``int64``.

    Raises:
        ValueError: Если точка имеет неверную форму или находится вне изображения.
    """
    p = np.asarray(point, dtype=np.int64)
    if p.shape != (3,):
        raise ValueError(f"Point must have shape (3,), got {p.shape}")
    if np.any(p < 0) or np.any(p >= np.asarray(shape)):
        raise ValueError(f"Point {p.tolist()} is outside image shape {shape}")
    return p


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Маркирует 26-связные компоненты трехмерной бинарной маски.

    Args:
        mask: Трехмерная маска объекта.

    Returns:
        Массив меток компонент и их количество.
    """
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    return ndi.label(mask.astype(bool), structure=structure)


def _largest_component_label(labels: np.ndarray) -> int | None:
    """Возвращает метку крупнейшей ненулевой связной компоненты.

    Args:
        labels: Массив меток связных компонент.

    Returns:
        Метка крупнейшей компоненты или ``None``, если компонент нет.
    """
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        return None
    counts[0] = 0
    return int(np.argmax(counts))


def snap_to_mask(
    point_zyx: np.ndarray,
    mask: np.ndarray,
    scale_zyx: np.ndarray,
    max_distance_um: float | None = None,
) -> np.ndarray:
    """Привязывает точку к ближайшему вокселю заданной маски.

    Расстояния вычисляются в физических координатах с учетом масштаба по
    осям Z, Y и X.

    Args:
        point_zyx: Исходная точка в координатах ZYX.
        mask: Маска допустимых вокселей.
        scale_zyx: Физический размер вокселя по осям Z, Y и X.
        max_distance_um: Максимально допустимое расстояние до маски в мкм.

    Returns:
        Координаты ближайшего допустимого вокселя.

    Raises:
        ValueError: Если маска пуста или ближайший воксель находится дальше
            допустимого расстояния.
    """
    point_zyx = _as_zyx_point(point_zyx, mask.shape)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise ValueError("Cannot snap point: target mask is empty")

    tree = cKDTree(coords * scale_zyx)
    dist, idx = tree.query(point_zyx * scale_zyx, k=1)

    if max_distance_um is not None and dist > max_distance_um:
        raise ValueError(
            f"Nearest valid voxel is too far: {dist:.4f} um > {max_distance_um:.4f} um"
        )

    return coords[int(idx)].astype(np.uint16)


def snap_to_valid_spine_point(
    point_zyx: np.ndarray,
    binarization: np.ndarray,
    scale_zyx: Iterable[float],
    max_distance_um: float | None = None,
) -> tuple[np.ndarray, int]:
    """Привязывает точку шипика к ближайшей отсоединенной компоненте.

    Крупнейшая связная компонента бинаризации считается стволом, а остальные
    ненулевые компоненты рассматриваются как отсоединенные шипики.

    Args:
        point_zyx: Исходная точка шипика в координатах ZYX.
        binarization: Текущая бинарная маска дендрита.
        scale_zyx: Физический размер вокселя по осям Z, Y и X.
        max_distance_um: Максимально допустимое расстояние привязки в мкм.

    Returns:
        Привязанная точка и метка выбранной компоненты шипика.

    Raises:
        ValueError: Если бинаризация не содержит компонент или привязка
            невозможна.
    """
    scale_zyx = np.asarray(scale_zyx, dtype=float)
    labels, _ = _components(binarization > 0)
    shaft_label = _largest_component_label(labels)
    if shaft_label is None:
        raise ValueError("No components found in binarization")

    detached_mask = (labels > 0) & (labels != shaft_label)
    snapped = snap_to_mask(point_zyx, detached_mask, scale_zyx, max_distance_um)
    component_id = int(labels[tuple(snapped)])
    return snapped, component_id


def snap_to_valid_shaft_point(
    point_zyx: np.ndarray,
    binarization: np.ndarray,
    scale_zyx: Iterable[float],
    max_distance_um: float | None = None,
) -> tuple[np.ndarray, int]:
    """Привязывает точку крепления к ближайшему вокселю ствола.

    Крупнейшая связная компонента бинаризации используется как маска ствола.

    Args:
        point_zyx: Исходная точка крепления в координатах ZYX.
        binarization: Текущая бинарная маска дендрита.
        scale_zyx: Физический размер вокселя по осям Z, Y и X.
        max_distance_um: Максимально допустимое расстояние привязки в мкм.

    Returns:
        Привязанная точка и метка компоненты ствола.

    Raises:
        ValueError: Если компонент ствола не найден или привязка невозможна.
    """
    scale_zyx = np.asarray(scale_zyx, dtype=float)
    labels, _ = _components(binarization > 0)
    shaft_label = _largest_component_label(labels)
    if shaft_label is None:
        raise ValueError("No shaft component found in binarization")

    shaft_mask = labels == shaft_label
    snapped = snap_to_mask(point_zyx, shaft_mask, scale_zyx, max_distance_um)
    return snapped, shaft_label


def build_local_roi(
    spine_point_zyx: np.ndarray,
    shaft_point_zyx: np.ndarray,
    image_shape: tuple[int, int, int],
    scale_zyx: Iterable[float],
    roi_margin_voxels: int | Iterable[int] = 8,
    roi_margin_um: float | None = None,
) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    """Строит локальный ROI вокруг пары точек восстановления шеи.

    ROI представляет собой ограничивающий параллелепипед между точкой шипика
    и точкой ствола с дополнительным отступом и обрезается по границам
    изображения. Если задан ``roi_margin_um``, он имеет приоритет над отступом
    в вокселях.

    Args:
        spine_point_zyx: Точка на отсоединенном шипике.
        shaft_point_zyx: Точка крепления на стволе.
        image_shape: Размер исходного изображения в порядке ZYX.
        scale_zyx: Физический размер вокселя по осям Z, Y и X.
        roi_margin_voxels: Отступ ROI в вокселях.
        roi_margin_um: Отступ ROI в мкм.

    Returns:
        Срезы локального ROI и координаты его начала в исходном изображении.
    """
    scale_zyx = np.asarray(scale_zyx, dtype=float)
    p0 = _as_zyx_point(spine_point_zyx, image_shape)
    p1 = _as_zyx_point(shaft_point_zyx, image_shape)

    if roi_margin_um is not None:
        margin = np.ceil(roi_margin_um / scale_zyx).astype(np.int64)
    else:
        margin = np.asarray(roi_margin_voxels, dtype=np.int64)
        if margin.shape == ():
            margin = np.repeat(margin, 3)

    lo = np.maximum(np.minimum(p0, p1) - margin, 0)
    hi = np.minimum(np.maximum(p0, p1) + margin + 1, np.asarray(image_shape))

    slices = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    return slices, lo.astype(np.int64)


def _corridor_mask(
    roi_shape: tuple[int, int, int],
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    scale_zyx: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    """Создает маску коридора вокруг отрезка между двумя точками.

    Расстояние от каждого вокселя ROI до отрезка вычисляется в физических
    координатах. В маску входят воксели не дальше ``radius_um`` от отрезка.

    Args:
        roi_shape: Размер локального ROI в порядке ZYX.
        start_zyx: Начальная точка внутри ROI.
        target_zyx: Конечная точка внутри ROI.
        scale_zyx: Физический размер вокселя по осям Z, Y и X.
        radius_um: Радиус допустимого коридора в мкм.

    Returns:
        Булева маска допустимого коридора.
    """
    coords = np.indices(roi_shape).reshape(3, -1).T
    a = start_zyx.astype(float) * scale_zyx
    b = target_zyx.astype(float) * scale_zyx
    x = coords.astype(float) * scale_zyx

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        d = np.linalg.norm(x - a, axis=1)
    else:
        t = np.clip(((x - a) @ ab) / denom, 0.0, 1.0)
        closest = a + t[:, None] * ab
        d = np.linalg.norm(x - closest, axis=1)

    return (d <= radius_um).reshape(roi_shape)


def shortest_path_between_points(
    image_roi: np.ndarray,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    scale_zyx: Iterable[float],
    allowed_mask: np.ndarray | None = None,
    forbidden_mask: np.ndarray | None = None,
    intensity_factor: float = 1.0,
    corridor_radius_um: float | None = None,
) -> np.ndarray:
    """Ищет взвешенный путь между двумя фиксированными точками в ROI.

    Поиск выполняется алгоритмом A* по 26-связному графу вокселей. Стоимость
    перехода учитывает физическую длину ребра и интенсивность конечного
    вокселя: путь через более яркие области имеет меньший штраф. Допустимая
    область может дополнительно ограничиваться масками и коридором между
    точками.

    Args:
        image_roi: Интенсивности изображения внутри локального ROI.
        start_zyx: Начальная точка в локальных координатах ROI.
        target_zyx: Конечная точка в локальных координатах ROI.
        scale_zyx: Физический размер вокселя по осям Z, Y и X.
        allowed_mask: Маска вокселей, разрешенных для поиска.
        forbidden_mask: Маска вокселей, запрещенных для поиска.
        intensity_factor: Вес штрафа за низкую интенсивность.
        corridor_radius_um: Радиус коридора поиска между точками в мкм.

    Returns:
        Последовательность координат пути в локальной системе ROI.

    Raises:
        ValueError: Если между точками нет допустимого пути.
    """
    scale_zyx = np.asarray(scale_zyx, dtype=float)
    shape = image_roi.shape
    start = _as_zyx_point(start_zyx, shape).astype(np.int64)
    target = _as_zyx_point(target_zyx, shape).astype(np.int64)

    if allowed_mask is None:
        allowed = np.ones(shape, dtype=bool)
    else:
        allowed = allowed_mask.astype(bool).copy()

    if forbidden_mask is not None:
        allowed &= ~forbidden_mask.astype(bool)

    if corridor_radius_um is not None and corridor_radius_um > 0:
        allowed &= _corridor_mask(shape, start, target, scale_zyx, corridor_radius_um)

    allowed[tuple(start)] = True
    allowed[tuple(target)] = True

    img = image_roi.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)

    neighbors = [
        np.asarray(d, dtype=np.int64)
        for d in product((-1, 0, 1), repeat=3)
        if d != (0, 0, 0)
    ]

    def heuristic(p: np.ndarray) -> float:
        return float(np.linalg.norm((target - p) * scale_zyx))

    def edge_cost(p: np.ndarray, q: np.ndarray) -> float:
        edge_len = float(np.linalg.norm((q - p) * scale_zyx))
        intensity_cost = 1.0 + intensity_factor * (1.0 - img[tuple(q)])
        return edge_len * intensity_cost

    start_t = tuple(start)
    target_t = tuple(target)

    dist = {start_t: 0.0}
    parent: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    heap = [(heuristic(start), 0.0, start_t)]

    while heap:
        _, cur_cost, cur_t = heapq.heappop(heap)
        if cur_t == target_t:
            break

        cur = np.asarray(cur_t, dtype=np.int64)
        if cur_cost > dist.get(cur_t, np.inf):
            continue

        for d in neighbors:
            nxt = cur + d
            if np.any(nxt < 0) or np.any(nxt >= np.asarray(shape)):
                continue
            nxt_t = tuple(nxt)
            if not allowed[nxt_t]:
                continue

            new_cost = cur_cost + edge_cost(cur, nxt)
            if new_cost < dist.get(nxt_t, np.inf):
                dist[nxt_t] = new_cost
                parent[nxt_t] = cur_t
                heapq.heappush(heap, (new_cost + heuristic(nxt), new_cost, nxt_t))

    if target_t not in parent and start_t != target_t:
        raise ValueError("No valid path between spine_point and shaft_point")

    path = [target_t]
    while path[-1] != start_t:
        path.append(parent[path[-1]])
    path.reverse()

    return np.asarray(path, dtype=np.uint16)


def path_to_neck_mask(
    path_zyx: np.ndarray,
    image_shape: tuple[int, int, int],
    scale_zyx: Iterable[float],
    radius_voxels: int = 2,
    radius_um: float | None = None,
) -> np.ndarray:
    """Расширяет путь до объемной маски шеи заданного радиуса.

    Args:
        path_zyx: Координаты вокселей пути в глобальной системе ZYX.
        image_shape: Размер выходной маски.
        scale_zyx: Физический размер вокселя по осям Z, Y и X.
        radius_voxels: Радиус расширения в вокселях.
        radius_um: Радиус расширения в мкм; при задании имеет приоритет.

    Returns:
        Булева маска восстановленной шеи.
    """
    scale_zyx = np.asarray(scale_zyx, dtype=float)
    mask = np.zeros(image_shape, dtype=bool)

    if radius_um is not None:
        r_vox = np.ceil(radius_um / scale_zyx).astype(int)
    else:
        r_vox = np.repeat(int(radius_voxels), 3)

    offsets = []
    for dz in range(-r_vox[0], r_vox[0] + 1):
        for dy in range(-r_vox[1], r_vox[1] + 1):
            for dx in range(-r_vox[2], r_vox[2] + 1):
                d = np.asarray([dz, dy, dx])
                if radius_um is None:
                    ok = np.linalg.norm(d) <= radius_voxels
                else:
                    ok = np.linalg.norm(d * scale_zyx) <= radius_um
                if ok:
                    offsets.append(d)

    shape = np.asarray(image_shape)
    for p in path_zyx.astype(np.int64):
        for d in offsets:
            q = p + d
            if np.all(q >= 0) and np.all(q < shape):
                mask[tuple(q)] = True

    return mask


def validate_neck_mask(
    binarization: np.ndarray,
    neck_mask: np.ndarray,
    spine_component_id: int | None,
    shaft_component_id: int | None,
) -> NeckValidationResult:
    """Проверяет, какие компоненты бинаризации пересекает новая шея.

    Корректная маска должна касаться выбранного шипика и ствола и не должна
    пересекать посторонние компоненты. Проверка формирует предупреждения, но
    не изменяет исходную бинаризацию или preview.

    Args:
        binarization: Исходная бинарная маска дендрита.
        neck_mask: Маска восстановленной шеи.
        spine_component_id: Метка выбранной компоненты шипика.
        shaft_component_id: Метка компоненты ствола.

    Returns:
        Результат проверки со списком предупреждений и затронутых меток.
    """
    warnings: list[str] = []

    labels, _ = _components(binarization > 0)
    touched = sorted(set(labels[neck_mask & (labels > 0)].astype(int).tolist()))

    allowed = {0}
    if spine_component_id is not None:
        allowed.add(int(spine_component_id))
    if shaft_component_id is not None:
        allowed.add(int(shaft_component_id))

    extra = [lbl for lbl in touched if lbl not in allowed]
    if extra:
        warnings.append(f"Neck touches unrelated components: {extra}")

    if spine_component_id is not None and int(spine_component_id) not in touched:
        warnings.append("Neck does not touch selected detached spine component")

    if shaft_component_id is not None and int(shaft_component_id) not in touched:
        warnings.append("Neck does not touch shaft component")

    return NeckValidationResult(
        ok=(len(warnings) == 0), warnings=warnings, touched_labels=touched
    )


def apply_neck_preview(
    binarization: np.ndarray,
    neck_mask: np.ndarray,
    neck_label: int = 2,
) -> np.ndarray:
    """Добавляет восстановленную шею в копию бинаризации для preview.

    Метка шеи записывается только в фоновые воксели, поэтому исходная
    сегментация и существующие метки не изменяются.

    Args:
        binarization: Текущая бинаризация дендрита.
        neck_mask: Маска восстановленной шеи.
        neck_label: Значение метки для добавленных вокселей.

    Returns:
        Новая маска preview с добавленной шеей.
    """
    out = binarization.copy()
    add = neck_mask & (out == 0)
    out[add] = neck_label
    return out
