import os
import json
import numpy as np
import kimimaro
import networkx as nx
from tifffile import imread, imwrite
from scipy.ndimage import distance_transform_edt, find_objects
from skimage.segmentation import watershed

# RES_YXZ = (500,100, 100)#(24, 24, 30)
# RES_YXZ = (24, 24, 30)

def prune_and_segment_skeleton(skel, res_zyx, min_length_nm=6000):
    """
    Фильтрует короткие ветки (шипики) и разбивает остов на сегменты.
    """
    # 1. Создаем граф из вершин и ребер скелета
    G = nx.Graph()
    G.add_edges_from(skel.edges)
    
    # Координаты вершин в нанометрах
    vertices = skel.vertices 
    # Разрываем все замкнутые кольца, которые насоздавал Kimimaro
    G = nx.minimum_spanning_tree(G)
    # 2. Ищем точки ветвления (степень > 2) и концы (степень == 1)
    while True:
        degrees = dict(G.degree())
        leaves = [n for n, d in degrees.items() if d == 1]
        junctions = set(n for n, d in degrees.items() if d > 2)
        nodes_to_remove = []
        
        for leaf in leaves:
            path = [leaf]
            current = leaf

            # Идем от кончика вглубь дендрита до первой развилки
            while True:
                neighbors = list(G.neighbors(current))
                next_nodes = [n for n in neighbors if n not in path]
                
                if not next_nodes:
                    break 
                
                next_node = next_nodes[0] 
                
                if next_node in junctions:
                    break # Уперлись в развилку ствола или крупной ветки
                
                path.append(next_node)
                current = next_node
                
            # Вычисляем физическую длину этой веточки
            length_nm = 0.0
            if len(path) > 1:
                for i in range(len(path) - 1):
                    length_nm += np.linalg.norm(vertices[path[i]] - vertices[path[i+1]])
                    
            # Плюс расстояние от последнего узла веточки до самой развилки
            if next_node in junctions:
                 length_nm += np.linalg.norm(vertices[path[-1]] - vertices[next_node])
                 
            # Если веточка короче порога (6 мкм) - отправляем в мусор
            if length_nm < min_length_nm:
                nodes_to_remove.extend(path)
                
        # Если удалять больше нечего - ствол абсолютно чист! Выходим из цикла.
        if not nodes_to_remove:
            break 
            
        # Удаляем мусор и идем на следующий круг
        G.remove_nodes_from(nodes_to_remove)
    
    # 3. РАЗДЕЛЕНИЕ НА СЕГМЕНТЫ 
    # Теперь тут остались только реальные крупные ветвления самого дендрита
    degrees = dict(G.degree())
    new_junctions = set(n for n, d in degrees.items() if d > 2)
    G.remove_nodes_from(new_junctions)
    
    segments = list(nx.connected_components(G))
    return segments, vertices

def scelete(json_path=None, base_path="../../", do_segmentation=True):
    with open(json_path, 'r') as f:
        config = json.load(f)

    def resolve(p): 
        if p is None: return None
        return os.path.normpath(os.path.join(base_path, p))
    
    for item in config["data"]:
        print(f"\n--- Обработка: {item['name']} ---")
        # 1. Загрузка исходников
        area_path = resolve(item.get('area_of_interest'))
        area = None
        if area_path and os.path.exists(area_path):
            area = imread(area_path).astype(np.uint8)
            area[area > 0] = 1    

        input_file = resolve(item.get("shaft_mask", item["general_mask"])) 
        skel_out = resolve(item["skeleton"])
        seg_out = resolve(item.get("segments_mask",""))

        #расчет масштаба из JSON [Z, Y, X] мкм -> [Z, Y, X] нм
        scale_um = item["scale"]
        res_z = scale_um[0] * 1000
        res_y = scale_um[1] * 1000
        res_x = scale_um[2] * 1000
        res_zyx = (res_z, res_y, res_x)

        max_voxel_size_nm = max(res_zyx)
        voxel_volume_nm3 = res_zyx[0] * res_zyx[1] * res_zyx[2]
        adaptive_const = max_voxel_size_nm * 3.5
        adaptive_soma_const = max_voxel_size_nm * 2.0

        
        target_dust_volume_nm3 = 5000000.0 # Объем "пылинки" в нм^3, который мы хотим удалять
        
        # Переводим физический объем в количество вокселей
        adaptive_dust = int(target_dust_volume_nm3 / voxel_volume_nm3)
        # Ограничиваем разумными пределами (например, не меньше 20 вокселей)
        adaptive_dust = max(20, adaptive_dust)

        vol = imread(input_file) 
        if area is not None: vol = vol * area

        labels = (vol > 0).astype(np.uint32)

        skels = kimimaro.skeletonize(
            labels,
            teasar_params={
                'scale': 1.5,#4
                'const': adaptive_const, #1000 physical units
                'pdrf_exponent': 4,
                'pdrf_scale': 100000,
                'soma_detection_threshold': 99999999, #1100 physical units
                'soma_acceptance_threshold': 99999999, #3500 physical units
                'soma_invalidation_scale': 1.0,
                'soma_invalidation_const': adaptive_soma_const, # physical units
                'max_paths': None, # default None
            },
            # object_ids=[ ... ], # process only the specified labels
            # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
            # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
            dust_threshold=adaptive_dust, # skip connected components with fewer than this many voxels
            anisotropy=res_zyx, # default True
            fix_branching=True, # default True
            fix_borders=True, # default True/у гришы было False
            fill_holes=False, # default False
            fix_avocados=False, # default False
            progress=True, # default False, show progress bar
            parallel=10, # <= 0 all cpu, 1 single process, 2+ multiprocess
            parallel_chunk_size=100, # how many skeletons to process before updating progress bar
        )
        print(len(skels))

        if not skels:
            print("Скелет не найден!")
            continue

        skel = list(skels.values())[0]
        # 3. СОХРАНЕНИЕ ПОЛНОГО СКЕЛЕТА (ds_skeleton)
        binimg_skel = np.zeros(labels.shape, dtype=np.uint8)
        if do_segmentation:
            print("Фильтрация шипиков < 6 мкм и разделение на сегменты...")
            segments, vertices = prune_and_segment_skeleton(skel, res_zyx, min_length_nm=6000)
            markers = np.zeros(labels.shape, dtype=np.uint16)
            for segment_id, node_set in enumerate(segments, start=1):
                for node in node_set:
                    idx = (vertices[node] / np.array(res_zyx)).astype(int)
                    if (0 <= idx[0] < markers.shape[0] and 
                        0 <= idx[1] < markers.shape[1] and 
                        0 <= idx[2] < markers.shape[2]):
                    
                    # Записываем в файл скелета (белый цвет)
                        binimg_skel[idx[0], idx[1], idx[2]] = 255 
                    
                    # Записываем уникальный ID сегмента для водораздела
                        markers[idx[0], idx[1], idx[2]] = segment_id
            imwrite(skel_out, binimg_skel, photometric='minisblack')
            print(f"Полный скелет сохранен: {skel_out}")
            print("Распределение вокселей по сегментам (Watershed)...")
        # если очень большой снимок, то памяти не хватает,выделяем только нужную часть и считаем на ней, потом вставляем обратно
#         # 1. Находим минимальную "коробочку" (Bounding Box), куда влезает весь дендрит
            slices = find_objects(labels > 0)[0] 

            # 2. Вырезаем маленькие кусочки из огромных массивов
            labels_crop = labels[slices]
            markers_crop = markers[slices]

            # 3. Считаем SciPy EDT и Watershed только на маленьком кусочке!
            distance_crop = distance_transform_edt(labels_crop > 0, sampling=res_zyx)
            seg_crop = watershed(-distance_crop, markers_crop, mask=(labels_crop > 0))

            # 4. Создаем пустой черный объем оригинального размера и вставляем туда результат
            segmented_volume = np.zeros(labels.shape, dtype=np.uint16)
            segmented_volume[slices] = seg_crop

            imwrite(seg_out, segmented_volume.astype(np.uint16), photometric='minisblack')
            print(f"Готово! Результат сохранен в {seg_out}")
        else:
            print("Сохранение сырого полного скелета (без сегментации)...")
            verts_all = (skel.vertices / np.array(res_zyx)).astype(int)
            for v in verts_all:
                if (0 <= v[0] < binimg_skel.shape[0] and 
                    0 <= v[1] < binimg_skel.shape[1] and 
                    0 <= v[2] < binimg_skel.shape[2]):
                    binimg_skel[v[0], v[1], v[2]] = 255
            imwrite(skel_out, binimg_skel, photometric='minisblack')

            #analazy
            G_test = nx.Graph()
            G_test.add_edges_from(skel.edges)
        
        # 2. Считаем количество независимых "кусков" (компонент связности)
            num_components = nx.number_connected_components(G_test)
        
            print(f"Всего вершин (вокселей) в скелете: {G_test.number_of_nodes()}")
            print(f"Количество независимых кусков: {num_components}")
        
            if num_components == 1:
                print("==> СКЕЛЕТ ИДЕАЛЬНО ЦЕЛЫЙ")
            else:
                print("==> ВНИМАНИЕ: Скелет порван")
            print(f"Полный скелет сохранен: {skel_out}")        

if __name__ == "__main__":
    # #проверка получившегося скелета
    # skel = imread("C:/Users/Student/datasets/in_vitro/9/necks.tif").astype(np.uint8)
    # print(skel.shape)
    # num_voxels = np.count_nonzero(skel)

    # print(f"Количество положительных вокселов: {num_voxels}")
    # max_val = skel.max()

    # if max_val > 0:
    #     print(f"Снимок не пустой. Максимальная яркость: {max_val}")
    # else:
    #     print("Снимок полностью черный.")
    # # skel = np.max(skel, axis=2)
    # skel[skel>0]=255
    # imwrite("skel_vitronecks9.tif", data=np.max(skel, axis=0)) 
    # scelete("data_preprocessing/CVstage1/9009_patch.json")
    #for stage2:
    scelete("data_preprocessing/CVstage2/folds1.json", do_segmentation=False)
    # scelete("data_preprocessing/description_stage4_vsot.json")#запускать без разбиения на сегменты, так как там и так один сегмент
