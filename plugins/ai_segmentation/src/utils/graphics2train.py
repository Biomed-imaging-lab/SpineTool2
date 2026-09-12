from functools import reduce
import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_scalars_by_epoch_averaged(log_dir, tag):
    """
    Извлекает значения метрики и УСРЕДНЯЕТ их внутри каждой эпохи,
    чтобы убрать "забор" (вертикальные линии) из шагов.
    """
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0} 
    )
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        print(f"Тег {tag} не найден в {log_dir}")
        return [], []
        
    events = ea.Scalars(tag)
    
    # Пытаемся вытащить связи шагов и эпох
    step_to_epoch = {}
    if 'epoch' in ea.Tags()['scalars']:
        epoch_events = ea.Scalars('epoch')
        step_to_epoch = {e.step: e.value for e in epoch_events}
    
    # Словарь для группировки: {Эпоха: [лосс_батча1, лосс_батча2, ...]}
    epoch_values = defaultdict(list)
    
    for e in events:
        step = e.step
        if step in step_to_epoch:
            current_epoch = step_to_epoch[step]
        else:
            known_steps = [s for s in step_to_epoch.keys() if s <= step]
            current_epoch = step_to_epoch[max(known_steps)] if known_steps else 0
            
        epoch_values[current_epoch].append(e.value)
        
    # Усредняем значения для каждой эпохи
    epochs_x = sorted(epoch_values.keys())
    averaged_y = [np.mean(epoch_values[ep]) for ep in epochs_x]
    
    return epochs_x, averaged_y

def extract_scalars(log_dir, tag):
    """
    Извлекает шаги и значения конкретной метрики из логов TensorBoard.
    """
    # Загружаем лог (size_guidance нужен, чтобы загрузить все точки, а не только последние)
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0} 
    )
    ea.Reload()
    
    # 1. Проверяем, есть ли нужная метрика
    if tag not in ea.Tags()['scalars']:
        print(f"Тег {tag} не найден в {log_dir}")
        return [], []
        
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # 2. Пытаемся вытащить эпохи, чтобы заменить ими шаги по оси X
    if 'epoch' in ea.Tags()['scalars']:
        epoch_events = ea.Scalars('epoch')
        # Создаем словарь связи: шаг -> эпоха
        step_to_epoch = {e.step: e.value for e in epoch_events}
        
        epochs_x = []
        for step in steps:
            # Ищем, к какой эпохе относится этот шаг
            if step in step_to_epoch:
                epochs_x.append(step_to_epoch[step])
            else:
                # Если точного совпадения нет, берем последнюю известную эпоху
                known_steps = [s for s in step_to_epoch.keys() if s <= step]
                current_epoch = step_to_epoch[max(known_steps)] if known_steps else 0
                epochs_x.append(current_epoch)
                
        return epochs_x, values
    else:
        print("Внимание: метрика 'epoch' не найдена. Ось X останется в шагах.")
        return steps, values


def build_metrics_graph(log_dir):
    metric_tags = [
        "train/loss",
        "val/loss",
        "val/dendrite_f1",
        "val/spine_f1",
        "val/spine_iou",
        "val/dice",
        "val/iou",
    ]

    log_dir = Path(log_dir)
    file_paths = list(log_dir.rglob("events.out*"))

    metrics = []

    for file_path in file_paths:
        run_name = file_path.parent.name

        for tag in metric_tags:
            steps, values = extract_scalars(str(file_path), tag)

            if not steps:
                continue

            column_name = f"{tag}_{run_name}"

            metric_df = (
                pd.DataFrame({
                    "epoch": steps,
                    column_name: values,
                })
                .groupby("epoch", as_index=False)
                .mean()
                .set_index("epoch")
            )

            metrics.append(metric_df)

    if not metrics:
        return pd.DataFrame()

    result = (
        pd.concat(metrics, axis=1)
        .sort_index()
        .reset_index()
    )

    result.to_csv(log_dir / "metrics.csv", index=False)

    return result


def plot_cv_metrics(df, dir):
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df)

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)

    fold_colors = [
        "#00BCD4",  # cyan
        "#FBC02D",  # yellow
        "#7C4DFF",  # violet
        "#616161",  # dark grey
        "#E91E63",  # pink
    ]

    metric_tags = [
        "train/loss",
        "val/loss",
        "val/dendrite_f1",
        "val/spine_f1",
        "val/spine_iou",
        "val/dice",
        "val/iou",
    ]

    for metric in metric_tags:
        columns = sorted(
            col for col in df.columns
            if col.startswith(f"{metric}_")
        )

        if not columns:
            continue

        plt.figure(figsize=(5, 4))

        for i, column in enumerate(columns):
            plt.plot(
                df["epoch"],
                df[column],
                color=fold_colors[i],
                alpha=0.7,
                linewidth=1.5,
                label=f"Fold {i + 1}",
            )

        plt.plot(
            df["epoch"],
            df[columns].mean(axis=1),
            color="black",
            linewidth=3,
            label="Mean",
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(metric)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        filename = metric.replace("/", "_") + ".png"
        plt.savefig(dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    log_dir_1 = "C:/Users/Student/source/repos/ai_spines_segmentation/tensorboard_logs_stage4_new/stage3_model/version_0/events.out.tfevents.1773878642.IBS-HL11-LABID4.27448.0"
    log_dir_2 = "C:/Users/Student/source/repos/ai_spines_segmentation/tensorboard_logs_stage4_vsot_new/stage3_model/version_4/events.out.tfevents.1774457416.IBS-HL11-LABID4.35340.0"

    # Название метрики, которую хотите сравнить (точно так же, как она называется в TensorBoard)
    metric_tag = "val/dendrite_f1" # или "train/loss", "val/spine_f1", "val/spine_iou", "val/dendrite_f1", "val/loss"

    # Извлекаем данные
    steps1, vals1 = extract_scalars(log_dir_1, metric_tag)
    steps2, vals2 = extract_scalars(log_dir_2, metric_tag)
    # steps1, vals1 = extract_scalars_by_epoch_averaged(log_dir_1, metric_tag)
    # steps2, vals2 = extract_scalars_by_epoch_averaged(log_dir_2, metric_tag)

    plt.figure(figsize=(10, 6))

    if steps1:
        plt.plot(steps1, vals1, label='Model without vsot', color='blue', linewidth=2)
    if steps2:
        plt.plot(steps2, vals2, label='Model with vsot', color='red', linewidth=2)
    # plt.yscale('log')
    plt.title(f'{metric_tag}', fontsize=14)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Значение', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Сохраняем в картинку и показываем
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #log_dir_stage_1 = r"C:\Users\Student\datasets\CVstage1\tensorboard_logs_2\folds"
    #df = build_metrics_graph(log_dir_stage_1)
    #plot_cv_metrics(df, log_dir_stage_1)

    log_dir_stage_2 = r"C:\Users\Student\datasets\CVstage2\tensorboard_logs_2\08skel\folds"
    df = build_metrics_graph(log_dir_stage_2)
    plot_cv_metrics(df, log_dir_stage_2)

    log_dir_stage_3 = r"D:\shteinberg\CVstage3\tensorboard_logs_f_s2_08s\folds"
    df = build_metrics_graph(log_dir_stage_3)
    plot_cv_metrics(df, log_dir_stage_3)

    log_dir_stage_4 = r"C:\Users\Student\datasets\CVstage4\tensorboard_logs_gt_ch2\folds"
    df = build_metrics_graph(log_dir_stage_4)
    plot_cv_metrics(df, log_dir_stage_4)