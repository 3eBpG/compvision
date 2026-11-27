import argparse
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import cKDTree


@dataclass
class PlaneSegment:
    """Единый формат найденной плоскости для обоих алгоритмов."""
    normal: np.ndarray      # единичная нормаль (3,)
    d: float                # свободный член ax + by + cz + d = 0
    indices: np.ndarray     # индексы точек плоскости в предобработанном облаке
    rms: float              # среднеквадратичное отклонение точек от плоскости (в метрах)
    centroid: np.ndarray    # центр масс точек плоскости (3,)


def fit_plane_svd(points: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Точная аппроксимация плоскости методом PCA/SVD и расчёт RMS."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    norm_len = np.linalg.norm(normal)
    normal = normal / norm_len if norm_len > 0 else np.array([0.0, 0.0, 1.0])
    d = -float(np.dot(normal, centroid))
    dists = np.abs(points @ normal + d)
    rms = float(np.sqrt(np.mean(dists ** 2)))
    return normal, d, rms, centroid


def dihedral_angle_deg(n1: np.ndarray, n2: np.ndarray) -> float:
    """Острый/прямой двугранный угол между двумя плоскостями в градусах [0, 90]."""
    cos_val = np.clip(abs(float(np.dot(n1, n2))), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


class PlaneDetector(ABC):
    """Единый интерфейс алгоритмов выделения плоскостей."""

    def __init__(self, name: str, params: dict):
        self.name = name
        self.params = params

    @abstractmethod
    def detect(self, pcd: o3d.geometry.PointCloud) -> Tuple[List[PlaneSegment], float]:
        """Возвращает (список найденных плоскостей, время работы в секундах)."""
        pass


class SequentialRANSACDetector(PlaneDetector):
    """Алгоритм 1: Последовательный RANSAC."""

    def __init__(
        self,
        distance_threshold: float = 0.02,
        ransac_n: int = 3,
        num_iterations: int = 1000,
        min_points: int = 400,
        max_planes: int = 8,
    ):
        params = {
            "distance_threshold": distance_threshold,
            "ransac_n": ransac_n,
            "num_iterations": num_iterations,
            "min_points": min_points,
            "max_planes": max_planes,
        }
        super().__init__("Sequential RANSAC", params)

    def detect(self, pcd: o3d.geometry.PointCloud) -> Tuple[List[PlaneSegment], float]:
        t0 = time.perf_counter()
        all_points = np.asarray(pcd.points)
        remaining_idx = np.arange(len(all_points))
        segments: List[PlaneSegment] = []

        for _ in range(self.params["max_planes"]):
            if len(remaining_idx) < self.params["min_points"]:
                break

            sub_pcd = o3d.geometry.PointCloud()
            sub_pcd.points = o3d.utility.Vector3dVector(all_points[remaining_idx])

            _, local_inliers = sub_pcd.segment_plane(
                distance_threshold=self.params["distance_threshold"],
                ransac_n=self.params["ransac_n"],
                num_iterations=self.params["num_iterations"],
            )

            if len(local_inliers) < self.params["min_points"]:
                break

            local_inliers = np.asarray(local_inliers, dtype=int)
            global_inliers = remaining_idx[local_inliers]
            plane_pts = all_points[global_inliers]

            normal, d, rms, centroid = fit_plane_svd(plane_pts)
            segments.append(PlaneSegment(normal, d, global_inliers, rms, centroid))

            mask = np.ones(len(remaining_idx), dtype=bool)
            mask[local_inliers] = False
            remaining_idx = remaining_idx[mask]

        elapsed = time.perf_counter() - t0
        return segments, elapsed


class RegionGrowingDetector(PlaneDetector):
    """Алгоритм 2: Региональный рост по нормалям и кривизне."""

    def __init__(
        self,
        k_neighbors: int = 20,
        angle_threshold_deg: float = 8.0,
        curvature_threshold: float = 0.03,
        distance_threshold: float = 0.03,
        min_points: int = 400,
        max_planes: int = 8,
    ):
        params = {
            "k_neighbors": k_neighbors,
            "angle_threshold_deg": angle_threshold_deg,
            "curvature_threshold": curvature_threshold,
            "distance_threshold": distance_threshold,
            "min_points": min_points,
            "max_planes": max_planes,
        }
        super().__init__("Region Growing", params)

    def detect(self, pcd: o3d.geometry.PointCloud) -> Tuple[List[PlaneSegment], float]:
        t0 = time.perf_counter()
        points = np.asarray(pcd.points)
        n_pts = len(points)
        k = min(self.params["k_neighbors"], n_pts)

        # 1. Быстрый поиск k соседей и векторизованный расчёт нормалей + кривизны
        tree = cKDTree(points)
        _, neighbors_idx = tree.query(points, k=k)

        neigh_pts = points[neighbors_idx]  # (N, k, 3)
        centered = neigh_pts - neigh_pts.mean(axis=1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", centered, centered) / k
        eigvals, eigvecs = np.linalg.eigh(cov)

        normals = eigvecs[:, :, 0]  # собственный вектор для минимального собств. значения
        curvature = eigvals[:, 0] / (eigvals.sum(axis=1) + 1e-12)

        # 2. Сортируем точки по возрастанию кривизны (самые плоские участки — первые)
        order = np.argsort(curvature)
        visited = np.zeros(n_pts, dtype=bool)
        cos_thresh = np.cos(np.radians(self.params["angle_threshold_deg"]))
        curv_thresh = self.params["curvature_threshold"]
        dist_thresh = self.params["distance_threshold"]

        raw_regions: List[np.ndarray] = []

        for seed in order:
            if visited[seed]:
                continue
            if curvature[seed] > curv_thresh:
                break

            visited[seed] = True
            region = [seed]
            queue = deque([seed])
            seed_normal = normals[seed]
            seed_point = points[seed]

            while queue:
                curr = queue.popleft()
                curr_n = normals[curr]
                for nb in neighbors_idx[curr]:
                    if visited[nb]:
                        continue
                    # Проверка согласованности нормалей (локальной и с затравкой)
                    if abs(np.dot(curr_n, normals[nb])) < cos_thresh:
                        continue
                    if abs(np.dot(seed_normal, normals[nb])) < cos_thresh:
                        continue
                    # Защита от «переползания» на параллельную смещённую плоскость
                    if abs(np.dot(seed_normal, points[nb] - seed_point)) > dist_thresh:
                        continue

                    visited[nb] = True
                    region.append(nb)
                    if curvature[nb] <= curv_thresh:
                        queue.append(nb)

            if len(region) >= self.params["min_points"]:
                raw_regions.append(np.asarray(region, dtype=int))

        # 3. Сортируем найденные регионы по размеру и аппроксимируем плоскости
        raw_regions.sort(key=len, reverse=True)
        segments: List[PlaneSegment] = []
        for idx_arr in raw_regions[: self.params["max_planes"]]:
            normal, d, rms, centroid = fit_plane_svd(points[idx_arr])
            segments.append(PlaneSegment(normal, d, idx_arr, rms, centroid))

        elapsed = time.perf_counter() - t0
        return segments, elapsed


def preprocess_cloud(
    pcd: o3d.geometry.PointCloud,
    scale: float = 1.0,
    voxel_size: float = 0.02,
) -> o3d.geometry.PointCloud:
    """Масштабирование, воксельная прореживающая фильтрация и чистка выбросов."""
    cloud = o3d.geometry.PointCloud(pcd)
    if scale != 1.0:
        cloud.scale(scale, center=(0.0, 0.0, 0.0))
    if voxel_size > 0:
        cloud = cloud.voxel_down_sample(voxel_size=voxel_size)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return cloud


def evaluate_predictions(
    segments: List[PlaneSegment],
    expected_planes: int,
    gt_angles_deg: List[float],
    total_points: int,
) -> Dict[str, float]:
    """Расчёт метрик: полнота, угловая ошибка, средний RMS, покрытие точек."""
    # Полнота по числу найденных явных плоскостей (ограничена 1.0)
    recall = min(len(segments) / max(expected_planes, 1), 1.0)

    # Попарные углы между найденными плоскостями (берём топ основных плоскостей)
    top_segs = segments[: max(expected_planes, len(segments))]
    pred_angles = []
    for i in range(len(top_segs)):
        for j in range(i + 1, len(top_segs)):
            pred_angles.append(dihedral_angle_deg(top_segs[i].normal, top_segs[j].normal))

    # Сопоставление эталонных углов с ближайшими найденными (жадный матчинг без повторов)
    angle_errors = []
    available = pred_angles.copy()
    for gt_ang in gt_angles_deg:
        gt_acute = min(gt_ang, 180.0 - gt_ang)
        if not available:
            angle_errors.append(90.0)
            continue
        diffs = [abs(a - gt_acute) for a in available]
        best_i = int(np.argmin(diffs))
        angle_errors.append(diffs[best_i])
        available.pop(best_i)

    mean_angle_err = float(np.mean(angle_errors)) if angle_errors else 0.0
    mean_rms_cm = float(np.mean([s.rms for s in segments]) * 100.0) if segments else 0.0
    covered_pts = sum(len(s.indices) for s in segments)
    coverage = covered_pts / max(total_points, 1)

    return {
        "planes_found": len(segments),
        "recall": round(recall, 3),
        "angle_err_deg": round(mean_angle_err, 2),
        "rms_cm": round(mean_rms_cm, 2),
        "coverage": round(coverage, 3),
    }


def save_colored_planes(
    pcd: o3d.geometry.PointCloud,
    segments: List[PlaneSegment],
    out_path: str,
) -> None:
    """Сохраняет раскрашенное облако плоскостей для скриншотов в доклад."""
    palette = np.array([
        [0.90, 0.22, 0.21],  # красный
        [0.12, 0.53, 0.90],  # синий
        [0.26, 0.63, 0.28],  # зелёный
        [0.98, 0.75, 0.18],  # жёлтый
        [0.56, 0.14, 0.67],  # фиолетовый
        [0.00, 0.67, 0.76],  # бирюзовый
        [0.96, 0.49, 0.00],  # оранжевый
        [0.55, 0.43, 0.39],  # коричневый
    ])
    colors = np.full((len(pcd.points), 3), 0.75)  # несегментированные точки — серые
    for i, seg in enumerate(segments):
        colors[seg.indices] = palette[i % len(palette)]

    vis_pcd = o3d.geometry.PointCloud(pcd)
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, vis_pcd)


def generate_demo_scenes(out_dir: str = "demo_data") -> Dict[str, dict]:
    """Создаёт 3 синтетические сцены для мгновенной проверки пайплайна."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    def make_rect(n, u_range, v_range, axis_const, val, noise=0.005):
        u = rng.uniform(*u_range, n)
        v = rng.uniform(*v_range, n)
        c = np.full(n, val) + rng.normal(0, noise, n)
        if axis_const == "x":
            return np.column_stack([c, u, v])
        if axis_const == "y":
            return np.column_stack([u, c, v])
        return np.column_stack([u, v, c])

    # Сцена 1: Угол комнаты (2 стены + пол)
    s1 = np.vstack([
        make_rect(4000, (0, 2.5), (0, 2.6), "x", 0.0),
        make_rect(4000, (0, 2.5), (0, 2.6), "y", 0.0),
        make_rect(4000, (0, 2.5), (0, 2.5), "z", 0.0),
    ])

    # Сцена 2: Коридор (2 параллельные стены + пол + потолок)
    s2 = np.vstack([
        make_rect(4500, (0, 4.0), (0, 2.7), "x", 0.0),
        make_rect(4500, (0, 4.0), (0, 2.7), "x", 1.6),
        make_rect(4000, (0, 1.6), (0, 4.0), "z", 0.0),
        make_rect(4000, (0, 1.6), (0, 4.0), "z", 2.7),
    ])

    # Сцена 3: Помещение с мебелью (2 стены + пол + столешница + фасад шкафа + шум)
    clutter = rng.uniform([0.2, 0.2, 0.0], [2.8, 2.8, 1.5], (1200, 3))
    s3 = np.vstack([
        make_rect(4500, (0, 3.0), (0, 2.6), "x", 0.0, noise=0.008),
        make_rect(4500, (0, 3.0), (0, 2.6), "y", 0.0, noise=0.008),
        make_rect(4500, (0, 3.0), (0, 3.0), "z", 0.0, noise=0.008),
        make_rect(2000, (0.8, 2.0), (0.8, 1.6), "z", 0.75, noise=0.006),  # стол
        make_rect(2000, (0.1, 1.2), (0.0, 2.0), "y", 0.55, noise=0.006),  # шкаф
        clutter,
    ])

    scenes = {}
    configs = [
        ("1_corner", s1, 3, [90.0, 90.0, 90.0]),
        ("2_corridor", s2, 4, [0.0, 90.0, 90.0, 90.0, 90.0, 0.0]),
        ("3_room_furniture", s3, 5, [90.0, 90.0, 90.0, 0.0, 0.0]),
    ]
    for name, pts, exp_n, gt_ang in configs:
        path = os.path.join(out_dir, f"{name}.ply")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(path, pcd)
        scenes[name] = {
            "path": path,
            "scale": 1.0,
            "expected_planes": exp_n,
            "gt_angles_deg": gt_ang,
        }
    return scenes


# --- КОНФИГУРАЦИЯ ТВОИХ РЕАЛЬНЫХ СЦЕН ---
# После съёмки пропиши сюда пути к .ply/.pcd, масштаб (если COLMAP) и средние углы из ручного эталона
REAL_SCENES = {
    "1_Угол_комнаты": {
        "path": "03_clouds_ply/test1.ply",
        "scale": 1.0,  # реальная_длина_в_метрах / длина_в_облаке
        "expected_planes": 3,  # стена 1, стена 2, пол
        "gt_angles_deg": [90.0, 90.0, 90.0],  # подставь свои замеры угломером
    },
    "2_Коридор": {
        "path": "03_clouds_ply/test1.ply",
        "scale": 1.0,
        "expected_planes": 4,  # левая стена, правая стена, пол, потолок (или торец)
        "gt_angles_deg": [0.0, 90.0, 90.0, 90.0, 90.0, 0.0],
    },
    "3_Помещение_с_мебелью": {
        "path": "03_clouds_ply/room1.ply",
        "scale": 1.0,
        "expected_planes": 5,  # 2 стены, пол, стол, шкаф
        "gt_angles_deg": [90.0, 90.0, 90.0, 0.0, 0.0],
    },
}


def run_benchmark(scenes: Dict[str, dict], out_dir: str = "results") -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Зафиксированные параметры обоих алгоритмов (одинаковые для всех 3 сцен!)
    detectors: List[PlaneDetector] = [
        SequentialRANSACDetector(
            distance_threshold=0.025,
            ransac_n=3,
            num_iterations=1500,
            min_points=400,
            max_planes=8,
        ),
        RegionGrowingDetector(
            k_neighbors=20,
            angle_threshold_deg=10.0,
            curvature_threshold=0.04,
            distance_threshold=0.035,
            min_points=400,
            max_planes=8,
        ),
    ]

    rows = []
    for scene_name, cfg in scenes.items():
        if not os.path.exists(cfg["path"]):
            print(f"[ПРОПУСК] Файл не найден: {cfg['path']}")
            continue

        raw_pcd = o3d.io.read_point_cloud(cfg["path"])
        pcd = preprocess_cloud(raw_pcd, scale=cfg["scale"], voxel_size=0.02)
        print(f"\n=== Сцена: {scene_name} (точек после фильтра: {len(pcd.points)}) ===")

        for det in detectors:
            segments, elapsed = det.detect(pcd)
            metrics = evaluate_predictions(
                segments=segments,
                expected_planes=cfg["expected_planes"],
                gt_angles_deg=cfg["gt_angles_deg"],
                total_points=len(pcd.points),
            )

            out_ply = os.path.join(
                out_dir, f"{scene_name}_{det.name.replace(' ', '_')}.ply"
            )
            save_colored_planes(pcd, segments, out_ply)

            row = {
                "Сцена": scene_name,
                "Алгоритм": det.name,
                "Найдено плоскостей": f"{metrics['planes_found']} / {cfg['expected_planes']}",
                "Полнота (Recall)": metrics["recall"],
                "Угловая ошибка (°)": metrics["angle_err_deg"],
                "RMS (см)": metrics["rms_cm"],
                "Время (с)": round(elapsed, 3),
                "Покрытие точек": metrics["coverage"],
            }
            rows.append(row)
            print(
                f"  {det.name:18s} | Полнота: {metrics['recall']:.2f} | "
                f"Угол: {metrics['angle_err_deg']:.2f}° | RMS: {metrics['rms_cm']:.2f} см | "
                f"Время: {elapsed:.3f} с"
            )

    if rows:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "comparison_table.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print("\n--- Сводная таблица (сцена x алгоритм) ---")
        print(df.to_string(index=False))
        print(f"\nСохранено в {csv_path} и раскрашенные облака в папке {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Прогон на синтетических сценах")
    args = parser.parse_args()

    scenes_to_run = generate_demo_scenes() if args.demo else REAL_SCENES
    run_benchmark(scenes_to_run)