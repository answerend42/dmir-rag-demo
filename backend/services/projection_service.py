"""! @file projection_service.py
@brief 嵌入向量二维或三维投影服务。
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


PLOT_METHODS = {
    "tsne": {
        "label": "t-SNE",
        "requires_query": False,
    },
    "pca": {
        "label": "PCA",
        "requires_query": False,
    },
}

_EPSILON = 1e-12
_TSNE_RANDOM_STATE = 42
_DEFAULT_TARGET_DIMENSIONS = 3
_SUPPORTED_TARGET_DIMENSIONS = {2, 3}
_PLOT_AXIS_NAMES = ["x", "y", "z"]


class VectorProjectionService:
    """! @brief 统一在后端完成 embedding 降维和坐标归一化。"""

    @classmethod
    def project_embeddings(
        cls,
        embeddings: List[Dict[str, Any]],
        method: str = "tsne",
        overlays: List[Dict[str, Any]] | None = None,
        source_id: str = "",
        target_dimensions: int = _DEFAULT_TARGET_DIMENSIONS,
    ) -> Dict[str, Any]:
        """! @brief 将向量条目投影到二维或三维坐标。
        @param embeddings 包含 embedding 和 metadata 的条目列表。
        @param method 投影方法，当前支持 tsne、pca。
        @param overlays 需要放入同一坐标系的附加向量，例如查询向量。
        @param source_id 来源标识，用于前端展示和调试。
        @param target_dimensions 输出维度，只支持 2 或 3。
        @return 含投影点位、坐标轴说明和方法元信息的响应。
        """
        selected_method = str(method or "tsne").strip().lower()
        if selected_method not in PLOT_METHODS:
            raise ValueError(f"不支持的投影方法: {method}")
        selected_dimensions = cls._normalize_target_dimensions(target_dimensions)

        usable_embeddings, matrix, dimension = cls._extract_matrix(embeddings)
        clean_overlays, overlay_matrix = cls._extract_overlays(overlays or [], dimension)
        if matrix.size == 0 or dimension == 0:
            return cls._empty_projection(selected_method, source_id, selected_dimensions)

        if selected_method == "tsne":
            raw_points, raw_overlays, axes = cls._project_tsne(
                usable_embeddings,
                matrix,
                clean_overlays,
                overlay_matrix,
                selected_dimensions,
            )
        else:
            raw_points, raw_overlays, axes = cls._project_pca(
                usable_embeddings,
                matrix,
                clean_overlays,
                overlay_matrix,
                selected_dimensions,
            )

        points, normalized_overlays = cls._normalize_plot(raw_points, raw_overlays)
        return {
            "source_id": source_id,
            "method": selected_method,
            "method_label": PLOT_METHODS[selected_method]["label"],
            "available_methods": cls._available_methods(),
            "target_dimensions": selected_dimensions,
            "available_dimensions": [3, 2],
            "dimension": dimension,
            "count": len(points),
            "axes": axes,
            "points": points,
            "overlays": normalized_overlays,
        }

    @staticmethod
    def _empty_projection(method: str, source_id: str, target_dimensions: int) -> Dict[str, Any]:
        """! @brief 生成空数据响应。"""
        return {
            "source_id": source_id,
            "method": method,
            "method_label": PLOT_METHODS[method]["label"],
            "available_methods": VectorProjectionService._available_methods(),
            "target_dimensions": target_dimensions,
            "available_dimensions": [3, 2],
            "dimension": 0,
            "count": 0,
            "axes": {
                "x": {"label": "x", "explained_variance": None},
                "y": {"label": "y", "explained_variance": None},
                **({"z": {"label": "z", "explained_variance": None}} if target_dimensions == 3 else {}),
            },
            "points": [],
            "overlays": [],
        }

    @staticmethod
    def _normalize_target_dimensions(target_dimensions: int) -> int:
        """! @brief 校验并归一化前端请求的投影输出维度。"""
        try:
            selected_dimensions = int(target_dimensions)
        except (TypeError, ValueError) as exc:
            raise ValueError("投影维度只支持 2 或 3") from exc
        if selected_dimensions not in _SUPPORTED_TARGET_DIMENSIONS:
            raise ValueError("投影维度只支持 2 或 3")
        return selected_dimensions

    @staticmethod
    def _available_methods() -> List[Dict[str, Any]]:
        """! @brief 返回前端可选择的后端投影方法。"""
        return [
            {"id": key, **value}
            for key, value in PLOT_METHODS.items()
        ]

    @staticmethod
    def _extract_matrix(embeddings: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
        """! @brief 从响应条目中提取等长浮点矩阵。"""
        usable = [
            item
            for item in embeddings
            if isinstance(item, dict) and isinstance(item.get("embedding"), (list, tuple)) and item.get("embedding")
        ]
        if not usable:
            return [], np.empty((0, 0), dtype=float), 0

        dimension = min(len(item["embedding"]) for item in usable)
        matrix = np.array(
            [
                [float(value) if np.isfinite(float(value)) else 0.0 for value in item["embedding"][:dimension]]
                for item in usable
            ],
            dtype=float,
        )
        return usable, matrix, dimension

    @staticmethod
    def _extract_overlays(
        overlays: List[Dict[str, Any]],
        dimension: int,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """! @brief 读取附加向量，并去掉响应中不需要回传的大数组。"""
        if dimension <= 0:
            return [], np.empty((0, 0), dtype=float)

        clean_items: List[Dict[str, Any]] = []
        vectors: List[List[float]] = []
        for overlay in overlays:
            if not isinstance(overlay, dict):
                continue
            vector = overlay.get("vector") or overlay.get("embedding")
            if not isinstance(vector, (list, tuple)) or not vector:
                continue
            values = []
            for value in vector[:dimension]:
                numeric = float(value)
                values.append(numeric if np.isfinite(numeric) else 0.0)
            if len(values) < dimension:
                continue
            clean_items.append({
                key: value
                for key, value in overlay.items()
                if key not in {"vector", "embedding"}
            })
            clean_items[-1]["vector_dimension"] = dimension
            vectors.append(values)

        if not vectors:
            return [], np.empty((0, dimension), dtype=float)
        return clean_items, np.array(vectors, dtype=float)

    @classmethod
    def _project_tsne(
        cls,
        embeddings: List[Dict[str, Any]],
        matrix: np.ndarray,
        overlays: List[Dict[str, Any]],
        overlay_matrix: np.ndarray,
        target_dimensions: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """! @brief 使用 t-SNE 展示高维向量的局部邻域结构。"""
        combined_matrix = np.vstack([matrix, overlay_matrix]) if overlay_matrix.size else matrix
        if combined_matrix.shape[0] < 3:
            raise ValueError("t-SNE 至少需要 3 个向量点。")

        from sklearn.manifold import TSNE

        coords = TSNE(
            n_components=target_dimensions,
            perplexity=cls._tsne_perplexity(combined_matrix.shape[0]),
            learning_rate="auto",
            init="pca" if min(combined_matrix.shape) >= target_dimensions else "random",
            metric="euclidean",
            max_iter=750,
            random_state=_TSNE_RANDOM_STATE,
        ).fit_transform(combined_matrix)
        point_count = len(embeddings)
        point_coords = coords[:point_count]
        overlay_coords = coords[point_count:] if overlay_matrix.size else np.empty((0, target_dimensions))
        axes = cls._tsne_axes(target_dimensions)
        return cls._pack_points(embeddings, point_coords), cls._pack_overlays(overlays, overlay_coords), axes

    @staticmethod
    def _tsne_perplexity(sample_count: int) -> float:
        """! @brief 根据样本量选择合法且稳定的 t-SNE perplexity。"""
        return max(1.0, min(30.0, (sample_count - 1) / 3))

    @classmethod
    def _project_pca(
        cls,
        embeddings: List[Dict[str, Any]],
        matrix: np.ndarray,
        overlays: List[Dict[str, Any]],
        overlay_matrix: np.ndarray,
        target_dimensions: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """! @brief 使用 SVD 计算 PCA 投影。"""
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        components, explained = cls._pca_components(centered, target_dimensions)
        point_coords = centered @ components.T
        overlay_coords = (
            (overlay_matrix - mean) @ components.T
            if overlay_matrix.size
            else np.empty((0, target_dimensions))
        )
        axes = cls._pca_axes(explained, target_dimensions)
        return cls._pack_points(embeddings, point_coords), cls._pack_overlays(overlays, overlay_coords), axes

    @staticmethod
    def _pca_components(centered: np.ndarray, target_dimensions: int) -> Tuple[np.ndarray, List[float]]:
        """! @brief 返回 PCA 分量和解释方差占比。"""
        dimension = centered.shape[1] if centered.ndim == 2 else 0
        components = np.zeros((target_dimensions, dimension), dtype=float)
        if centered.shape[0] < 2 or dimension == 0:
            return components, [0.0 for _ in range(target_dimensions)]

        _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
        component_count = min(target_dimensions, vectors.shape[0])
        components[:component_count] = vectors[:component_count]
        variances = (singular_values ** 2) / max(centered.shape[0] - 1, 1)
        total_variance = float(variances.sum())
        explained = [
            float(variances[index] / total_variance) if index < len(variances) and total_variance > _EPSILON else 0.0
            for index in range(target_dimensions)
        ]
        return components, explained

    @staticmethod
    def _tsne_axes(target_dimensions: int) -> Dict[str, Any]:
        """! @brief 生成 t-SNE 坐标轴说明。"""
        return {
            axis_name: {
                "label": f"t-SNE-{index + 1}",
                "explained_variance": None,
                "description": (
                    "t-SNE 坐标只用于展示局部邻域结构，不参与检索排序。"
                    if index == 0
                    else "TopK 排名仍来自 Chroma 检索结果。"
                ),
            }
            for index, axis_name in enumerate(_PLOT_AXIS_NAMES[:target_dimensions])
        }

    @staticmethod
    def _pca_axes(explained: List[float], target_dimensions: int) -> Dict[str, Any]:
        """! @brief 生成 PCA 坐标轴说明。"""
        return {
            axis_name: {
                "label": f"PC{index + 1}",
                "explained_variance": VectorProjectionService._json_float(explained[index]),
            }
            for index, axis_name in enumerate(_PLOT_AXIS_NAMES[:target_dimensions])
        }

    @staticmethod
    def _pack_points(embeddings: List[Dict[str, Any]], coords: np.ndarray) -> List[Dict[str, Any]]:
        """! @brief 将点坐标和原始条目合并。"""
        axis_names = _PLOT_AXIS_NAMES[:coords.shape[1]]
        points = []
        for index, embedding in enumerate(embeddings):
            point = {
                "index": index,
                "embedding": embedding,
            }
            for axis_index, axis_name in enumerate(axis_names):
                point[f"raw_{axis_name}"] = VectorProjectionService._json_float(coords[index, axis_index])
            points.append(point)
        return points

    @staticmethod
    def _pack_overlays(overlays: List[Dict[str, Any]], coords: np.ndarray) -> List[Dict[str, Any]]:
        """! @brief 将 overlay 坐标和元信息合并。"""
        axis_names = _PLOT_AXIS_NAMES[:coords.shape[1]]
        packed_overlays = []
        for index, overlay in enumerate(overlays):
            packed_overlay = {**overlay}
            for axis_index, axis_name in enumerate(axis_names):
                packed_overlay[f"raw_{axis_name}"] = VectorProjectionService._json_float(coords[index, axis_index])
            packed_overlays.append(packed_overlay)
        return packed_overlays

    @staticmethod
    def _normalize_plot(
        points: List[Dict[str, Any]],
        overlays: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """! @brief 将原始坐标归一化到 0 到 1，便于前端渲染。"""
        all_items = points + overlays
        if not all_items:
            return points, overlays

        axis_stats = {}
        for axis_name in _PLOT_AXIS_NAMES:
            raw_key = f"raw_{axis_name}"
            if not all(raw_key in item for item in all_items):
                continue
            values = np.array([item[raw_key] for item in all_items], dtype=float)
            min_value, max_value = float(values.min()), float(values.max())
            axis_stats[axis_name] = {
                "min": min_value,
                "span": max(max_value - min_value, _EPSILON),
            }

        def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
            normalized = {**item}
            for axis_name, stats in axis_stats.items():
                raw_key = f"raw_{axis_name}"
                normalized[axis_name] = VectorProjectionService._json_float(
                    (item[raw_key] - stats["min"]) / stats["span"]
                )
            return {
                **normalized,
            }

        return [normalize_item(point) for point in points], [normalize_item(overlay) for overlay in overlays]

    @staticmethod
    def _json_float(value: Any) -> float:
        """! @brief 将 numpy 数值转换为 JSON 安全浮点数。"""
        numeric = float(value)
        if not np.isfinite(numeric):
            return 0.0
        return numeric
