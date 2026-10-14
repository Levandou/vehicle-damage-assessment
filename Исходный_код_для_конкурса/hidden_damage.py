
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


_DEFAULT_THRESHOLD = 0.6
_REQUIRED_BUNDLE_KEYS = ('model', 'part_columns', 'feature_columns', 'threshold')


def _aggregate_severity_per_part(
    severity_items: Iterable[Dict[str, Any]],
    part_columns: List[str],
) -> Dict[str, float]:
    severities: Dict[str, float] = {col: 0.0 for col in part_columns}

    for item in severity_items or []:
        if not isinstance(item, dict):
            continue
        part = item.get('part')
        sev_percent = item.get('severity_percent')
        if part is None or sev_percent is None:
            continue
        try:
            sev01 = max(0.0, min(1.0, float(sev_percent) / 100.0))
        except Exception:
            continue
        key = str(part).strip().upper()
        if key not in severities:
            continue
        if sev01 > severities[key]:
            severities[key] = sev01

    return severities


def _compute_derived_features(severities: Dict[str, float]) -> Dict[str, float]:
    parts_arr = np.array(list(severities.values()), dtype=np.float64)
    damaged_count = int((parts_arr > 0).sum())

    max_sev = float(parts_arr.max()) if parts_arr.size else 0.0
    sum_sev = float(parts_arr.sum())
    mean_sev = (sum_sev / damaged_count) if damaged_count > 0 else 0.0

    return {
        'damaged_parts_count': float(damaged_count),
        'max_severity': float(max_sev),
        'sum_severity': float(sum_sev),
        'mean_severity': float(mean_sev),
        'has_bumper_damage': float(severities.get('BACK_BUMPER', 0.0) > 0),
        'has_lower_bumper_damage': float(severities.get('LOWER_BACK_BUMPER', 0.0) > 0),
        'has_trunk_damage': float(severities.get('TRUNK', 0.0) > 0),
        'has_back_window_damage': float(severities.get('BACK_WINDOW', 0.0) > 0),
        'has_lights_damage': float(
            severities.get('BACK_LEFT_HEADLIGHT', 0.0) > 0
            or severities.get('BACK_RIGHT_HEADLIGHT', 0.0) > 0
            or severities.get('BACK_LEFT_FOG_LAMP', 0.0) > 0
            or severities.get('BACK_RIGHT_FOG_LAMP', 0.0) > 0
        ),
        'has_strong_bumper_damage': float(severities.get('BACK_BUMPER', 0.0) >= 0.5),
        'has_strong_trunk_damage': float(severities.get('TRUNK', 0.0) >= 0.4),
    }


class HiddenDamageClassifier:

    def __init__(
        self,
        model_path: str | Path,
        *,
        threshold_override: Optional[float] = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f'Hidden-damage model not found: {self.model_path}'
            )

        import joblib

        bundle = joblib.load(self.model_path)
        if not isinstance(bundle, dict):
            raise ValueError(
                f'Bad XGBoost bundle: expected dict, got {type(bundle).__name__}'
            )
        missing = [k for k in _REQUIRED_BUNDLE_KEYS if k not in bundle]
        if missing:
            raise ValueError(
                f'XGBoost bundle missing keys: {missing}. Expected {_REQUIRED_BUNDLE_KEYS}'
            )

        self._model = bundle['model']
        self._part_columns: List[str] = list(bundle['part_columns'])
        self._feature_columns: List[str] = list(bundle['feature_columns'])
        self._bundle_threshold: float = float(
            bundle.get('threshold', _DEFAULT_THRESHOLD)
        )
        self.threshold: float = (
            float(threshold_override)
            if threshold_override is not None
            else self._bundle_threshold
        )
        self._target_column: str = str(bundle.get('target_column', 'hidden_damage'))

    @property
    def part_columns(self) -> List[str]:
        return list(self._part_columns)

    @property
    def feature_columns(self) -> List[str]:
        return list(self._feature_columns)

    def predict_from_severity_list(
        self,
        severity_per_part: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        severities = _aggregate_severity_per_part(severity_per_part, self._part_columns)
        derived = _compute_derived_features(severities)

        feature_values: Dict[str, float] = {**severities, **derived}
        ordered_values: List[float] = [
            float(feature_values.get(col, 0.0)) for col in self._feature_columns
        ]

        import pandas as pd
        X = pd.DataFrame([ordered_values], columns=self._feature_columns)

        try:
            proba = float(self._model.predict_proba(X)[:, 1][0])
        except Exception as exc:
            return {
                'hidden_damage_confidence': None,
                'hidden_damage_pred': None,
                'threshold': self.threshold,
                'status': 'ERROR',
                'error': f'predict_proba failed: {exc}',
                'features': feature_values,
                'severities_per_part_used': severities,
            }

        pred = int(proba >= self.threshold)
        status = 'EXPERT_REQUIRED' if pred == 1 else 'OK'

        return {
            'hidden_damage_confidence': round(proba, 6),
            'hidden_damage_pred': pred,
            'threshold': float(self.threshold),
            'status': status,
            'features': feature_values,
            'severities_per_part_used': severities,
            'model_path': str(self.model_path),
        }
