
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


def _select_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


PARENT_OF: Dict[str, str] = {
    'bb': 'back_bumper',
    'tr': 'trunk',
    'fl_l': 'back_left_fog_lamp',
    'fl_r': 'back_right_fog_lamp',
    'lb': 'lower_back_bumper',
    'bw': 'back_window',
}


class PartsYoloSegmenter:

    DEFAULT_IMG_SIZE = 640
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        conf: float = 0.30,
        img_size: int = DEFAULT_IMG_SIZE,
        encoder_name: str = 'resnet50',
    ):
        import segmentation_models_pytorch as smp

        self.conf = float(conf)
        self.img_size = int(img_size)
        self.device = _select_device()

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if not isinstance(ckpt, dict) or 'model_state' not in ckpt:
            raise RuntimeError(
                f'DeepLabV3+ checkpoint {model_path!r}: ожидался dict с '
                f"ключом 'model_state' (как из обучающего ноутбука)."
            )
        self.class_names: List[str] = list(ckpt.get('class_names') or [])
        if not self.class_names:
            raise RuntimeError(
                f"DeepLabV3+ checkpoint {model_path!r}: отсутствует 'class_names'."
            )
        self.num_classes = len(self.class_names)

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=self.num_classes,
            activation=None,
        )
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device).eval()
        self.names = {i: n for i, n in enumerate(self.class_names)}

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    def predict(self, img_bgr: np.ndarray) -> List[Dict]:
        if img_bgr is None or img_bgr.size == 0:
            return []
        h, w = img_bgr.shape[:2]
        tensor = self._preprocess(img_bgr)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = logits.softmax(dim=1).squeeze(0).cpu().numpy()

        pred_small = probs.argmax(axis=0).astype(np.uint8)
        pred_orig = cv2.resize(pred_small, (w, h), interpolation=cv2.INTER_NEAREST)

        items: List[Dict] = []
        for cls_idx in range(1, self.num_classes):
            mask = (pred_orig == cls_idx).astype(np.uint8)
            if mask.sum() == 0:
                continue
            mean_conf = float(probs[cls_idx][pred_small == cls_idx].mean())
            if mean_conf < self.conf:
                continue
            items.append({
                'label': self.class_names[cls_idx],
                'mask': mask,
                'conf': mean_conf,
            })
        return items


class DamagesYoloSegmenter:

    DEFAULT_CLASS_NAMES: Tuple[str, ...] = ('Broken',)

    def __init__(
        self,
        model_path: str,
        conf: float = 0.2,
        nms_thresh: float = 0.5,
        min_size_test: int = 1024,
        max_size_test: int = 1333,
        class_names: Optional[Tuple[str, ...]] = None,
    ):
        try:
            from detectron2 import model_zoo as _dt2_model_zoo
            from detectron2.config import get_cfg as _dt2_get_cfg
            from detectron2.engine import DefaultPredictor as _DefaultPredictor
        except Exception as exc:
            raise RuntimeError(
                'detectron2 не установлен. Поставьте Visual Studio Build Tools и '
                "затем `pip install --no-build-isolation "
                "'git+https://github.com/facebookresearch/detectron2.git'` "
                f'(оригинальная ошибка: {exc}).'
            )

        self.conf = float(conf)
        self.class_names = tuple(class_names) if class_names else self.DEFAULT_CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.device = _select_device()

        cfg = _dt2_get_cfg()
        cfg.merge_from_file(
            _dt2_model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        )
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = float(nms_thresh)
        cfg.INPUT.MIN_SIZE_TEST = int(min_size_test)
        cfg.INPUT.MAX_SIZE_TEST = int(max_size_test)
        cfg.INPUT.FORMAT = 'BGR'
        cfg.INPUT.MASK_FORMAT = 'bitmask'
        self._cfg = cfg
        self.predictor = _DefaultPredictor(cfg)
        self.names = {i: n for i, n in enumerate(self.class_names)}

    def predict(self, img_bgr: np.ndarray) -> List[Dict]:
        if img_bgr is None or img_bgr.size == 0:
            return []
        outputs = self.predictor(img_bgr)
        instances = outputs.get('instances')
        if instances is None or len(instances) == 0:
            return []
        instances = instances.to('cpu')

        masks_t = instances.pred_masks
        scores_t = instances.scores
        classes_t = instances.pred_classes

        masks = masks_t.numpy().astype(np.uint8)
        scores = scores_t.numpy().astype(float)
        classes = classes_t.numpy().astype(int)

        items: List[Dict] = []
        for m, cf, cls in zip(masks, scores, classes):
            label = self.class_names[int(cls)] if 0 <= int(cls) < self.num_classes else f'cls_{int(cls)}'
            items.append({
                'label': label or 'damage',
                'mask': m,
                'conf': float(cf),
            })
        return items


class _AddCoords(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        ys = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, xs, ys], dim=1)


class _HRNetPoseFast(nn.Module):

    def __init__(self, num_keypoints: int, backbone_name: str = 'hrnet_w32'):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        feat_channels = self.backbone.feature_info.channels()
        total_ch = int(sum(feat_channels))
        self.add_coords = _AddCoords()
        self.head = nn.Sequential(
            nn.Conv2d(total_ch, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head_final = nn.Sequential(
            nn.Conv2d(256 + 2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_keypoints, kernel_size=1),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(pixel_values)
        target_size = feats[0].shape[-2:]
        ups = [feats[0]]
        for f in feats[1:]:
            ups.append(F.interpolate(f, size=target_size, mode='bilinear', align_corners=False))
        x = torch.cat(ups, dim=1)
        x = self.head(x)
        x = self.add_coords(x)
        return self.head_final(x)


class KeypointsYoloDetector:

    DEFAULT_INPUT_H = 384
    DEFAULT_INPUT_W = 288
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str,
        conf: float = 0.0,
        input_h: int = DEFAULT_INPUT_H,
        input_w: int = DEFAULT_INPUT_W,
        backbone_name: str = 'hrnet_w32',
        box_half_px: float = 8.0,
    ):
        self.conf = float(conf)
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.box_half_px = float(box_half_px)
        self.device = _select_device()

        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if not isinstance(ckpt, dict) or 'model_state_dict' not in ckpt:
            raise RuntimeError(
                f'HRNet checkpoint {model_path!r}: ожидался dict с '
                f"ключом 'model_state_dict' (как из обучающего ноутбука)."
            )
        self.kp_names: List[str] = list(ckpt.get('kp_names') or [])
        if not self.kp_names:
            raise RuntimeError(
                f"HRNet checkpoint {model_path!r}: отсутствует 'kp_names'."
            )
        num_kpts = len(self.kp_names)

        self.model = _HRNetPoseFast(num_keypoints=num_kpts, backbone_name=backbone_name)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()
        self.names = {i: n for i, n in enumerate(self.kp_names)}

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - self.IMAGENET_MEAN) / self.IMAGENET_STD
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    @staticmethod
    def _subpixel_refine(hm: np.ndarray, py: int, px: int) -> Tuple[float, float]:
        hm_h, hm_w = hm.shape
        if 0 < px < hm_w - 1:
            dx = hm[py, px + 1] - hm[py, px - 1]
            px_sub = float(px) + 0.25 * float(np.sign(dx))
        else:
            px_sub = float(px)
        if 0 < py < hm_h - 1:
            dy = hm[py + 1, px] - hm[py - 1, px]
            py_sub = float(py) + 0.25 * float(np.sign(dy))
        else:
            py_sub = float(py)
        return px_sub, py_sub

    def predict(self, img_bgr: np.ndarray) -> List[Dict]:
        if img_bgr is None or img_bgr.size == 0:
            return []
        h, w = img_bgr.shape[:2]
        tensor = self._preprocess(img_bgr)
        with torch.no_grad():
            heatmaps = self.model(tensor).squeeze(0).cpu().numpy()

        K, hm_h, hm_w = heatmaps.shape
        sx = float(w) / float(hm_w)
        sy = float(h) / float(hm_h)
        items: List[Dict] = []
        for k in range(K):
            hm = heatmaps[k]
            py, px = np.unravel_index(int(np.argmax(hm)), hm.shape)
            px_sub, py_sub = self._subpixel_refine(hm, int(py), int(px))
            peak = float(hm[py, px])
            if peak >= 0:
                score = 1.0 / (1.0 + float(np.exp(-peak)))
            else:
                ex = float(np.exp(peak))
                score = ex / (1.0 + ex)
            if score < self.conf:
                continue

            cx = (px_sub + 0.5) * sx
            cy = (py_sub + 0.5) * sy
            glb_name = self.kp_names[k]
            items.append({
                'cls_name': glb_name,
                'prefix': None,
                'k': None,
                'cx': float(cx),
                'cy': float(cy),
                'conf': float(score),
                'box': [
                    float(cx - self.box_half_px),
                    float(cy - self.box_half_px),
                    float(cx + self.box_half_px),
                    float(cy + self.box_half_px),
                ],
                'glb_name': glb_name,
            })
        return items


@dataclass
class LargestCarResult:
    found: bool
    mask: Optional[np.ndarray]
    bbox_xyxy: Optional[Tuple[int, int, int, int]]
    confidence: Optional[float]
    area_pixels: Optional[int]


def clip_box_to_image(box_xyxy: Tuple[int, int, int, int], image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(int(x1), image_w - 1))
    y1 = max(0, min(int(y1), image_h - 1))
    x2 = max(0, min(int(x2), image_w))
    y2 = max(0, min(int(y2), image_h))
    return x1, y1, x2, y2


class LargestCarFinder:

    def __init__(
        self,
        model_path: str,
        car_class_id: int = 2,
        conf: float = 0.20,
        iou: float = 0.50,
    ):
        self.model = YOLO(model_path)
        self.car_class_id = car_class_id
        self.conf = conf
        self.iou = iou

    def find_largest_car(self, image_bgr: np.ndarray) -> LargestCarResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=image_rgb,
            classes=[self.car_class_id],
            conf=self.conf,
            iou=self.iou,
            retina_masks=True,
            verbose=False,
        )
        if not results:
            return LargestCarResult(False, None, None, None, None)

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return LargestCarResult(False, None, None, None, None)

        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(int)
        scores = r.boxes.conf.detach().cpu().numpy().astype(float)
        masks = r.masks.data.detach().cpu().numpy()

        img_h, img_w = image_bgr.shape[:2]
        if masks.shape[1] != img_h or masks.shape[2] != img_w:
            resized = [cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST) for m in masks]
            masks = np.stack(resized, axis=0)

        masks_bin = (masks > 0.5).astype(np.uint8)
        areas = masks_bin.reshape(masks_bin.shape[0], -1).sum(axis=1)
        largest_idx = int(np.argmax(areas))

        mask = (masks_bin[largest_idx] * 255).astype(np.uint8)
        bbox = clip_box_to_image(tuple(int(v) for v in boxes[largest_idx].tolist()), img_w, img_h)
        return LargestCarResult(
            found=True,
            mask=mask,
            bbox_xyxy=bbox,
            confidence=float(scores[largest_idx]),
            area_pixels=int(areas[largest_idx]),
        )


def draw_car_overlay(
    image_bgr: np.ndarray,
    mask_255: np.ndarray,
    bbox_xyxy: Optional[Tuple[int, int, int, int]] = None,
    label: Optional[str] = None,
    color_bgr: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.35,
) -> np.ndarray:
    result = image_bgr.copy()
    overlay = result.copy()
    overlay[mask_255 > 0] = color_bgr
    result = cv2.addWeighted(result, 1.0 - alpha, overlay, alpha, 0)

    contours, _ = cv2.findContours(mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color_bgr, 2)

    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(result, (x1, y1), (x2, y2), color_bgr, 2)
        if label:
            cv2.putText(
                result,
                label,
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color_bgr,
                2,
                cv2.LINE_AA,
            )
    return result


class AdaptivePhotoQualityGate:

    def __init__(
        self,
        min_width: int = 300,
        min_height: int = 300,
        target_short_side: int = 1,
        min_contrast_std: float = 28.0,
        blur_fix_threshold: float = 90.0,
        blur_reject_threshold: float = 60.0,
        dark_ratio_fix_threshold: float = 0.35,
        dark_ratio_reject_threshold: float = 0.75,
        bright_ratio_fix_threshold: float = 0.08,
        bright_ratio_reject_threshold: float = 0.22,
        glare_ratio_fix_threshold: float = 0.010,
        glare_ratio_reject_threshold: float = 0.040,
        noise_fix_threshold: float = 10.0,
        noise_reject_threshold: float = 20.0,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.target_short_side = target_short_side
        self.min_contrast_std = min_contrast_std
        self.blur_fix_threshold = blur_fix_threshold
        self.blur_reject_threshold = blur_reject_threshold
        self.dark_ratio_fix_threshold = dark_ratio_fix_threshold
        self.dark_ratio_reject_threshold = dark_ratio_reject_threshold
        self.bright_ratio_fix_threshold = bright_ratio_fix_threshold
        self.bright_ratio_reject_threshold = bright_ratio_reject_threshold
        self.glare_ratio_fix_threshold = glare_ratio_fix_threshold
        self.glare_ratio_reject_threshold = glare_ratio_reject_threshold
        self.noise_fix_threshold = noise_fix_threshold
        self.noise_reject_threshold = noise_reject_threshold

    @staticmethod
    def _round(x: float) -> float:
        return float(np.round(x, 6))

    @staticmethod
    def _gamma_correct(image_bgr: np.ndarray, gamma: float = 1.4) -> np.ndarray:
        gamma = max(gamma, 1e-6)
        lut = np.array(
            [np.clip(((i / 255.0) ** (1.0 / gamma)) * 255.0, 0, 255) for i in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(image_bgr, lut)

    @staticmethod
    def _apply_clahe_lab(
        image_bgr: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _denoise(image_bgr: np.ndarray, h: int = 6, h_color: int = 6) -> np.ndarray:
        return cv2.fastNlMeansDenoisingColored(image_bgr, None, h, h_color, 7, 21)

    @staticmethod
    def _mild_unsharp_mask(image_bgr: np.ndarray, sigma: float = 1.0, amount: float = 0.6) -> np.ndarray:
        blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    @staticmethod
    def _compress_highlights(image_bgr: np.ndarray, strength: float = 0.6) -> np.ndarray:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        v_norm = v / 255.0
        threshold = 0.78
        v_new = np.where(v_norm > threshold, threshold + (v_norm - threshold) * strength, v_norm)
        v_new = np.clip(v_new * 255.0, 0, 255).astype(np.uint8)
        hsv_new = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v_new])
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        median = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, median)
        edges = cv2.Canny(gray, 50, 150)
        flat_mask = edges == 0
        values = residual[flat_mask] if np.count_nonzero(flat_mask) > 500 else residual.reshape(-1)
        return float(values.std())

    @staticmethod
    def _compute_glare_ratio(image_bgr: np.ndarray) -> float:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        glare_mask = ((v > 245) & (s < 40)).astype(np.uint8) * 255
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(glare_mask, 8)

        area_total = image_bgr.shape[0] * image_bgr.shape[1]
        glare_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= max(30, int(0.0003 * area_total)):
                glare_area += area
        return glare_area / float(area_total)

    def evaluate(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Изображение пустое или не загружено.")

        h, w = image_bgr.shape[:2]
        area = max(h * w, 1)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        brightness_mean = float(gray.mean())
        contrast_std = float(gray.std())
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        dark_ratio = float(np.mean(gray < 40))
        bright_ratio = float(np.mean(gray > 245))
        glare_ratio = float(self._compute_glare_ratio(image_bgr))
        noise_std = float(self._estimate_noise(gray))
        edge_density = float(np.count_nonzero(cv2.Canny(gray, 100, 200))) / float(area)
        short_side = min(h, w)
        size_ok = (w >= self.min_width) and (h >= self.min_height)

        empty_like = contrast_std < 8.0 and edge_density < 0.003
        severe_blur = laplacian_variance < self.blur_reject_threshold
        mild_blur = self.blur_reject_threshold <= laplacian_variance < self.blur_fix_threshold
        severe_dark = dark_ratio > self.dark_ratio_reject_threshold
        mild_dark = self.dark_ratio_fix_threshold < dark_ratio <= self.dark_ratio_reject_threshold
        severe_bright = bright_ratio > self.bright_ratio_reject_threshold
        mild_bright = self.bright_ratio_fix_threshold < bright_ratio <= self.bright_ratio_reject_threshold
        severe_glare = glare_ratio > self.glare_ratio_reject_threshold
        mild_glare = self.glare_ratio_fix_threshold < glare_ratio <= self.glare_ratio_reject_threshold
        severe_noise = noise_std > self.noise_reject_threshold
        mild_noise = self.noise_fix_threshold < noise_std <= self.noise_reject_threshold
        low_contrast = contrast_std < self.min_contrast_std
        too_small = short_side < self.target_short_side

        checks = {
            "readable": {"passed": not empty_like, "message": f"contrast_std={contrast_std:.2f}, edge_density={edge_density:.4f}"},
            "blur": {"passed": not severe_blur, "message": f"laplacian_variance={laplacian_variance:.2f}"},
            "overexposure": {"passed": not severe_bright, "message": f"bright_ratio={bright_ratio:.4f}"},
            "glare": {"passed": not severe_glare, "message": f"glare_ratio={glare_ratio:.4f}"},
            "noise": {"passed": not severe_noise, "message": f"noise_std={noise_std:.2f}"},
            "size": {"passed": size_ok, "message": f"width={w}, height={h}, short_side={short_side}"},
            "contrast": {"passed": not low_contrast, "message": f"contrast_std={contrast_std:.2f}"},
            "darkness": {"passed": not severe_dark, "message": f"dark_ratio={dark_ratio:.4f}"},
        }

        reject_reasons: List[str] = []
        fix_reasons: List[str] = []

        if empty_like:
            reject_reasons.append("photo_not_readable_or_almost_empty")
        if severe_blur:
            reject_reasons.append("severe_blur")
        if severe_bright:
            reject_reasons.append("severe_overexposure")
        if severe_glare:
            reject_reasons.append("strong_glare")
        if severe_dark:
            reject_reasons.append("severe_underexposure")

        if mild_blur:
            fix_reasons.append("mild_blur")
        if mild_dark:
            fix_reasons.append("mild_underexposure")
        if mild_bright:
            fix_reasons.append("moderate_overexposure")
        if mild_glare:
            fix_reasons.append("moderate_glare")
        if mild_noise:
            fix_reasons.append("visible_noise")
        if low_contrast:
            fix_reasons.append("low_contrast")
        if too_small:
            fix_reasons.append("small_scale")

        if reject_reasons:
            status = "REJECT"
        elif fix_reasons:
            status = "FIX_AND_RETRY"
        else:
            status = "PASS"

        metrics = {
            "width": int(w),
            "height": int(h),
            "short_side": int(short_side),
            "brightness_mean": self._round(brightness_mean),
            "contrast_std": self._round(contrast_std),
            "laplacian_variance": self._round(laplacian_variance),
            "dark_ratio": self._round(dark_ratio),
            "bright_ratio": self._round(bright_ratio),
            "glare_ratio": self._round(glare_ratio),
            "noise_std": self._round(noise_std),
            "edge_density": self._round(edge_density),
        }

        return {
            "status": status,
            "metrics": metrics,
            "checks": checks,
            "reject_reasons": reject_reasons,
            "fix_reasons": fix_reasons,
        }

    def evaluate_in_region(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Изображение пустое или не загружено.")

        m = np.asarray(mask) > 0
        if not m.any():
            return {
                "status": "REJECT",
                "metrics": {},
                "checks": {},
                "reject_reasons": ["empty_region_mask"],
                "fix_reasons": [],
            }

        ys, xs = np.where(m)
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        region_w = x2 - x1
        region_h = y2 - y1
        region_area = int(m.sum())

        gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_pix = gray_full[m]

        brightness_mean = float(gray_pix.mean())
        contrast_std = float(gray_pix.std())
        dark_ratio = float(np.mean(gray_pix < 40))
        bright_ratio = float(np.mean(gray_pix > 245))

        lap = cv2.Laplacian(gray_full, cv2.CV_64F)
        laplacian_variance = float(lap[m].var())

        canny_full = cv2.Canny(gray_full, 100, 200)
        edge_density = float((canny_full[m] > 0).sum()) / float(region_area)

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        _, s_ch, v_ch = cv2.split(hsv)
        glare_local = ((v_ch > 245) & (s_ch < 40) & m).astype(np.uint8) * 255
        glare_local = cv2.morphologyEx(glare_local, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_lab, _, stats, _ = cv2.connectedComponentsWithStats(glare_local, 8)
        glare_area = 0
        for i in range(1, n_lab):
            a = int(stats[i, cv2.CC_STAT_AREA])
            if a >= max(30, int(0.0003 * region_area)):
                glare_area += a
        glare_ratio = float(glare_area) / float(region_area)

        med = cv2.medianBlur(gray_full, 3)
        residual = cv2.absdiff(gray_full, med).astype(np.float32)
        edges50_150 = cv2.Canny(gray_full, 50, 150)
        flat_mask = m & (edges50_150 == 0)
        if int(flat_mask.sum()) > 500:
            noise_std = float(residual[flat_mask].std())
        else:
            noise_std = float(residual[m].std())

        short_side = min(region_w, region_h)
        size_ok = (region_w >= self.min_width) and (region_h >= self.min_height)

        empty_like = contrast_std < 8.0 and edge_density < 0.003
        severe_blur = laplacian_variance < self.blur_reject_threshold
        mild_blur = self.blur_reject_threshold <= laplacian_variance < self.blur_fix_threshold
        severe_dark = dark_ratio > self.dark_ratio_reject_threshold
        mild_dark = self.dark_ratio_fix_threshold < dark_ratio <= self.dark_ratio_reject_threshold
        severe_bright = bright_ratio > self.bright_ratio_reject_threshold
        mild_bright = self.bright_ratio_fix_threshold < bright_ratio <= self.bright_ratio_reject_threshold
        severe_glare = glare_ratio > self.glare_ratio_reject_threshold
        mild_glare = self.glare_ratio_fix_threshold < glare_ratio <= self.glare_ratio_reject_threshold
        severe_noise = noise_std > self.noise_reject_threshold
        mild_noise = self.noise_fix_threshold < noise_std <= self.noise_reject_threshold
        low_contrast = contrast_std < self.min_contrast_std
        too_small = short_side < self.target_short_side

        checks = {
            "readable": {"passed": not empty_like, "message": f"contrast_std={contrast_std:.2f}, edge_density={edge_density:.4f}"},
            "blur": {"passed": not severe_blur, "message": f"laplacian_variance={laplacian_variance:.2f}"},
            "overexposure": {"passed": not severe_bright, "message": f"bright_ratio={bright_ratio:.4f}"},
            "glare": {"passed": not severe_glare, "message": f"glare_ratio={glare_ratio:.4f}"},
            "noise": {"passed": not severe_noise, "message": f"noise_std={noise_std:.2f}"},
            "size": {"passed": size_ok, "message": f"region_w={region_w}, region_h={region_h}, short_side={short_side}"},
            "contrast": {"passed": not low_contrast, "message": f"contrast_std={contrast_std:.2f}"},
            "darkness": {"passed": not severe_dark, "message": f"dark_ratio={dark_ratio:.4f}"},
        }

        reject_reasons: List[str] = []
        fix_reasons: List[str] = []
        if empty_like:
            reject_reasons.append("photo_not_readable_or_almost_empty")
        if severe_blur:
            reject_reasons.append("severe_blur")
        if severe_bright:
            reject_reasons.append("severe_overexposure")
        if severe_glare:
            reject_reasons.append("strong_glare")
        if severe_dark:
            reject_reasons.append("severe_underexposure")

        if mild_blur:
            fix_reasons.append("mild_blur")
        if mild_dark:
            fix_reasons.append("mild_underexposure")
        if mild_bright:
            fix_reasons.append("moderate_overexposure")
        if mild_glare:
            fix_reasons.append("moderate_glare")
        if mild_noise:
            fix_reasons.append("visible_noise")
        if low_contrast:
            fix_reasons.append("low_contrast")
        if too_small:
            fix_reasons.append("small_scale")

        if reject_reasons:
            status = "REJECT"
        elif fix_reasons:
            status = "FIX_AND_RETRY"
        else:
            status = "PASS"

        metrics = {
            "region_width": int(region_w),
            "region_height": int(region_h),
            "region_short_side": int(short_side),
            "region_pixel_count": int(region_area),
            "region_bbox": [int(x1), int(y1), int(x2), int(y2)],
            "brightness_mean": self._round(brightness_mean),
            "contrast_std": self._round(contrast_std),
            "laplacian_variance": self._round(laplacian_variance),
            "dark_ratio": self._round(dark_ratio),
            "bright_ratio": self._round(bright_ratio),
            "glare_ratio": self._round(glare_ratio),
            "noise_std": self._round(noise_std),
            "edge_density": self._round(edge_density),
        }

        return {
            "status": status,
            "metrics": metrics,
            "checks": checks,
            "reject_reasons": reject_reasons,
            "fix_reasons": fix_reasons,
        }

    def preprocess(self, image_bgr: np.ndarray, evaluation: Dict[str, Any]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        processed = image_bgr.copy()
        applied_ops: List[Dict[str, Any]] = []
        status = evaluation["status"]
        metrics = evaluation["metrics"]
        fix_reasons = set(evaluation["fix_reasons"])

        if status == "REJECT":
            return processed, applied_ops

        if "visible_noise" in fix_reasons:
            processed = self._denoise(processed, h=6, h_color=6)
            applied_ops.append({"op": "denoise_nlm", "h": 6, "h_color": 6})
        if "mild_underexposure" in fix_reasons:
            dark_ratio = metrics["dark_ratio"]
            gamma = 1.35 if dark_ratio < 0.50 else 1.60
            processed = self._gamma_correct(processed, gamma=gamma)
            applied_ops.append({"op": "gamma_brighten", "gamma": gamma})
        if "moderate_overexposure" in fix_reasons or "moderate_glare" in fix_reasons:
            processed = self._compress_highlights(processed, strength=0.55)
            applied_ops.append({"op": "highlight_compression", "strength": 0.55})
        if "low_contrast" in fix_reasons:
            processed = self._apply_clahe_lab(processed, clip_limit=2.0, tile_grid_size=(8, 8))
            applied_ops.append({"op": "clahe_lab", "clip_limit": 2.0, "tile_grid_size": [8, 8]})
        if "mild_blur" in fix_reasons:
            processed = self._mild_unsharp_mask(processed, sigma=1.0, amount=0.5)
            applied_ops.append({"op": "mild_unsharp_mask", "sigma": 1.0, "amount": 0.5})

        return processed, applied_ops
