
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


_DLOG: Optional[logging.Logger] = None


def init_pipeline_diag_log(out_dir: Path, console_echo: bool = False) -> Path:
    global _DLOG
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'pipeline_diag.log'
    _DLOG = logging.getLogger('damage_to_3d.pipeline_diag')
    _DLOG.handlers.clear()
    _DLOG.setLevel(logging.DEBUG)
    _DLOG.propagate = False
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    _DLOG.addHandler(fh)
    if console_echo:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter('DIAG | %(message)s'))
        _DLOG.addHandler(ch)
    dlog('=' * 88)
    dlog('INIT pipeline_diag.log | пришли этот файл целиком для разбора flip/точности/скорости')
    dlog(f'log_path={log_path.resolve()}')
    try:
        dlog(f'python={__import__("sys").version}')
        dlog(f'numpy={np.__version__} opencv={cv2.__version__}')
    except Exception as e:
        dlog(f'версии библиотек: не удалось прочитать ({e})')
    dlog('=' * 88)
    return log_path


def dlog(msg: str) -> None:
    if _DLOG is not None:
        _DLOG.debug(msg)
    else:
        logging.getLogger(__name__).debug(msg)


def dlog_arr(name: str, arr: np.ndarray, precision: int = 5) -> None:
    a = np.asarray(arr)
    dlog(f'{name} shape={a.shape} dtype={a.dtype} sample={np.array2string(a.ravel()[:min(24, a.size)], precision=precision)}')


@contextmanager
def dlog_timer(block_name: str):
    t0 = time.perf_counter()
    dlog(f'[TIME] >>> start {block_name}')
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        dlog(f'[TIME] <<< end   {block_name}  elapsed_ms={1000.0 * dt:.2f}')


def dlog_cv2_imread(path: Path, img) -> None:
    p = str(path)
    if img is None:
        dlog(f'[IMREAD] FAIL path={p} -> None (файл отсутствует/битый/не изображение)')
        return
    h, w = img.shape[:2]
    dlog(f'[IMREAD] OK path={p} shape={img.shape} dtype={img.dtype} min/max={img.min()}/{img.max()}')
    dlog(
        '[IMREAD] Примечание: cv2.imread не применяет EXIF Orientation; '
        'если фото с телефона «левым» в просмотрщике, в OpenCV оно может отличаться.'
    )


class _StopwatchSection:

    def __init__(self, sw: 'Stopwatch', name: str, echo: bool):
        self.sw = sw
        self.name = name
        self.echo = bool(echo)
        self._t0 = 0.0

    def __enter__(self) -> '_StopwatchSection':
        self._t0 = time.perf_counter()
        self.sw._depth += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.perf_counter() - self._t0
        self.sw._depth -= 1
        depth_after_exit = self.sw._depth
        self.sw._records.append({
            'name': self.name,
            'duration_s': float(dt),
            'depth': int(depth_after_exit),
            'failed': exc_type is not None,
        })
        indent = '  ' * depth_after_exit
        fail_tag = ' [FAILED]' if exc_type is not None else ''
        msg = f'[timing] {indent}{self.name}: {dt:.2f}s{fail_tag}'
        dlog(msg)
        if self.echo:
            print(msg, flush=True)


class Stopwatch:

    def __init__(self, name: str = 'pipeline', echo_steps: bool = True):
        self.name = name
        self.echo_steps = bool(echo_steps)
        self._t_start = time.perf_counter()
        self._records: List[Dict[str, Any]] = []
        self._depth = 0

    def section(self, name: str, *, echo: Optional[bool] = None) -> _StopwatchSection:
        if echo is None:
            echo = self.echo_steps
        return _StopwatchSection(self, name, echo=echo)

    @property
    def total_elapsed_s(self) -> float:
        return time.perf_counter() - self._t_start

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'total_s': round(self.total_elapsed_s, 4),
            'records': [
                {**r, 'duration_s': round(r['duration_s'], 4)}
                for r in self._records
            ],
        }

    def finalize(self, *, echo: bool = True) -> Dict[str, Any]:
        total = self.total_elapsed_s
        top_level = [r for r in self._records if int(r['depth']) == 0]
        top_sum = sum(r['duration_s'] for r in top_level)
        unaccounted = total - top_sum

        title = f'  ===== TIMINGS  ({self.name})  ====='
        if not self._records:
            lines = [
                '',
                title,
                f'  TOTAL elapsed: {total:.2f}s   (no recorded sections)',
            ]
        else:
            max_visible = max(len(r['name']) + 2 * int(r['depth']) for r in self._records)
            name_w = max(36, max_visible + 1)
            sep = '  ' + '-' * (name_w + 26)
            header_indent = ' ' * 0
            lines = [
                '',
                title,
                (
                    f'  TOTAL: {total:6.2f}s   |   top-level sum: {top_sum:6.2f}s   |   '
                    f'unaccounted: {unaccounted:+.2f}s'
                ),
                sep,
                f'  {header_indent}{"step":<{name_w}}  {"time":>8}  {"%total":>7}  flag',
                sep,
            ]
            for r in self._records:
                depth = int(r['depth'])
                indent = '  ' * depth
                visible_name = r['name']
                name_field = f'{indent}{visible_name}'.ljust(name_w)
                pct = (r['duration_s'] / total * 100.0) if total > 0 else 0.0
                flag = ' FAIL' if r.get('failed') else ''
                lines.append(
                    f'  {name_field}  {r["duration_s"]:7.3f}s  {pct:6.2f}%  {flag}'
                )
            lines.append(sep)

        msg = '\n'.join(lines)
        dlog(msg)
        if echo:
            print(msg, flush=True)

        return self.to_dict()
