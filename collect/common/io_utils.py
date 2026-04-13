"""
IO 工具：异步任务调度、chunk 存储、日志重定向
"""
import sys
import cv2
import numpy as np
import concurrent.futures
from pathlib import Path
from typing import Dict, List


class TeeOutput:
    """将 stdout/stderr 同时输出到终端和日志文件"""

    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_logging(log_file: Path):
    """设置双向日志输出（终端 + 文件）"""
    tee = TeeOutput(log_file)
    sys.stdout = tee
    sys.stderr = tee
    return tee


def submit_io_task(
    executor: concurrent.futures.ThreadPoolExecutor,
    io_futures: List[concurrent.futures.Future],
    max_pending: int,
    fn,
    *args,
):
    """提交异步 IO 任务，队列超过 max_pending 时回收已完成任务。"""
    io_futures.append(executor.submit(fn, *args))
    if len(io_futures) >= max_pending:
        _, not_done = concurrent.futures.wait(
            io_futures, return_when=concurrent.futures.FIRST_COMPLETED,
        )
        io_futures[:] = list(not_done)


def drain_io_futures(io_futures: List[concurrent.futures.Future]):
    """等待所有异步 IO 任务完成并清空列表"""
    if io_futures:
        print(f"Waiting for {len(io_futures)} pending IO operations...")
        concurrent.futures.wait(io_futures)
        io_futures.clear()
        print("All IO completed")


def save_chunk_npz(
    chunk_path: str,
    frame_ids: np.ndarray,
    rgb_by_dir: Dict[str, np.ndarray],
    depth_by_dir: Dict[str, np.ndarray],
    pose_by_dir: Dict[str, np.ndarray],
    jpg_quality: int = 90,
):
    """
    将一个时间块内的多视角数据打包写入单个 NPZ 文件。
    RGB 以 JPEG 编码存储，大幅减小文件体积。

    Args:
        chunk_path: 输出 .npz 路径
        frame_ids: [N] int32 帧号
        rgb_by_dir: {direction: [N, H, W, C] uint8 BGR}
        depth_by_dir: {direction: [N, H, W] float16}  (可以仅包含部分方向)
        pose_by_dir: {direction: [N, 4, 4] float32}
        jpg_quality: JPEG 压缩质量
    """
    chunk_dict = {"frame_ids": frame_ids}
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]

    for d in rgb_by_dir:
        jpg_list = []
        for i in range(len(rgb_by_dir[d])):
            _, buf = cv2.imencode(".jpg", rgb_by_dir[d][i], encode_params)
            jpg_list.append(buf.astype(np.uint8).ravel())
        chunk_dict[f"rgb_{d}"] = np.array(jpg_list, dtype=object)
        if d in depth_by_dir:
            chunk_dict[f"depth_{d}"] = depth_by_dir[d]
        chunk_dict[f"pose_{d}"] = pose_by_dir[d]

    np.savez(chunk_path, **chunk_dict)


def save_image_async(path: Path, image: np.ndarray) -> bool:
    """保存图像（设计用于 ThreadPoolExecutor.submit）"""
    try:
        return cv2.imwrite(str(path), image)
    except Exception as e:
        print(f"  Image save failed: {path.name}, error: {e}")
        return False
