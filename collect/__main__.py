#!/usr/bin/env python3
"""
VLN-CE 数据采集统一入口

Usage:
    python -m collect r2r [args...]           # R2R 基础数据采集（单视角 + 动作）
    python -m collect panoramic [args...]     # R2R 全景数据采集（4方向 + lookdown）
    python -m collect heatmap [args...]       # 热力图巡逻数据采集（多点往返）

示例:
    python -m collect r2r --split train --num-clips 1000 --gpu 0
    python -m collect panoramic --split train --num-clips 5000 --lookdown-pitch 30
    python -m collect heatmap --num-clips 500 --num-waypoints 4
"""
import sys
import importlib

MODES = {
    "r2r":       ("collect.r2r.collector",       "R2R 基础数据采集（单视角 + 动作）"),
    "panoramic": ("collect.panoramic.collector",  "R2R 全景数据采集（4方向 + lookdown）"),
    "heatmap":   ("collect.heatmap.collector",    "热力图巡逻数据采集（多点往返）"),
}


def print_usage():
    print("Usage: python -m collect {mode} [args...]\n")
    print("Available modes:")
    for mode, (_, desc) in MODES.items():
        print(f"  {mode:<12s} {desc}")
    print("\nUse 'python -m collect {mode} --help' for mode-specific options.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0)

    mode = sys.argv[1]
    if mode not in MODES:
        print(f"Error: unknown mode '{mode}'\n")
        print_usage()
        sys.exit(1)

    module_path = MODES[mode][0]
    sys.argv = [f"collect {mode}"] + sys.argv[2:]

    module = importlib.import_module(module_path)
    module.main()


if __name__ == "__main__":
    main()
