import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def check_tools():
    for tool in ["ffmpeg", "ffprobe"]:
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"没有找到 {tool}。请先安装 FFmpeg，并确认 ffmpeg/ffprobe 已加入系统 PATH。"
            )


def run_cmd(cmd):
    print("\n运行命令：")
    print(subprocess.list2cmdline(cmd))
    subprocess.run(cmd, check=True)


def get_duration_seconds(input_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ]
    result = subprocess.check_output(cmd, text=True).strip()
    return float(result)


def compress_crf(input_path: Path, output_path: Path, width: int, crf: int, preset: str, audio_kbps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", f"scale={width}:-2",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", f"{audio_kbps}k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    run_cmd(cmd)


def compress_target_size(input_path: Path, output_path: Path, width: int, target_mb: float, preset: str, audio_kbps: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = get_duration_seconds(input_path)

    # 目标大小转成总码率，单位 kbps
    target_kbits = target_mb * 1024 * 1024 * 8 / 1000
    total_kbps = target_kbits / duration

    video_kbps = int(total_kbps - audio_kbps)

    if video_kbps < 200:
        raise RuntimeError(
            f"目标体积太小，计算得到的视频码率只有 {video_kbps} kbps。"
            f"请增大 --target-mb，或者降低 --width。"
        )

    print(f"\n视频时长：{duration:.2f} 秒")
    print(f"目标大小：{target_mb:.1f} MB")
    print(f"音频码率：{audio_kbps} kbps")
    print(f"视频码率：{video_kbps} kbps")

    passlog = str(Path(tempfile.gettempdir()) / "ffmpeg_2pass_video_compress")

    common = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-vf", f"scale={width}:-2",
        "-c:v", "libx264",
        "-b:v", f"{video_kbps}k",
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-passlogfile", passlog,
    ]

    pass1 = common + [
        "-pass", "1",
        "-an",
        "-f", "null",
        os.devnull,
    ]

    pass2 = common + [
        "-pass", "2",
        "-c:a", "aac",
        "-b:a", f"{audio_kbps}k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    run_cmd(pass1)
    run_cmd(pass2)

    # 清理二次编码日志文件
    for suffix in ["-0.log", "-0.log.mbtree", ".log", ".log.mbtree"]:
        p = Path(passlog + suffix)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Compress MP4 video for GitHub Pages.")
    parser.add_argument("-i", "--input", required=True, help="输入视频路径")
    parser.add_argument("-o", "--output", required=True, help="输出视频路径")
    parser.add_argument("--width", type=int, default=1280, help="输出视频宽度，默认 1280")
    parser.add_argument("--crf", type=int, default=28, help="CRF 压缩质量，数值越大体积越小，推荐 26-32")
    parser.add_argument("--preset", default="medium", help="编码速度，推荐 medium 或 slow")
    parser.add_argument("--audio-kbps", type=int, default=96, help="音频码率，默认 96kbps")
    parser.add_argument("--target-mb", type=float, default=None, help="目标文件大小 MB，例如 80")

    args = parser.parse_args()

    check_tools()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    if args.target_mb is not None:
        compress_target_size(
            input_path=input_path,
            output_path=output_path,
            width=args.width,
            target_mb=args.target_mb,
            preset=args.preset,
            audio_kbps=args.audio_kbps,
        )
    else:
        compress_crf(
            input_path=input_path,
            output_path=output_path,
            width=args.width,
            crf=args.crf,
            preset=args.preset,
            audio_kbps=args.audio_kbps,
        )

    input_size = input_path.stat().st_size / 1024 / 1024
    output_size = output_path.stat().st_size / 1024 / 1024

    print("\n压缩完成！")
    print(f"原文件大小：{input_size:.2f} MB")
    print(f"新文件大小：{output_size:.2f} MB")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()