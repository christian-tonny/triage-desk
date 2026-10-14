"""Build demo/triage-desk-demo.mp4 from a real run of demo/record.sh.

Runs the script, splits its output at each '# step' caption, renders every step
as a terminal frame (rich → SVG → PNG via macOS qlmanage), and joins the frames
with ffmpeg. Nothing is mocked: the frames are the console's actual output.

    uv run --with pillow python demo/make_video.py        (macOS; needs ffmpeg)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "demo" / "frames"
OUT = ROOT / "demo" / "triage-desk-demo.mp4"
COLS, ROWS = 150, 62
W, H = 1920, 1080
CAPTION = re.compile(r"^\x1b\[1;36m# (.*)\x1b\[0m$")


def run_demo() -> str:
    env = {**os.environ, "PAUSE": "0", "FORCE_COLOR": "1", "TERM": "xterm-256color", "COLUMNS": str(COLS), "LINES": "400"}
    res = subprocess.run([str(ROOT / "demo" / "record.sh")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(f"record.sh failed ({res.returncode})")
    (ROOT / "demo" / "record_output.txt").write_text(re.sub(r"\x1b\[[0-9;]*m", "", res.stdout))
    return res.stdout


def split_steps(out: str) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = []
    for line in out.splitlines():
        m = CAPTION.match(line.strip())
        if m:
            steps.append((m.group(1), []))
        elif steps:
            steps[-1][1].append(line)
    return steps


def frame(idx: int, caption: str, lines: list[str], part: str = "") -> Path:
    c = Console(record=True, width=COLS, height=ROWS, file=open(os.devnull, "w"), force_terminal=True, color_system="truecolor")
    c.print(Text(f"# {caption}{part}", style="bold cyan"))
    c.print(Text("Triage Desk · synthetic data · local Qwen2.5-7B · github.com/irachrist1/triage-desk", style="dim"))
    c.print()
    body = lines[: ROWS - 4]
    for ln in body:
        c.print(Text.from_ansi(ln), overflow="crop", no_wrap=True)
    for _ in range(ROWS - 4 - len(body)):
        c.print("")
    svg = FRAMES / f"f{idx:03d}.svg"
    c.save_svg(str(svg), title="triage-desk")
    subprocess.run(["qlmanage", "-t", "-s", "2400", "-o", str(FRAMES), str(svg)], capture_output=True, check=True)
    png = Path(str(svg) + ".png")
    im = Image.open(png).convert("RGB")
    bbox = Image.eval(im, lambda p: 255 - p).getbbox()
    im = im.crop(bbox) if bbox else im
    im.thumbnail((W, H))
    canvas = Image.new("RGB", (W, H), (41, 41, 41))
    canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    outp = FRAMES / f"f{idx:03d}.png"
    canvas.save(outp)
    png.unlink()
    svg.unlink()
    return outp


def main() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("qlmanage"):
        raise SystemExit("needs ffmpeg and macOS qlmanage")
    shutil.rmtree(FRAMES, ignore_errors=True)
    FRAMES.mkdir(parents=True)
    steps = split_steps(run_demo())
    items: list[tuple[Path, float]] = []
    i = 0
    for caption, lines in steps:
        lines = [ln for ln in lines if ln.strip() or True]
        chunks = [lines[k:k + ROWS - 4] for k in range(0, max(len(lines), 1), ROWS - 4)][:3]
        for n, ch in enumerate(chunks):
            part = f" ({n + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            items.append((frame(i, caption, ch, part), 7.0 if len(ch) > 20 else 5.0))
            i += 1
    concat = FRAMES / "list.txt"
    concat.write_text("".join(f"file '{p.name}'\nduration {d}\n" for p, d in items) + f"file '{items[-1][0].name}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-vf", "fps=10,format=yuv420p", "-c:v", "libx264", "-crf", "20", str(OUT)], cwd=FRAMES, check=True)
    print(f"wrote {OUT} ({len(items)} frames, {sum(d for _, d in items):.0f} s)")


if __name__ == "__main__":
    main()
