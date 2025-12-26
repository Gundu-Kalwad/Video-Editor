#!/usr/bin/env python3
"""
FFmpeg-based video editor (Agent/CI friendly).
Features: Base64 decoding, URL downloading, Auto-Cleanup, Text Wrapping.
"""
import argparse
import base64
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Global list to track temp files for cleanup
TEMP_FILES = []

def register_temp_file(path: str) -> None:
    TEMP_FILES.append(path)

def cleanup_temp_files() -> None:
    if not TEMP_FILES:
        return
    print(f"[Info] Cleaning up {len(TEMP_FILES)} temporary files...")
    for path in TEMP_FILES:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[Warning] Failed to delete temp file {path}: {e}")

# --- ASSET LOADING HELPERS ---

def decode_data_uri(uri: str) -> str:
    if not uri.startswith("data:"): return uri
    try:
        header, encoded = uri.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
        ext = mime.split("/")[1]
        if ext == "jpeg": ext = "jpg"
        data = base64.b64decode(encoded)
        t = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
        t.write(data)
        t.close()
        register_temp_file(t.name)
        return t.name
    except Exception as e:
        print(f"[Error] Failed to decode Base64: {e}")
        return uri

def download_asset(url: str) -> str:
    try:
        print(f"[Info] Downloading asset: {url}")
        ext = Path(url).suffix or ".tmp"
        if "?" in ext: ext = ext.split("?")[0]
        t = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(t.name, 'wb') as out_file:
            out_file.write(response.read())
        register_temp_file(t.name)
        return t.name
    except Exception as e:
        print(f"[Warning] Failed to download {url}: {e}")
        return url

def preprocess_spec(spec: Dict) -> Dict:
    assets = spec.get("assets", {})
    new_assets = {}
    for key, val in assets.items():
        def process_item(v):
            if not isinstance(v, str): return v
            if v.startswith("data:"): return decode_data_uri(v)
            if v.startswith("http://") or v.startswith("https://"): return download_asset(v)
            return v
        if isinstance(val, list): new_assets[key] = [process_item(v) for v in val]
        elif isinstance(val, str): new_assets[key] = process_item(val)
        else: new_assets[key] = val
    spec["assets"] = new_assets
    return spec

# --- EDITOR LOGIC ---

def run(cmd: List[str], dry_run: bool = False) -> None:
    printable = shlex.join(cmd)
    print(f"$ {printable}")
    if dry_run: return
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.stdout.buffer.write(result.stdout)
        raise SystemExit(result.returncode)

def escape_drawtext(text: str) -> str:
    # CRITICAL FIX: Escape for FFmpeg filter syntax
    text = text.replace("\\", "\\\\") 
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    text = text.replace("'", "\\'") 
    return text

def escape_path_for_filter(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:")

def atempo_chain(factor: float) -> List[str]:
    chain = []
    remaining = factor
    while remaining > 2.0:
        chain.append(2.0); remaining /= 2.0
    while remaining < 0.5:
        chain.append(0.5); remaining /= 0.5
    if not math.isclose(remaining, 1.0): chain.append(remaining)
    return chain or [1.0]

def position_to_xy(position: Optional[str], x: Optional[int], y: Optional[int]) -> Tuple[str, str]:
    if position:
        pos = position.lower()
        if pos == "center": return "(main_w-text_w)/2", "(main_h-text_h)/2"
        if pos == "top-left": return "20", "20"
        if pos == "top-right": return "main_w-text_w-20", "20"
        if pos == "bottom-left": return "20", "main_h-text_h-20"
        if pos == "bottom-right": return "main_w-text_w-20", "main_h-text_h-20"
    return str(x if x is not None else 20), str(y if y is not None else 20)

def overlay_position_to_xy(position: Optional[str], x: Optional[int], y: Optional[int]) -> Tuple[str, str]:
    if position:
        pos = position.lower()
        if pos == "center": return "(main_w-overlay_w)/2", "(main_h-overlay_h)/2"
        if pos == "top-left": return "20", "20"
        if pos == "top-right": return "main_w-overlay_w-20", "20"
        if pos == "bottom-left": return "20", "main_h-overlay_h-20"
        if pos == "bottom-right": return "main_w-overlay_w-20", "main_h-overlay_h-20"
    return str(x if x is not None else 20), str(y if y is not None else 20)

def has_audio_stream(path: str, ffprobe: str = "ffprobe") -> bool:
    cmd = [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        return res.stdout.strip() != b""
    except FileNotFoundError: return True

class FFmpegEditor:
    def __init__(self, spec: Dict, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"):
        self.spec = spec
        self.assets = spec.get("assets", {})
        self.default_fontfile = spec.get("default_fontfile")
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.inputs = []
        self.filter_lines = []
        self.vlabel = None; self.alabel = None
        self.next_v = 1; self.next_a = 1
        self.drop_audio = False
        self.thumb_actions = []

        main_input = self._get_main_input()
        self.inputs.append(main_input)
        self._init_streams(main_input)

    def _get_main_input(self) -> str:
        if "input_video" in self.spec and self.spec["input_video"]: return str(self.spec["input_video"])
        if "input_video" in self.assets and self.assets["input_video"]: return str(self.assets["input_video"])
        vids = self.assets.get("input_videos")
        if isinstance(vids, list) and vids: return str(vids[0])
        raise ValueError("input_video is required")

    def _resolve_path(self, op: Dict, key: str = "path") -> Optional[str]:
        if key in op and op[key]: return str(op[key])
        desired = op.get("asset") or op.get("filename")
        if not desired: return None
        if desired in self.assets:
            val = self.assets[desired]
            return str(val[0]) if isinstance(val, list) else str(val)
        return desired

    def _init_streams(self, main_input: str) -> None:
        self.vlabel = "v0"; self.filter_lines.append(f"[0:v]null[{self.vlabel}]")
        if has_audio_stream(main_input, self.ffprobe):
            self.alabel = "a0"; self.filter_lines.append(f"[0:a]anull[{self.alabel}]")
        else: self.alabel = None

    def _new_v(self) -> str:
        lbl = f"v{self.next_v}"; self.next_v += 1; return lbl
    def _new_a(self) -> str:
        lbl = f"a{self.next_a}"; self.next_a += 1; return lbl

    def add_video_filter(self, expr: str, input_label: Optional[str] = None) -> None:
        if input_label is None: input_label = self.vlabel
        out = self._new_v()
        self.filter_lines.append(f"[{input_label}]{expr}[{out}]")
        self.vlabel = out

    def add_audio_filter(self, expr: str, input_label: Optional[str] = None) -> None:
        if input_label is None: input_label = self.alabel
        if input_label is None: return
        out = self._new_a()
        self.filter_lines.append(f"[{input_label}]{expr}[{out}]")
        self.alabel = out

    def add_overlay_input(self, path: str, scale_expr: Optional[str], opacity: Optional[float]) -> str:
        idx = len(self.inputs)
        self.inputs.append(path)
        base_label = f"ov{idx}"
        overlay_filters = []
        overlay_filters.append(f"[{idx}:v]format=rgba[{base_label}_fmt]")
        in_label = f"{base_label}_fmt"
        if scale_expr:
            overlay_filters.append(f"[{in_label}]scale={scale_expr}[{base_label}_s]")
            in_label = f"{base_label}_s"
        if opacity is not None:
            overlay_filters.append(f"[{in_label}]colorchannelmixer=aa={opacity}[{base_label}_o]")
            in_label = f"{base_label}_o"
        final_label = f"{base_label}_out"
        overlay_filters.append(f"[{in_label}]copy[{final_label}]")
        self.filter_lines.extend(overlay_filters)
        return final_label

    def handle_operations(self) -> None:
        for op in self.spec.get("operations", []):
            action = op.get("action", "").lower()
            if action == "trim":
                st = op.get("start_time", 0)
                et = op.get("end_time")
                if et is not None:
                    self.add_video_filter(f"trim=start={st}:end={et},setpts=PTS-STARTPTS")
                    if self.alabel: self.add_audio_filter(f"atrim=start={st}:end={et},asetpts=PTS-STARTPTS")
                else:
                    self.add_video_filter(f"trim=start={st},setpts=PTS-STARTPTS")
                    if self.alabel: self.add_audio_filter(f"atrim=start={st},asetpts=PTS-STARTPTS")
            elif action == "set_resolution":
                w, h = op.get("width"), op.get("height")
                mode = (op.get("mode") or "fit").lower()
                bg = op.get("background_color", "black")
                if w and h:
                    if mode == "fill": 
                        self.add_video_filter(f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}")
                    elif mode == "blur":
                        split1, split2 = self._new_v(), self._new_v()
                        self.filter_lines.append(f"[{self.vlabel}]split=2[{split1}][{split2}]")
                        bg_out = self._new_v()
                        self.filter_lines.append(f"[{split1}]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=luma_radius=min(h\,w)/20:luma_power=1:chroma_radius=min(cw\,ch)/20:chroma_power=1[{bg_out}]")
                        fg_out = self._new_v()
                        self.filter_lines.append(f"[{split2}]scale={w}:{h}:force_original_aspect_ratio=decrease[{fg_out}]")
                        final_out = self._new_v()
                        self.filter_lines.append(f"[{bg_out}][{fg_out}]overlay=(W-w)/2:(H-h)/2[{final_out}]")
                        self.vlabel = final_out
                    else: 
                        self.add_video_filter(f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color={bg}")
            elif action == "speed":
                factor = float(op.get("factor", 1.0))
                if not math.isclose(factor, 1.0):
                    self.add_video_filter(f"setpts=PTS/{factor}")
                    if self.alabel:
                        if op.get("preserve_audio_pitch", True):
                            for f in atempo_chain(1 / factor): self.add_audio_filter(f"atempo={f}")
                        else: self.add_audio_filter(f"asetrate=sample_rate*{factor},aresample=sample_rate")
            elif action == "rotate":
                deg = int(op.get("degrees", 0)) % 360
                if deg == 90: self.add_video_filter("transpose=1")
                elif deg == 180: self.add_video_filter("transpose=1,transpose=1")
                elif deg == 270: self.add_video_filter("transpose=2")
            elif action == "flip_horizontal": self.add_video_filter("hflip")
            elif action == "flip_vertical": self.add_video_filter("vflip")
            elif action == "crop":
                x, y, w, h = op.get("x", 0), op.get("y", 0), op.get("width"), op.get("height")
                if w and h: self.add_video_filter(f"crop={w}:{h}:{x}:{y}")
            elif action == "fade_video":
                self.add_video_filter(f"fade=t={op.get('direction', 'in')}:st={op.get('start_time', 0)}:d={op.get('duration', 1)}:color={op.get('color', 'black')}")
            elif action == "text":
                raw_content = op.get("content", "")
                wrap_width = op.get("wrap_width", 40)
                wrapped_lines = textwrap.fill(raw_content, width=wrap_width)
                content = escape_drawtext(wrapped_lines)
                fx, fy = position_to_xy(op.get("position"), op.get("x"), op.get("y"))
                parts = [f"text='{content}'", f"x={fx}", f"y={fy}", f"fontsize={op.get('size', 36)}", f"fontcolor={op.get('color', 'white')}", f"line_spacing={op.get('line_spacing', 5)}"]
                font_path = op.get("fontfile") or self.default_fontfile
                if font_path: parts.append(f"fontfile='{escape_path_for_filter(font_path)}'")
                elif op.get("font"): parts.append(f"font='{op['font']}'")
                if op.get("outline"): parts.append("borderw=2:bordercolor=black")
                if op.get("shadow"): parts.append("shadowcolor=black:shadowx=2:shadowy=2")
                if op.get("start_time") is not None and op.get("duration") is not None:
                    parts.append(f"enable='between(t,{op['start_time']},{op['start_time']+op['duration']})'")
                self.add_video_filter("drawtext=" + ":".join(parts))
            elif action == "image_overlay":
                path = self._resolve_path(op)
                if not path: continue
                scale_expr = None
                raw_scale = op.get("scale")
                if raw_scale: scale_expr = f"trunc(iw*{raw_scale}/2)*2:trunc(ih*{raw_scale}/2)*2"
                ov_label = self.add_overlay_input(path, scale_expr, op.get("opacity"))
                ox, oy = overlay_position_to_xy(op.get("position"), op.get("x"), op.get("y"))
                overlay_expr = f"overlay=x={ox}:y={oy}"
                if op.get("start_time") is not None and op.get("duration") is not None:
                    overlay_expr += f":enable='between(t,{op['start_time']},{op['start_time']+op['duration']})'"
                out = self._new_v()
                self.filter_lines.append(f"[{self.vlabel}][{ov_label}]{overlay_expr}[{out}]")
                self.vlabel = out
            elif action == "adjust":
                args = []
                for k in ["brightness", "contrast", "saturation", "gamma"]:
                    if op.get(k): args.append(f"{k}={op[k]}")
                if args: self.add_video_filter("eq=" + ":".join(args))
            elif action == "blur":
                if op.get("sigma"): self.add_video_filter(f"gblur=sigma={op['sigma']}")
            elif action == "sharpen":
                self.add_video_filter(f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={op.get('amount', 1.0)}")
            elif action == "replace_audio":
                path = self._resolve_path(op)
                if path:
                    idx = len(self.inputs); self.inputs.append(path)
                    self.alabel = self._new_a()
                    self.filter_lines.append(f"[{idx}:a]anull[{self.alabel}]")
            elif action == "mix_audio":
                path = self._resolve_path(op)
                if path:
                    music_vol = op.get("music_volume", 1.0)
                    idx = len(self.inputs); self.inputs.append(path)
                    music_lbl = self._new_a()
                    self.filter_lines.append(f"[{idx}:a]anull[{music_lbl}_in]")
                    self.filter_lines.append(f"[{music_lbl}_in]volume={music_vol}[{music_lbl}]")
                    if self.alabel:
                        out = self._new_a()
                        self.filter_lines.append(f"[{self.alabel}][{music_lbl}]amix=inputs=2:duration=first:normalize=0[{out}]")
                        self.alabel = out
                    else: self.alabel = music_lbl
            elif action == "volume":
                if self.alabel:
                    if "gain" in op: self.add_audio_filter(f"volume={op['gain']}dB")
                    elif "multiplier" in op: self.add_audio_filter(f"volume={op['multiplier']}")
            elif action == "mute":
                self.drop_audio = True; self.alabel = None
            elif action == "fade_audio":
                if self.alabel: self.add_audio_filter(f"afade=t={op.get('direction', 'in')}:st={op.get('start_time', 0)}:d={op.get('duration', 1)}")
            elif action == "burn_subtitles":
                path = self._resolve_path(op)
                if path: self.add_video_filter(f"subtitles='{escape_path_for_filter(path)}'")
            elif action == "thumbnail": self.thumb_actions.append(op)

    def build(self) -> Tuple[List[str], List[List[str]]]:
        self.handle_operations()
        output_cfg = self.spec.get("output", {})
        out_path = output_cfg.get("path")
        if not out_path: raise ValueError("output.path is required")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [self.ffmpeg, "-y"]
        for inp in self.inputs: cmd += ["-i", inp]
        if self.filter_lines: cmd += ["-filter_complex", ";".join(self.filter_lines)]
        cmd += ["-map", f"[{self.vlabel}]"]
        if not self.drop_audio and self.alabel: cmd += ["-map", f"[{self.alabel}]"]
        else: cmd += ["-an"]
        
        if output_cfg.get("video_bitrate"): cmd += ["-b:v", str(output_cfg["video_bitrate"])]
        if output_cfg.get("audio_bitrate"): cmd += ["-b:a", str(output_cfg["audio_bitrate"])]
        if output_cfg.get("crf") is not None: cmd += ["-crf", str(output_cfg["crf"])]
        if output_cfg.get("preset"): cmd += ["-preset", str(output_cfg["preset"])]
        if output_cfg.get("format"): cmd += ["-f", str(output_cfg["format"])]
        cmd += [out_path]

        thumb_cmds = []
        for t in self.thumb_actions:
            thumb_out = t.get("output", "thumb.jpg")
            Path(thumb_out).parent.mkdir(parents=True, exist_ok=True)
            thumb_cmds.append([self.ffmpeg, "-y", "-ss", str(t.get("timestamp", 0)), "-i", out_path, "-vframes", "1", thumb_out])
        return cmd, thumb_cmds

def load_spec_from_path(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    return preprocess_spec(spec)

def load_spec_from_env() -> Dict:
    # Safely load JSON from Environment Variable
    env_str = os.environ.get("JSON_PAYLOAD", "")
    if not env_str:
        raise ValueError("JSON_PAYLOAD environment variable is empty.")
    try:
        spec = json.loads(env_str)
        return preprocess_spec(spec)
    except json.JSONDecodeError as e:
        print(f"[Fatal] Invalid JSON in Environment Variable: {e}")
        print(f"Content dump: {env_str[:100]}...")
        sys.exit(1)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", help="Path to JSON spec file")
    p.add_argument("--json", help="Direct JSON string")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    
    spec = None
    try:
        # Priority 1: File
        if args.spec:
            spec = load_spec_from_path(args.spec)
        # Priority 2: Argument
        elif args.json:
            spec = preprocess_spec(json.loads(args.json))
        # Priority 3: Environment Variable (Recommended for CI)
        elif "JSON_PAYLOAD" in os.environ:
            spec = load_spec_from_env()
        else:
            print("Error: No input provided. Use --spec, --json, or set JSON_PAYLOAD env var.")
            sys.exit(1)

        editor = FFmpegEditor(spec)
        cmd, thumb_cmds = editor.build()
        run(cmd, dry_run=args.dry_run)
        if not args.dry_run:
            for tcmd in thumb_cmds: run(tcmd)
            
    except Exception as e:
        print(f"Error: {e}")
        raise
    finally:
        cleanup_temp_files()

if __name__ == "__main__":
    main()
