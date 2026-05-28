#!/usr/bin/env python3
"""Video Organizer — deduplicate, rename, and clean video files on local/external drives.

Inspired by PikPak Enhancement Master's file analysis engine.
Adapted for local filesystem use with no cloud API dependency.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

VIDEO_EXTS = frozenset([
    "mp4", "mkv", "avi", "mov", "wmv", "flv", "webm", "ts", "m4v", "3gp",
    "mpg", "mpeg", "rm", "rmvb", "vob", "ogv", "m2ts", "mts", "divx", "asf",
])

AUDIO_EXTS = frozenset([
    "mp3", "wav", "flac", "aac", "m4a", "ogg", "opus", "ape", "wma", "amr",
    "m4b", "alac", "aiff", "aif", "mid", "midi", "ra", "dts", "ac3", "dsf", "dff",
])

IMAGE_EXTS = frozenset([
    "jpg", "jpeg", "png", "gif", "webp", "bmp", "avif", "tiff", "tif", "svg", "ico",
])

ARCHIVE_EXTS = frozenset([
    "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "lz4", "zst",
])

PROTECT_FILE = ".video_organizer_protect.json"


@dataclass
class FileInfo:
    path: str
    name: str
    ext: str
    size: int
    hash_md5: str = ""
    duration: float = 0.0
    mime_type: str = ""
    is_video: bool = False
    is_audio: bool = False
    is_image: bool = False

    @property
    def fingerprint(self) -> str:
        if self.hash_md5:
            return f"{self.hash_md5}_{self.size}"
        return f"{self.name}_{self.size}"


@dataclass
class FolderInfo:
    id: str
    path: str
    name: str
    parent_id: Optional[str]
    depth: int
    size: int = 0
    files: List[str] = field(default_factory=list)
    file_counts: Dict[str, int] = field(default_factory=dict)
    weighted_total: float = 0.0
    is_shell: bool = False
    is_root: bool = False
    lineage: List[str] = field(default_factory=list)
    sub_folder_ids: List[str] = field(default_factory=list)


def get_ext(path: str) -> str:
    dot = path.rfind(".")
    if dot <= 0:
        return ""
    return path[dot + 1:].lower()


def is_video_file(name: str) -> bool:
    ext = get_ext(name)
    return ext in VIDEO_EXTS


def is_audio_file(name: str) -> bool:
    ext = get_ext(name)
    return ext in AUDIO_EXTS


def is_image_file(name: str) -> bool:
    ext = get_ext(name)
    return ext in IMAGE_EXTS


def is_archive_file(name: str) -> bool:
    ext = get_ext(name)
    return ext in ARCHIVE_EXTS


def fmt_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size < 1024 * 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"
    return f"{size / (1024 * 1024 * 1024 * 1024):.2f} TB"


def compute_md5(filepath: str, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_video_duration(filepath: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return 0.0


def load_protect_list(root: str) -> Set[str]:
    p = Path(root) / PROTECT_FILE
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return set(data.get("protected", []))
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def save_protect_list(root: str, items: Set[str]) -> None:
    p = Path(root) / PROTECT_FILE
    data = {"protected": sorted(items)}
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scan_directory(root: str, compute_hash: bool = False, compute_duration: bool = False,
                   progress_callback=None) -> List[FileInfo]:
    files = []
    root_path = Path(root)
    if not root_path.exists():
        print(f"Error: path does not exist: {root}", file=sys.stderr)
        return files

    all_paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            all_paths.append(os.path.join(dirpath, fn))

    total = len(all_paths)
    for idx, fpath in enumerate(all_paths):
        try:
            st = os.stat(fpath)
        except OSError:
            continue

        fn = os.path.basename(fpath)
        ext = get_ext(fn)
        fi = FileInfo(
            path=fpath,
            name=fn,
            ext=ext,
            size=st.st_size,
            is_video=ext in VIDEO_EXTS,
            is_audio=ext in AUDIO_EXTS,
            is_image=ext in IMAGE_EXTS,
        )

        if compute_hash and fi.size > 0:
            fi.hash_md5 = compute_md5(fpath)

        if compute_duration and fi.is_video:
            fi.duration = get_video_duration(fpath)

        files.append(fi)

        if progress_callback and (idx + 1) % 100 == 0:
            progress_callback(idx + 1, total)

    return files


def scan_folders(root: str, compute_hash: bool = True, progress_callback=None) -> Tuple[Dict[str, FolderInfo], List[FileInfo]]:
    root_path = Path(root).resolve()
    folder_map: Dict[str, FolderInfo] = {}
    all_files: List[FileInfo] = []

    folder_map[str(root_path)] = FolderInfo(
        id=str(root_path),
        path=str(root_path),
        name=root_path.name,
        parent_id=None,
        depth=0,
        is_root=True,
    )

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath_resolved = str(Path(dirpath).resolve())
        if dirpath_resolved not in folder_map:
            folder_map[dirpath_resolved] = FolderInfo(
                id=dirpath_resolved,
                path=dirpath,
                name=os.path.basename(dirpath),
                parent_id=str(Path(dirpath).parent.resolve()),
                depth=dirpath.count(os.sep) - str(root_path).count(os.sep),
            )

        parent_info = folder_map[dirpath_resolved]
        parent_info.sub_folder_ids = [str(Path(dirpath).resolve() / dn) for dn in dirnames]

        for fn in filenames:
            fpath = os.path.join(dirpath, fn)
            try:
                st = os.stat(fpath)
            except OSError:
                continue

            ext = get_ext(fn)
            fi = FileInfo(
                path=fpath, name=fn, ext=ext, size=st.st_size,
                is_video=ext in VIDEO_EXTS, is_audio=ext in AUDIO_EXTS, is_image=ext in IMAGE_EXTS,
            )

            if compute_hash and fi.size > 0:
                fi.hash_md5 = compute_md5(fpath)
            if fi.is_video:
                fi.duration = get_video_duration(fpath)

            all_files.append(fi)

            fp = fi.fingerprint
            cur_id = dirpath_resolved
            while cur_id and cur_id in folder_map:
                node = folder_map[cur_id]
                node.size += fi.size
                node.files.append(fp)
                cur_id = node.parent_id

    return folder_map, all_files


# ── Name Cleaning Engine (ported from PikPak Enhancement Master) ──

def clean_name_ad(name: str, loose: bool = False) -> str:
    clean = name
    clean = re.sub(r'^【[^】]+】 *[-_.]? *', '', clean)
    clean = re.sub(r'^[a-z0-9-]+[.](?:com|net|org|cc|xyz|vip|top|la) +', '', clean, flags=re.IGNORECASE)
    ad_kw = r"(?:[.]com|[.]net|[.]org|[.]cc|[.]xyz|[.]vip|[.]top|[.]la|2048|www[.])"
    clean = re.sub(r'^.*?' + ad_kw + r'.*?(?:@|--+|_\\s)', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'^[a-z0-9.-]+' + ad_kw + r'-', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'^(?:精品加群|福利合集)[0-9]+[-_]+ *', '', clean)
    clean = re.sub(r'^[-_. ,，:：;；\U0001F300-\U0001F9FF]+', '', clean)

    pairs = [('【', '】'), ('[', ']'), ('《', '》'), ('<', '>'), ('（', '）'), ('(', ')'), ('{', '}')]
    for left, right in pairs:
        idx_r = clean.find(right)
        idx_l = clean.find(left)
        if 0 < idx_r <= 10 and (idx_l == -1 or idx_l > idx_r):
            clean = left + clean
        chars = list(clean)
        stack = []
        to_remove = set()
        for i, c in enumerate(chars):
            if c == left:
                stack.append(i)
            elif c == right:
                if stack:
                    stack.pop()
                else:
                    to_remove.add(i)
        stack_set = set(stack)
        to_remove |= stack_set
        if to_remove:
            clean = "".join(c for i, c in enumerate(chars) if i not in to_remove)

    quote_count = clean.count("'")
    if quote_count % 2 != 0:
        clean = clean.replace("'", "", 1)

    last_dot = clean.rfind('.')
    if last_dot > 0:
        clean = clean[:last_dot]

    clean = re.sub(r'\s*[-_ .．。]*\s*(?:\(\s*\d+\s*\)|（\s*\d+\s*）|\[\s*\d+\s*\]|【\s*\d+\s*】)\s*$', '', clean, flags=re.UNICODE)
    clean = re.sub(r'\s*(?:[-_ .．。]*\s*)?(?:副本|复制|拷贝|拷貝|コピー|複製|복사본|사본|복사)\s*(?:\d+|\(\s*\d+\s*\)|（\s*\d+\s*）|\[\s*\d+\s*\]|【\s*\d+\s*】)?\s*$', '', clean, flags=re.UNICODE)
    clean = re.sub(r'\s*[-_ .．。]+\s*(?:copy|duplicate|dup|salinan|salin|duplikat)\s*(?:\d+|\(\s*\d+\s*\)|（\s*\d+\s*）|\[\s*\d+\s*\]|【\s*\d+\s*】)?\s*$', '', clean, flags=re.IGNORECASE)
    clean = clean.strip()

    result = clean.lower().strip() if clean else name.rsplit('.', 1)[0].lower().strip()
    if loose:
        result = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', result)
    return result


def clean_folder_name(name: str, loose: bool = False) -> str:
    return clean_name_ad(name, loose=loose)


# ── File Dedup: 3-Stage Pipeline ──

def dedup_files(root: str, strictness: str = "strict", file_types: Optional[List[str]] = None,
                progress_callback=None) -> List[Dict]:
    if file_types is None:
        file_types = ["video"]

    print(f"[Stage 0] Scanning files in {root} ...")
    files = scan_directory(root, compute_hash=True, compute_duration=True, progress_callback=progress_callback)

    type_filter = set()
    for ft in file_types:
        if ft == "video":
            type_filter |= VIDEO_EXTS
        elif ft == "image":
            type_filter |= IMAGE_EXTS
        elif ft == "audio":
            type_filter |= AUDIO_EXTS
        elif ft == "other":
            pass

    if file_types != ["other"]:
        candidates = [f for f in files if f.ext in type_filter]
    else:
        known = VIDEO_EXTS | AUDIO_EXTS | IMAGE_EXTS
        candidates = [f for f in files if f.ext not in known]

    print(f"  Found {len(candidates)} candidate files")

    groups = []
    assigned: Set[str] = set()

    # Stage 1: Hash exact match
    print("[Stage 1] Hash exact match ...")
    hash_map: Dict[str, List[FileInfo]] = defaultdict(list)
    for f in candidates:
        if f.hash_md5:
            key = f"{f.hash_md5}_{f.size}"
            hash_map[key].append(f)

    for key, items in hash_map.items():
        if len(items) > 1:
            ids = [f.path for f in items]
            for i in ids:
                assigned.add(i)
            groups.append({"ids": ids, "type": "hash", "items": items})

    print(f"  Found {len(groups)} hash-duplicate groups")

    # Stage 2: Video duration similarity
    stage2_groups = []
    if "video" in file_types:
        print("[Stage 2] Video duration similarity ...")
        dur_threshold = 3.0 if strictness == "loose" else 2.0
        size_ratio_limit = 1.0 if strictness == "loose" else 0.10

        videos = [f for f in candidates if f.is_video and f.duration > 0 and f.path not in assigned]
        videos.sort(key=lambda x: x.duration)

        for i, root_v in enumerate(videos):
            if root_v.path in assigned:
                continue
            group = [root_v]
            for j in range(i + 1, len(videos)):
                target = videos[j]
                if target.path in assigned:
                    continue
                dur_diff = abs(target.duration - root_v.duration)
                if dur_diff > dur_threshold:
                    break
                if root_v.size > 0 and target.size > 0:
                    size_diff = abs(target.size - root_v.size)
                    max_base = max(target.size, root_v.size)
                    if size_diff / max_base > size_ratio_limit:
                        continue
                group.append(target)

            if len(group) > 1:
                ids = [f.path for f in group]
                for i2 in ids:
                    assigned.add(i2)
                stage2_groups.append({"ids": ids, "type": "duration", "items": group})

        groups.extend(stage2_groups)
        print(f"  Found {len(stage2_groups)} duration-similar groups")

    # Stage 3: Name similarity
    print("[Stage 3] Name similarity ...")
    remaining = [f for f in candidates if f.path not in assigned]
    name_map: Dict[str, List[FileInfo]] = defaultdict(list)
    for f in remaining:
        cleaned = clean_name_ad(f.name, loose=(strictness == "loose"))
        if cleaned:
            type_group = "video" if f.is_video else ("image" if f.is_image else "other")
            key = f"{type_group}|{cleaned}"
            name_map[key].append(f)

    stage3_groups = []
    size_ratio_limit = 1.0 if strictness == "loose" else 0.10
    for key, items in name_map.items():
        if len(items) <= 1:
            continue
        sorted_items = sorted(items, key=lambda x: x.size)
        current_group = [sorted_items[0]]
        temp_groups = []

        for j in range(1, len(sorted_items)):
            target = sorted_items[j]
            root_item = current_group[0]
            root_size = root_item.size
            target_size = target.size

            is_match = False
            if root_size == 0 and target_size == 0:
                is_match = True
            elif root_size > 0 and target_size > 0:
                size_diff = abs(target_size - root_size)
                max_base = max(target_size, root_size)
                if size_diff / max_base <= size_ratio_limit:
                    is_match = True

            if is_match:
                current_group.append(target)
            else:
                if len(current_group) > 1:
                    temp_groups.append(current_group)
                current_group = [target]

        if len(current_group) > 1:
            temp_groups.append(current_group)

        for grp in temp_groups:
            ids = [f.path for f in grp]
            stage3_groups.append({"ids": ids, "type": "name", "items": grp})

    groups.extend(stage3_groups)
    print(f"  Found {len(stage3_groups)} name-similar groups")
    print(f"\n[Total] {len(groups)} duplicate groups found")

    return groups


# ── Folder Dedup: Jaccard + IDF + Containment ──

def dedup_folders(root: str, algo: str = "sim", threshold: float = 0.9,
                  progress_callback=None) -> List[Dict]:
    print(f"[Folder Dedup] Scanning {root} with algo={algo}, threshold={threshold} ...")
    folder_map, _ = scan_folders(root, compute_hash=True, progress_callback=progress_callback)

    for node in folder_map.values():
        counts: Dict[str, int] = defaultdict(int)
        for h in node.files:
            counts[h] += 1
        node.file_counts = dict(counts)

    for node in folder_map.values():
        if node.parent_id and node.parent_id in folder_map:
            parent = folder_map[node.parent_id]
            if len(parent.files) == len(node.files):
                parent.is_shell = True

    folder_arr = [
        f for f in folder_map.values()
        if not f.is_shell and (len(f.files) >= 2 or f.size > 1024 * 1024)
    ]
    folder_arr.sort(key=lambda x: len(x.files), reverse=True)

    inverted_index: Dict[str, List[int]] = defaultdict(list)
    for i, f in enumerate(folder_arr):
        for h in f.file_counts:
            inverted_index[h].append(i)

    total_docs = len(folder_arr)
    weight_map: Dict[str, float] = {}
    for h, arr in inverted_index.items():
        df = len(arr)
        w = 0.05 if total_docs >= 20 and (df / total_docs) > 0.05 else 1.0
        weight_map[h] = w

    for f in folder_arr:
        wt = 0.0
        for h, count in f.file_counts.items():
            wt += count * weight_map.get(h, 1.0)
        f.weighted_total = wt

    groups = []
    assigned: Set[str] = set()

    if algo == "name":
        name_groups: Dict[str, List[FolderInfo]] = defaultdict(list)
        for f in folder_arr:
            k = clean_folder_name(f.name)
            name_groups[k].append(f)

        size_ratio_limit = 0.10 if threshold >= 0.5 else float("inf")
        for k, items in name_groups.items():
            if len(items) <= 1:
                continue
            sorted_items = sorted(items, key=lambda x: x.size)
            current_group = [sorted_items[0]]
            for j in range(1, len(sorted_items)):
                target = sorted_items[j]
                root_item = current_group[0]
                root_size = root_item.size
                target_size = target.size
                is_match = (root_size == 0 and target_size == 0)
                if not is_match and root_size > 0 and target_size > 0:
                    is_match = abs(target_size - root_size) / max(target_size, root_size) <= size_ratio_limit
                if is_match:
                    current_group.append(target)
                else:
                    if len(current_group) > 1:
                        groups.append({"ids": [f.id for f in current_group], "type": "name", "folders": current_group})
                        for f in current_group:
                            assigned.add(f.id)
                    current_group = [target]
            if len(current_group) > 1:
                groups.append({"ids": [f.id for f in current_group], "type": "name", "folders": current_group})
                for f in current_group:
                    assigned.add(f.id)
    else:
        for i in range(len(folder_arr)):
            if folder_arr[i].id in assigned:
                continue
            f1 = folder_arr[i]
            group = [f1]

            candidate_indices = []
            marker = i + 1
            seen = set()
            for h in f1.file_counts:
                for idx in inverted_index.get(h, []):
                    if idx > i and idx not in seen and folder_arr[idx].id not in assigned:
                        seen.add(idx)
                        candidate_indices.append(idx)

            for j in candidate_indices:
                f2 = folder_arr[j]
                t1 = f1.weighted_total
                t2 = f2.weighted_total
                if t1 <= 0 or t2 <= 0:
                    continue

                max_s = max(t1, t2)
                min_s = min(t1, t2)
                if algo == "sim" and min_s / max_s < threshold:
                    continue

                intersect = 0.0
                f_small = f1 if len(f1.file_counts) < len(f2.file_counts) else f2
                f_large = f2 if len(f1.file_counts) < len(f2.file_counts) else f1
                for h in f_small.file_counts:
                    c_large = f_large.file_counts.get(h)
                    if c_large is not None:
                        c_small = f_small.file_counts[h]
                        intersect += min(c_small, c_large) * weight_map.get(h, 1.0)

                min_total = min(t1, t2)
                union = t1 + t2 - intersect
                sim = (intersect / min_total) if algo == "contain" else (intersect / union if union > 0 else 0)

                if sim >= threshold:
                    group.append(f2)

            if len(group) > 1:
                groups.append({"ids": [f.id for f in group], "type": algo, "folders": group})
                for f in group:
                    assigned.add(f.id)

    print(f"[Folder Dedup] Found {len(groups)} duplicate folder groups")
    return groups


# ── Batch Rename ──

def batch_rename(root: str, mode: str = "ad_remove", pattern: str = "Video {n}",
                 find: str = "", replace: str = "", use_regex: bool = False,
                 case_convert: str = "", dry_run: bool = True) -> List[Dict]:
    changes = []
    root_path = Path(root)

    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            old_path = os.path.join(dirpath, fn)
            ext = get_ext(fn)
            name_without_ext = fn[:fn.rfind(".")] if "." in fn else fn

            new_name = fn

            if mode == "ad_remove":
                cleaned = clean_name_ad(fn)
                if cleaned and cleaned != fn.lower().rsplit(".", 1)[0].strip():
                    new_name = cleaned + ("." + ext if ext else "")

            elif mode == "pattern":
                counter = len(changes) + 1
                new_base = pattern.replace("{n}", str(counter).zfill(2))
                new_name = new_base + ("." + ext if ext else "")

            elif mode == "replace":
                if use_regex:
                    new_base = re.sub(find, replace, name_without_ext)
                else:
                    new_base = name_without_ext.replace(find, replace)
                new_name = new_base + ("." + ext if ext else "")

            elif mode == "format":
                new_base = name_without_ext
                if case_convert == "lower":
                    new_base = new_base.lower()
                elif case_convert == "upper":
                    new_base = new_base.upper()
                elif case_convert == "title":
                    new_base = new_base.title()
                new_name = new_base + ("." + ext if ext else "")

            if new_name != fn:
                new_path = os.path.join(dirpath, new_name)
                changes.append({"old_path": old_path, "new_path": new_path, "old_name": fn, "new_name": new_name})

                if not dry_run:
                    try:
                        os.rename(old_path, new_path)
                    except OSError as e:
                        print(f"  Error renaming {fn} -> {new_name}: {e}")

    if dry_run:
        print(f"[Dry Run] {len(changes)} files would be renamed:")
    else:
        print(f"[Applied] {len(changes)} files renamed:")

    for c in changes[:50]:
        print(f"  {c['old_name']} -> {c['new_name']}")
    if len(changes) > 50:
        print(f"  ... and {len(changes) - 50} more")

    return changes


# ── Empty Folder Pruning (Bottom-Up Bubble) ──

def prune_empty_folders(root: str, dry_run: bool = True) -> List[str]:
    print(f"[Prune] Scanning empty folders in {root} ...")
    folder_map: Dict[str, Dict] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        has_files = any(os.path.isfile(os.path.join(dirpath, fn)) for fn in filenames)
        depth = dirpath.count(os.sep) - root.count(os.sep)
        folder_map[dirpath] = {
            "path": dirpath,
            "depth": depth,
            "has_files": has_files,
            "sub_dirs": [os.path.join(dirpath, dn) for dn in dirnames],
        }

    all_folders = sorted(folder_map.values(), key=lambda x: x["depth"], reverse=True)

    to_delete = []
    to_delete_set: Set[str] = set()

    for folder in all_folders:
        if folder["has_files"]:
            continue
        all_subs_will_be_deleted = all(sd in to_delete_set for sd in folder["sub_dirs"])
        if all_subs_will_be_deleted:
            to_delete_set.add(folder["path"])
            to_delete.append(folder["path"])

    print(f"  Found {len(to_delete)} empty folders")

    if not dry_run:
        for fpath in to_delete:
            try:
                shutil.rmtree(fpath)
                print(f"  Deleted: {fpath}")
            except OSError as e:
                print(f"  Error deleting {fpath}: {e}")
    else:
        print("[Dry Run] Would delete:")
        for fpath in to_delete:
            print(f"  {fpath}")

    return to_delete


# ── Directory Tree Export ──

def export_tree(root: str, output: Optional[str] = None) -> str:
    lines = []
    root_path = Path(root)
    prefix = ""

    def _walk(path: Path, prefix: str):
        entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            size_str = ""
            if entry.is_file():
                try:
                    size_str = f" ({fmt_size(entry.stat().st_size)})"
                except OSError:
                    pass
            lines.append(f"{prefix}{connector}{entry.name}{size_str}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                _walk(entry, prefix + extension)

    lines.append(root_path.name + "/")
    _walk(root_path, "")

    result = "\n".join(lines)

    if output:
        Path(output).write_text(result, encoding="utf-8")
        print(f"Tree exported to {output}")
    else:
        print(result)

    return result


# ── CLI ──

def cmd_scan(args):
    files = scan_directory(args.root, compute_hash=False, compute_duration=False)
    videos = [f for f in files if f.is_video]
    audios = [f for f in files if f.is_audio]
    images = [f for f in files if f.is_image]
    others = [f for f in files if not f.is_video and not f.is_audio and not f.is_image]

    total_size = sum(f.size for f in files)
    video_size = sum(f.size for f in videos)

    print(f"\n{'='*60}")
    print(f"  Scan Results: {args.root}")
    print(f"{'='*60}")
    print(f"  Total files:  {len(files):>8}  ({fmt_size(total_size)})")
    print(f"  Video files:  {len(videos):>8}  ({fmt_size(video_size)})")
    print(f"  Audio files:  {len(audios):>8}")
    print(f"  Image files:  {len(images):>8}")
    print(f"  Other files:  {len(others):>8}")
    print(f"{'='*60}")

    if videos:
        print(f"\nTop 20 largest videos:")
        for v in sorted(videos, key=lambda x: x.size, reverse=True)[:20]:
            print(f"  {fmt_size(v.size):>12}  {v.name}")


def cmd_dedup(args):
    file_types = args.types.split(",") if args.types else ["video"]
    groups = dedup_files(args.root, strictness=args.strictness, file_types=file_types)

    protect = load_protect_list(args.root)
    protected_count = 0

    for i, group in enumerate(groups):
        group_type = group["type"]
        items = group["items"]
        print(f"\n── Group {i+1} ({group_type}) ──")
        for item in items:
            is_protected = item.name in protect or item.path in protect
            tag = " [PROTECTED]" if is_protected else ""
            if is_protected:
                protected_count += 1
            dur_str = f"  dur={item.duration:.1f}s" if item.duration > 0 else ""
            print(f"  {fmt_size(item.size):>12}{dur_str}  {item.name}{tag}")

    if protected_count > 0:
        print(f"\n  {protected_count} protected files skipped from deletion suggestions")


def cmd_folder_dedup(args):
    groups = dedup_folders(args.root, algo=args.algo, threshold=args.threshold)

    for i, group in enumerate(groups):
        group_type = group["type"]
        folders = group["folders"]
        print(f"\n── Folder Group {i+1} ({group_type}) ──")
        for f in folders:
            print(f"  {fmt_size(f.size):>12}  {len(f.files)} files  {f.path}")


def cmd_rename(args):
    batch_rename(
        args.root, mode=args.mode, pattern=args.pattern,
        find=args.find, replace=args.replace, use_regex=args.regex,
        case_convert=args.case, dry_run=args.dry_run,
    )


def cmd_prune(args):
    prune_empty_folders(args.root, dry_run=args.dry_run)


def cmd_tree(args):
    export_tree(args.root, output=args.output)


def cmd_protect(args):
    if args.add:
        protect = load_protect_list(args.root)
        for item in args.add:
            protect.add(item)
        save_protect_list(args.root, protect)
        print(f"Added {len(args.add)} items to protection list")
    elif args.remove:
        protect = load_protect_list(args.root)
        for item in args.remove:
            protect.discard(item)
        save_protect_list(args.root, protect)
        print(f"Removed {len(args.remove)} items from protection list")
    elif args.list:
        protect = load_protect_list(args.root)
        if protect:
            print(f"Protection list ({len(protect)} items):")
            for item in sorted(protect):
                print(f"  {item}")
        else:
            print("Protection list is empty")
    else:
        print("Use --add, --remove, or --list")


def main():
    parser = argparse.ArgumentParser(description="Video Organizer — deduplicate, rename, and clean video files")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # scan
    p_scan = subparsers.add_parser("scan", help="Scan and list video files")
    p_scan.add_argument("root", help="Root directory to scan")

    # dedup
    p_dedup = subparsers.add_parser("dedup", help="Find duplicate video files")
    p_dedup.add_argument("root", help="Root directory to scan")
    p_dedup.add_argument("--strictness", choices=["strict", "loose"], default="strict", help="Matching strictness")
    p_dedup.add_argument("--types", default="video", help="File types: video,image,audio,other (comma-separated)")

    # folder-dedup
    p_fdedup = subparsers.add_parser("folder-dedup", help="Find duplicate folders")
    p_fdedup.add_argument("root", help="Root directory to scan")
    p_fdedup.add_argument("--algo", choices=["name", "sim", "contain"], default="sim", help="Similarity algorithm")
    p_fdedup.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold (0.0-1.0)")

    # rename
    p_rename = subparsers.add_parser("rename", help="Batch rename files")
    p_rename.add_argument("root", help="Root directory")
    p_rename.add_argument("--mode", choices=["ad_remove", "pattern", "replace", "format"], default="ad_remove")
    p_rename.add_argument("--pattern", default="Video {n}", help="Pattern for episode numbering")
    p_rename.add_argument("--find", default="", help="Text to find (replace mode)")
    p_rename.add_argument("--replace", default="", help="Replacement text (replace mode)")
    p_rename.add_argument("--regex", action="store_true", help="Use regex for find")
    p_rename.add_argument("--case", choices=["lower", "upper", "title"], default="", help="Case conversion (format mode)")
    p_rename.add_argument("--dry-run", action="store_true", default=True, help="Preview only (default)")
    p_rename.add_argument("--apply", dest="dry_run", action="store_false", help="Apply changes")

    # prune
    p_prune = subparsers.add_parser("prune", help="Remove empty folders")
    p_prune.add_argument("root", help="Root directory")
    p_prune.add_argument("--dry-run", action="store_true", default=True, help="Preview only (default)")
    p_prune.add_argument("--apply", dest="dry_run", action="store_false", help="Apply changes")

    # tree
    p_tree = subparsers.add_parser("tree", help="Export directory tree")
    p_tree.add_argument("root", help="Root directory")
    p_tree.add_argument("-o", "--output", help="Output file path")

    # protect
    p_protect = subparsers.add_parser("protect", help="Manage protection list")
    p_protect.add_argument("root", help="Root directory")
    p_protect.add_argument("--add", nargs="+", help="Add items to protection list")
    p_protect.add_argument("--remove", nargs="+", help="Remove items from protection list")
    p_protect.add_argument("--list", action="store_true", help="List protected items")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    commands = {
        "scan": cmd_scan,
        "dedup": cmd_dedup,
        "folder-dedup": cmd_folder_dedup,
        "rename": cmd_rename,
        "prune": cmd_prune,
        "tree": cmd_tree,
        "protect": cmd_protect,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
