---
name: video-organizer
description: "Deduplicate, rename, and clean video files on local/external drives. Invoke when user mentions video dedup, file cleanup, or drive organizing."
version: 1.1.0
author: digbug82 + hermes-agent
license: AGPL-3.0-or-later
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Video, Deduplication, FileManagement, Organization, Cleanup]
    category: media
prerequisites:
  commands: [ffprobe, ffmpeg]
  python_packages: [ImageHash, Pillow]
---

# Video Organizer Skill

Scan, deduplicate, rename, and clean video files on local or external drives. Inspired by PikPak Enhancement Master, Stash, and Jellyfin's file analysis engines.

## When to Use

- User wants to find and remove duplicate video files on a drive
- User wants to find duplicate videos that are visually similar but have different resolutions/encodings (Perceptual Hash)
- User wants to organize videos into Jellyfin/Emby compatible folder structures (`Show (Year)/Season 01/Show S01E01.mp4`)
- User wants to clean up ad prefixes / copy markers from video filenames
- User wants to find duplicate folders (same series from different sources)
- User wants to prune empty folders after cleanup
- User wants to export a directory tree of their video collection
- User mentions "视频整理", "视频查重", "重复视频", "清理空文件夹", "批量重命名", "Jellyfin目录规范"

## Prerequisites

- **ffmpeg** / **ffprobe** — required for video duration detection and frame extraction for phash
- **ImageHash** and **Pillow** — required for perceptual hash dedup (Stage 1.5)

Install system dependencies:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

Install Python dependencies:
```bash
pip install ImageHash Pillow
```

## How to Run

All operations are driven through the main script `scripts/video_organizer.py`.

```bash
# Scan a drive and show video summary
python scripts/video_organizer.py scan /Volumes/MyDrive

# Scan a drive and parse scene metadata (Title, Year, Studio, Season, Episode, Resolution)
python scripts/video_organizer.py scan /Volumes/MyDrive --parse

# Find duplicate videos (4-stage: hash → phash → duration → name)
python scripts/video_organizer.py dedup /Volumes/MyDrive --phash

# Find duplicate folders (Jaccard similarity)
python scripts/video_organizer.py folder-dedup /Volumes/MyDrive

# Clean ad prefixes from filenames (apply)
python scripts/video_organizer.py rename /Volumes/MyDrive --mode ad_remove --apply

# Organize files into Jellyfin/Emby compatible folder structure (preview)
python scripts/video_organizer.py organize /Volumes/MyDrive --target-dir /Volumes/MyDrive/Media

# Organize files into Jellyfin/Emby compatible folder structure (apply)
python scripts/video_organizer.py organize /Volumes/MyDrive --target-dir /Volumes/MyDrive/Media --apply

# Prune empty folders (apply)
python scripts/video_organizer.py prune /Volumes/MyDrive --apply
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `scan` | Recursively scan and list all video files with metadata (optional `--parse`) |
| `dedup` | 4-stage file dedup: hash → phash → duration → name similarity |
| `folder-dedup` | Folder-level dedup: name / Jaccard / containment |
| `rename` | Batch rename: ad_remove / pattern / replace / format |
| `organize` | Organize files into Jellyfin/Emby compatible folder structures |
| `prune` | Remove empty folders (bottom-up bubble algorithm) |
| `tree` | Export directory tree structure |
| `protect` | Manage protection list (prevent accidental deletion) |

## Dedup Stages

### Stage 1: Hash Exact Match
Computes MD5 hash + file size as fingerprint. Files with identical fingerprint are exact duplicates.

### Stage 1.5: Perceptual Hash (phash)
Extracts 3 frames across the video using `ffmpeg`, computes average hash using `ImageHash`, and compares Hamming distances. Excellent for finding identical videos with different resolutions or watermarks.

### Stage 2: Video Duration Similarity
Uses `ffprobe` to get duration. Videos with duration difference ≤ 2s (strict) or ≤ 3s (loose) AND size ratio ≤ 10% are considered similar duplicates.

### Stage 3: Name Similarity
Cleans filenames (removes ad prefixes, copy markers, bracket tags) then groups by normalized name + size ratio.

## Folder Dedup Algorithms

| Algorithm | Flag | Description |
|-----------|------|-------------|
| Name match | `--algo name` | Cleaned folder names match + size ratio check |
| Jaccard similarity | `--algo sim` | TF-IDF weighted Jaccard on file fingerprints |
| Containment | `--algo contain` | `intersection / min_total` — detects subset folders |

## Rename Modes

| Mode | Flag | Description |
|------|------|-------------|
| Ad remove | `--mode ad_remove` | Strip ad domain prefixes, bracket tags, copy markers |
| Pattern | `--mode pattern` | Episode numbering with `{n}` placeholder |
| Replace | `--mode replace` | Regex or plain text find-and-replace |
| Format | `--mode format` | Case conversion + full/half-width conversion |

## Safety

- All destructive operations support `--dry-run` (preview changes without applying)
- Protection list prevents protected files from being included in dedup results
- Deletions use `send2trash` (moves to trash, not permanent delete) when available, falls back to `rm`
- Empty folder pruning uses bottom-up bubble: only deletes a folder when all its subfolders are also empty

## Verification

After running dedup or rename, verify with:
```bash
# Re-scan to confirm changes
python scripts/video_organizer.py scan /Volumes/MyDrive

# Check protection list
python scripts/video_organizer.py protect /Volumes/MyDrive --list
```
