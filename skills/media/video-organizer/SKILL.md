---
name: video-organizer
description: "Deduplicate, rename, and clean video files on local/external drives. Invoke when user mentions video dedup, file cleanup, or drive organizing."
version: 1.0.0
author: digbug82 + hermes-agent
license: AGPL-3.0-or-later
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Video, Deduplication, FileManagement, Organization, Cleanup]
    category: media
prerequisites:
  commands: [ffprobe]
  optional_commands: [md5, shasum]
---

# Video Organizer Skill

Scan, deduplicate, rename, and clean video files on local or external drives. Inspired by PikPak Enhancement Master's file analysis engine, adapted for local filesystem use.

## When to Use

- User wants to find and remove duplicate video files on a drive
- User wants to clean up ad prefixes / copy markers from video filenames
- User wants to find duplicate folders (same series from different sources)
- User wants to prune empty folders after cleanup
- User wants to export a directory tree of their video collection
- User mentions "视频整理", "视频查重", "重复视频", "清理空文件夹", "批量重命名"

## Prerequisites

- **ffprobe** (from ffmpeg) — required for video duration detection in dedup
- **md5** or **shasum** — for file hash computation (platform built-in)

Install ffmpeg (includes ffprobe):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## How to Run

All operations are driven through the main script `scripts/video_organizer.py`.

```bash
# Scan a drive and show video summary
python scripts/video_organizer.py scan /Volumes/MyDrive

# Find duplicate videos (3-stage: hash → duration → name)
python scripts/video_organizer.py dedup /Volumes/MyDrive

# Find duplicate videos with loose matching
python scripts/video_organizer.py dedup /Volumes/MyDrive --strictness loose

# Find duplicate folders (Jaccard similarity)
python scripts/video_organizer.py folder-dedup /Volumes/MyDrive

# Clean ad prefixes from filenames (dry run)
python scripts/video_organizer.py rename /Volumes/MyDrive --mode ad_remove --dry-run

# Clean ad prefixes from filenames (apply)
python scripts/video_organizer.py rename /Volumes/MyDrive --mode ad_remove

# Batch rename with episode numbering
python scripts/video_organizer.py rename /Volumes/MyDrive --mode pattern --pattern "Episode {n}"

# Regex find-and-replace rename
python scripts/video_organizer.py rename /Volumes/MyDrive --mode replace --find "www\.example\.com" --replace ""

# Prune empty folders (bottom-up bubble)
python scripts/video_organizer.py prune /Volumes/MyDrive --dry-run

# Export directory tree
python scripts/video_organizer.py tree /Volumes/MyDrive -o tree.txt

# Protect specific files from deletion
python scripts/video_organizer.py protect /Volumes/MyDrive --add "important_movie.mkv"
python scripts/video_organizer.py protect /Volumes/MyDrive --list
```

## Quick Reference

| Command | Description |
|---------|-------------|
| `scan` | Recursively scan and list all video files with metadata |
| `dedup` | 3-stage file dedup: hash → duration → name similarity |
| `folder-dedup` | Folder-level dedup: name / Jaccard / containment |
| `rename` | Batch rename: ad_remove / pattern / replace / format |
| `prune` | Remove empty folders (bottom-up bubble algorithm) |
| `tree` | Export directory tree structure |
| `protect` | Manage protection list (prevent accidental deletion) |

## Dedup Stages

### Stage 1: Hash Exact Match
Computes MD5 hash + file size as fingerprint. Files with identical fingerprint are exact duplicates.

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
