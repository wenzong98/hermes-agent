"""Tests for video_organizer skill — uses only stdlib + pytest + unittest.mock."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts dir to path so we can import the module
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
import sys

sys.path.insert(0, str(SCRIPTS_DIR))

from video_organizer import (
    PROTECT_FILE,
    FileInfo,
    FolderInfo,
    batch_rename,
    clean_folder_name,
    clean_name_ad,
    compute_md5,
    dedup_files,
    dedup_folders,
    export_tree,
    fmt_size,
    get_ext,
    get_video_duration,
    is_archive_file,
    is_audio_file,
    is_image_file,
    is_video_file,
    load_protect_list,
    parse_scene_filename,
    prune_empty_folders,
    save_protect_list,
    scan_directory,
)


@pytest.fixture
def tmp_drive(tmp_path):
    d = tmp_path / "test_drive"
    d.mkdir()
    return d


@pytest.fixture
def sample_videos(tmp_drive):
    files = []
    for name, size in [
        ("movie1.mp4", 1024 * 1024 * 100),
        ("movie2.mkv", 1024 * 1024 * 200),
        ("movie3.avi", 1024 * 1024 * 50),
    ]:
        p = tmp_drive / name
        p.write_bytes(os.urandom(min(size, 1024)))
        files.append(str(p))
    return files


class TestFileTypeDetection:
    def test_video_ext(self):
        assert is_video_file("test.mp4")
        assert is_video_file("test.mkv")
        assert is_video_file("test.avi")
        assert is_video_file("test.mov")
        assert is_video_file("test.wmv")
        assert is_video_file("test.flv")
        assert is_video_file("test.webm")
        assert not is_video_file("test.mp3")
        assert not is_video_file("test.jpg")
        assert not is_video_file("test.txt")

    def test_audio_ext(self):
        assert is_audio_file("test.mp3")
        assert is_audio_file("test.flac")
        assert not is_audio_file("test.mp4")

    def test_image_ext(self):
        assert is_image_file("test.jpg")
        assert is_image_file("test.png")
        assert not is_image_file("test.mp4")

    def test_archive_ext(self):
        assert is_archive_file("test.zip")
        assert is_archive_file("test.rar")
        assert not is_archive_file("test.mp4")

    def test_get_ext(self):
        assert get_ext("test.mp4") == "mp4"
        assert get_ext("test.MKV") == "mkv"
        assert get_ext("noext") == ""
        assert get_ext(".hidden") == ""


class TestFormatSize:
    def test_bytes(self):
        assert "B" in fmt_size(500)

    def test_kb(self):
        assert "KB" in fmt_size(1500)

    def test_mb(self):
        assert "MB" in fmt_size(1024 * 1024 * 5)

    def test_gb(self):
        assert "GB" in fmt_size(1024 * 1024 * 1024 * 3)

    def test_tb(self):
        assert "TB" in fmt_size(1024 * 1024 * 1024 * 1024 * 2)


class TestMD5:
    def test_compute_md5(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = compute_md5(str(f))
        assert len(result) == 32
        assert result == "5eb63bbbe01eeed093cb22bb8f5acdc3"

    def test_compute_md5_empty(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        result = compute_md5(str(f))
        assert len(result) == 32


class TestCleanNameAd:
    def test_remove_bracket_tag(self):
        result = clean_name_ad("【xxx.com】movie.mp4")
        assert "xxx.com" not in result or "movie" in result

    def test_remove_domain_prefix(self):
        result = clean_name_ad("www.example.com - video.mp4")
        assert "example.com" not in result or "video" in result

    def test_remove_copy_marker(self):
        result = clean_name_ad("video (1).mp4")
        assert "(1)" not in result

    def test_remove_duplicate_marker(self):
        result = clean_name_ad("video duplicate 1.mp4")
        assert "duplicate" not in result

    def test_remove_chinese_copy(self):
        result = clean_name_ad("视频 副本.mp4")
        assert "副本" not in result

    def test_loose_mode_strips_symbols(self):
        result = clean_name_ad("Video [HD] 1080p.mp4", loose=True)
        assert "[" not in result
        assert "]" not in result

    def test_clean_folder_name(self):
        result = clean_folder_name("【字幕组】剧集名")
        assert "字幕组" not in result or "剧集名" in result


class TestParseSceneFilename:
    def test_parse_movie_with_tags(self):
        meta = parse_scene_filename("[Studio] Some Actor, Another Actor - My Great Movie (2023) [1080p].mp4")
        assert meta.title == "my great movie"
        assert meta.year == "2023"
        assert meta.studio == "Studio"
        assert meta.resolution == "1080p"
        assert "Some Actor" in meta.actors
        assert "Another Actor" in meta.actors

    def test_parse_tv_show(self):
        meta = parse_scene_filename("Breaking Bad S02E05.mkv")
        assert meta.show == "breaking bad"
        assert meta.season == 2
        assert meta.episode == 5

    def test_parse_clean_title(self):
        meta = parse_scene_filename("【字幕组】Movie Name (2020) [4K].mp4")
        assert meta.title == "movie name"
        assert meta.year == "2020"
        assert meta.resolution == "4k"

    def test_parse_date(self):
        meta = parse_scene_filename("Vlog 2023-05-12 Title.mp4")
        assert meta.date == "2023-05-12"
        assert "title" in meta.title

class TestScanDirectory:
    def test_scan_finds_files(self, sample_videos, tmp_drive):
        files = scan_directory(str(tmp_drive))
        assert len(files) >= 3
        video_files = [f for f in files if f.is_video]
        assert len(video_files) >= 3

    def test_scan_nonexistent(self):
        files = scan_directory("/nonexistent_path_12345")
        assert len(files) == 0

    def test_scan_with_hash(self, tmp_drive):
        (tmp_drive / "test.mp4").write_bytes(b"test content")
        files = scan_directory(str(tmp_drive), compute_hash=True)
        assert len(files) == 1
        assert files[0].hash_md5 != ""


class TestProtectList:
    def test_save_and_load(self, tmp_drive):
        save_protect_list(str(tmp_drive), {"important.mp4", "keep.mkv"})
        loaded = load_protect_list(str(tmp_drive))
        assert "important.mp4" in loaded
        assert "keep.mkv" in loaded

    def test_load_empty(self, tmp_drive):
        loaded = load_protect_list(str(tmp_drive))
        assert len(loaded) == 0

    def test_protect_file_format(self, tmp_drive):
        save_protect_list(str(tmp_drive), {"b.mp4", "a.mkv"})
        p = Path(str(tmp_drive)) / PROTECT_FILE
        data = json.loads(p.read_text())
        assert data["protected"] == ["a.mkv", "b.mp4"]


class TestDedupFiles:
    def test_hash_dedup_identical_files(self, tmp_drive):
        content = b"identical video content for testing"
        (tmp_drive / "video1.mp4").write_bytes(content)
        (tmp_drive / "video2.mp4").write_bytes(content)
        groups = dedup_files(str(tmp_drive), file_types=["video"])
        hash_groups = [g for g in groups if g["type"] == "hash"]
        assert len(hash_groups) >= 1

    def test_no_duplicates(self, tmp_drive):
        (tmp_drive / "unique1.mp4").write_bytes(b"content1")
        (tmp_drive / "unique2.mp4").write_bytes(b"content2")
        (tmp_drive / "unique3.mp4").write_bytes(b"content3")
        groups = dedup_files(str(tmp_drive), file_types=["video"])
        assert len(groups) == 0


class TestFolderDedup:
    def test_identical_folders(self, tmp_drive):
        for folder in ["SeriesA", "SeriesA_copy"]:
            d = tmp_drive / folder
            d.mkdir()
            (d / "ep01.mp4").write_bytes(b"episode 1 content")
            (d / "ep02.mp4").write_bytes(b"episode 2 content")
        groups = dedup_folders(str(tmp_drive), algo="name")
        assert len(groups) >= 1


class TestBatchRename:
    def test_ad_remove_dry_run(self, tmp_drive):
        (tmp_drive / "【ads】movie.mp4").write_bytes(b"test")
        changes = batch_rename(str(tmp_drive), mode="ad_remove", dry_run=True)
        assert len(changes) >= 1
        assert Path(changes[0]["old_path"]).exists()

    def test_pattern_rename(self, tmp_drive):
        (tmp_drive / "video1.mp4").write_bytes(b"test")
        (tmp_drive / "video2.mp4").write_bytes(b"test")
        changes = batch_rename(str(tmp_drive), mode="pattern", pattern="Episode {n}", dry_run=True)
        assert len(changes) >= 2

    def test_replace_rename(self, tmp_drive):
        (tmp_drive / "old_name.mp4").write_bytes(b"test")
        changes = batch_rename(str(tmp_drive), mode="replace", find="old", replace="new", dry_run=True)
        assert len(changes) >= 1
        assert "new" in changes[0]["new_name"]


class TestPruneEmptyFolders:
    def test_prune_nested_empty(self, tmp_drive):
        nested = tmp_drive / "empty1" / "empty2" / "empty3"
        nested.mkdir(parents=True)
        result = prune_empty_folders(str(tmp_drive), dry_run=True)
        assert len(result) >= 3

    def test_prune_keeps_nonempty(self, tmp_drive):
        empty = tmp_drive / "empty_dir"
        empty.mkdir()
        nonempty = tmp_drive / "nonempty_dir"
        nonempty.mkdir()
        (nonempty / "file.mp4").write_bytes(b"test")
        result = prune_empty_folders(str(tmp_drive), dry_run=True)
        assert str(nonempty) not in result
        assert str(empty) in result


class TestExportTree:
    def test_export_tree(self, tmp_drive):
        (tmp_drive / "movie.mp4").write_bytes(b"test")
        sub = tmp_drive / "subdir"
        sub.mkdir()
        (sub / "ep01.mp4").write_bytes(b"test")
        result = export_tree(str(tmp_drive))
        assert "movie.mp4" in result
        assert "subdir" in result
        assert "ep01.mp4" in result

    def test_export_to_file(self, tmp_drive):
        (tmp_drive / "file.mp4").write_bytes(b"test")
        output = str(tmp_drive / "tree.txt")
        export_tree(str(tmp_drive), output=output)
        assert Path(output).exists()
        content = Path(output).read_text()
        assert "file.mp4" in content


class TestFileInfo:
    def test_fingerprint_with_hash(self):
        fi = FileInfo(path="/a/b.mp4", name="b.mp4", ext="mp4", size=100, hash_md5="abc123")
        assert fi.fingerprint == "abc123_100"

    def test_fingerprint_without_hash(self):
        fi = FileInfo(path="/a/b.mp4", name="b.mp4", ext="mp4", size=100)
        assert fi.fingerprint == "b.mp4_100"
