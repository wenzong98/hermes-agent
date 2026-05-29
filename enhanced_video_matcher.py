#!/usr/bin/env python3
"""
Enhanced Video Matcher v2 - Deep Analysis Edition
基于10000+文件名的深度分析，包含全面的博主名称识别规则

新增识别模式（相比v1）:
- 来源站前缀: Porn_2048.cc-, gc2048.com-, guochan2048.com-, V_2048.cc-, 1_2048.cc-, xyfg5_2048.cc-
- 博主名_来源站_序号: fkabuto_56234, luckydog7_111_2048.cc, newyearst6_kcf9.com
- 博主名_子系列: bbw_ICIC_91_, bbw_ICIC_户外_
- 平台标记+博主名: Porn_91Tims, Porn_LuckyDog77
- 日文Myfans格式: 烈さんのプライベートSNS - myfans[マイファンズ]
- 番号格式: sivr00394, kavr00388, FC2-PPV-XXXXXXX, IPX-580, SNOS-131
- 日期+博主: 20250624_1937372275206074369_描述
- 博主名(序号): YoShiE冰块 (1), Little Sky (7)
- 描述_博主名_描述: 黑傑克_www.98T.la@描述
- 博主名_日期: ROSEWU_5月31日, fkabuto_2025年3月最新
- 中文描述【博主名】描述: 怒插小熊內褲妹_1【黑杰克】
- Emoji前缀: ⚫️新品福利, ✨国产泡良大神
- 移动硬盘_前缀: 移动硬盘_移动硬盘_806612_付费_描述
- 推特标记: #推特 #Enthus1asmm, #ICICIS, #fkabuto
- 91博主名格式: 91大庆哥, 91Tims, 91Mrber
- 博主名_子系列标记: fkabuto_111_2048.cc, luckydog7_222222_2048.cc
"""
import os
import re
import unicodedata
import subprocess
from collections import defaultdict
from datetime import datetime


BASE_PATH = "/Volumes/personal_folder-1/movie_a/中文"

GENERIC_FOLDERS = {
    "其他", "合集", "移动硬盘", "Porn", "gif", "AI增强", "IMG视频",
    "日期日记", "多P", "纯数字编号", "纯编号视频", "编号视频",
    "UUID编号视频", "重磅", "重磅核弹", "附大量聊天记录",
    "短片合集篇", "完整版合集", "gc2048", "2048.cc合集",
    "单视频合集", "绿帽NTR", "足控", "萝次元", "轻度猎奇",
    "公鸡俱乐部", "性爱记事本", "test_chinese_dst",
}

SOURCE_PREFIXES = [
    r'Porn_2048\.cc-',
    r'Porn_2048\.vip-',
    r'Porn_ri\.bi-',
    r'Porn_',
    r'gc2048\.com-',
    r'guochan2048\.com-',
    r'guochan2048\.com\s*-',
    r'V_2048\.cc-',
    r'1_2048\.cc-',
    r'xyfg5_2048\.cc-',
    r'视频_2048\.cc-',
    r'天然巨乳_天然巨乳_2048\.cc-',
    r'剧情演绎_2048\.cc-',
    r'2048\.cc-',
    r'2048\.vip-',
    r'2048\.cc_',
    r'rh2048\.com@',
    r'bbs2048\.org出品@',
    r'www\.98T\.la@',
    r'kcf9\.com-',
    r'4k2\.com@',
    r'4k2\.me@',
    r'madoubt\.com\s+\d+\.xyz\s+',
    r'hhd800\.com-',
    r'T66Y\.COM@',
    r'移动硬盘_移动硬盘_',
    r'移动硬盘_',
]

FOLDER_PATTERNS = [
    (r'^(.+?)-\d+[VM][\d.]*[GMK](?:【.*?】)?$', 'standard_nvng'),
    (r'^【[^】]*?-(.+?)】', 'platform_blogger'),
    (r'^(.+?)【.*?】$', 'blogger_with_tag'),
    (r'^@(.+)$', 'at_blogger'),
    (r'^(.{2,30})$', 'pure_name'),
]

FOLDER_PREFIX_PATTERNS = [
    r'^(?:推特|原创|约炮大神|泡良大神|约啪大神|新品|新档|付费|网黄|主播|OF|推|新流|国产|微博|台灣|SVIP|VIP|福利|91老博主|91)[-_ ]*(.+)$',
    r'^91(.+)$',
]

FILE_BLOGGER_PATTERNS = [
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_\d+_', 'blogger_source_seq'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_\d{4}年', 'blogger_date'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_[a-zA-Z]', 'blogger_subseries'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_\d{5,}', 'blogger_id_seq'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})-\d+[VM]', 'blogger_nvng'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_\(', 'blogger_paren'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_www\.', 'blogger_source'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_#', 'blogger_hashtag'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})_', 'blogger_prefix'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})\s*\(\d+\)', 'blogger_seq_paren'),
    (r'^([a-zA-Z][a-zA-Z0-9_]{2,30})\s+(?:最新|年|月)', 'blogger_date_desc'),
]

FILE_BRACKET_PATTERNS = [
    r'【(.+?)】',
    r'『(.+?)』',
    r'「(.+?)」',
]

FILE_HASHTAG_PATTERNS = [
    r'#(推特|Twitter)\s+#(.+?)[\s_]',
    r'#(.+?)[\s_]',
]

FILE_JAPANESE_MYFANS = r'^([^】]+?)さんのプライベートSNS'

FILE_91_PATTERN = r'^91([^\d][^】]{1,20}?)(?:\s|【|$)'

FILE_CHINESE_BLOGGER_PATTERNS = [
    (r'^([\u4e00-\u9fff]{2,10}?)_', 'cn_blogger_underscore'),
    (r'^([\u4e00-\u9fff]{2,10}?)（', 'cn_blogger_cn_paren'),
    (r'^([\u4e00-\u9fff]{2,10}?)\s*【', 'cn_blogger_bracket'),
    (r'【([\u4e00-\u9fff]{2,10}?)】', 'cn_bracket_blogger'),
    (r'_\d*【([\u4e00-\u9fff]{2,10}?)】', 'cn_suffix_bracket'),
    (r'^([\u4e00-\u9fff]{2,8}?)(?:最新|年|月|号|日)', 'cn_blogger_date'),
]

FILE_NANA_TAIPEI = r'^(娜娜|nana_taipei)[（(]nana_taipei[)）]'

FILE_YURI_PATTERN = r'^(?:onlyfans_)?(?:xoxo_)?[Yy]uri'

PREFIX_CLEAN = [
    '推特', '原创', '约炮大神', '泡良大神', '约啪大神', '新品', '新档',
    '付费', '网黄', '主播', 'OF', '推', '新流', '国产', '走马探全球',
    '微博', '泰国淫欲', '台灣', 'SVIP', 'VIP', '福利', '重磅',
    '顶级', '极品', '超人气', '超高颜值', '最新', '重磅核弹',
    '新品重磅', '新品福利', '精品', '独家', '爆火',
]

IGNORE_WORDS = {
    '视频', '合集', '合辑', '私拍', '订阅', '福利', '最新', '新品',
    '新流', '新档', '爆火', '原创', '约炮', '约啪', '泡良', '作品',
    '完整版', '精品', 'SVIP', 'VIP', '教程', '流出', '弹', '大学生',
    '户外', '体育生', '人妻', '反差', '网红', '自拍', '绿帽', 'NTR',
    '91', 'FC2', 'OnlyFans', 'MyFans', 'PikPak', '日语', '国产', '韩国',
    '欧美', '台湾', '露出', '巨乳', 'BBW', '情侣', '足控', '迷奸',
    '无套', '内射', '肌肉', 'Dom', 'JK', '制服', 'SM', '调教', '番号',
    'Part', 'part', 'MP4', 'mp4', 'AVI', 'avi', 'MKV', 'mkv',
    'new', 'NEW', 'img', 'IMG', 'vid', 'VID', 'mov', 'MOV',
    '回忆录', '回忆', '日记本', '日记', '备忘录', '回忆1', '回忆2',
    'mp4', 'v2', 'v1', '4K', 'HD', 'FHD', 'UHD', '8K',
    'Coomer', 'Fansly', 'source', 'Source', 'Lulustream',
    'restored', 'apo8', 'iris2', 'iris3', 'prob4', 'chf3',
}

NOISE_FILE_PATTERNS = [
    r'^\._',
    r'^V\s*\(\d+\)',
    r'^v\s*\(\d+\)',
    r'^1\s*\(\d+\)',
    r'^\d+\s*\(\d+\)',
    r'^V\d{3}$',
    r'^\d{1,3}\.\w+$',
    r'^\d{1,2}\.\w+$',
    r'^\d{1,2}\s*$',
    r'^[0-9a-f]{8}-',
    r'^\d{18,}$',
    r'^\d{4}_\d{2}_\d{2}',
    r'^\d{13,}_',
    r'^FC2-?PPV-?\d+',
    r'^FC2-?\d+',
    r'^[A-Z]+-\d+[-_]?\d*',
    r'^sivr\d+',
    r'^kavr\d+',
    r'^miab-?\d+',
    r'^mdvr\d+',
    r'^dsvr-?\d+',
    r'^IPVR-?\d+',
    r'^SNOS-?\d+',
    r'^IPX-?\d+',
    r'^ATID-?\d+',
    r'^MDVR-?\d+',
    r'^OGVN\d+',
    r'^YW\d+',
    r'^CF\d+',
    r'^RA-?\d+',
    r'^Post\s+by\s+',
    r'^Post\s+\'\'',
    r'^视频$',
    r'^至尊$',
    r'^内射$',
    r'^[\d.]+[GM]$', 
    r'^\d+月\d+日最新订阅付费视频',
    r'^9-10月\s*\(\d+\)',
    r'^5月27日最新订阅付费视频',
    r'^202[56]-\d+-\d+',
    r'^20250\d{10,}',
    r'^65\d{15,}',
    r'^66\d{15,}',
    r'^67\d{15,}',
    r'^70\d{15,}',
    r'^73\d{15,}',
    r'^75\d{15,}',
    r'^63\d{15,}',
    r'^64\d{15,}',
    r'^40[0-9a-f]{30,}',
]


def normalize_text(text):
    if not text:
        return text
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'[\s_\-【】（）\[\]()·/]+', '', text)
    return text


def strip_source_prefix(filename):
    cleaned = filename
    for prefix in SOURCE_PREFIXES:
        cleaned = re.sub(prefix, '', cleaned)
    return cleaned


def is_noise_file(filename):
    for pattern in NOISE_FILE_PATTERNS:
        if re.match(pattern, filename, re.IGNORECASE):
            return True
    return False


def is_valid_blogger_name(text):
    if not text:
        return False
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    if len(text_stripped) < 2:
        return False
    if len(text_stripped) > 40:
        return False
    for ignore in IGNORE_WORDS:
        if ignore.lower() == text_lower:
            return False
    if re.match(r'^[\d\W]+$', text_lower):
        return False
    if re.match(r'^\d+[VM][\d.]*[GMK]?$', text_lower):
        return False
    if re.match(r'^第[一二三四五六七八九十\d]+弹$', text_stripped):
        return False
    if re.match(r'^[\d.]+[年月号日]?$', text_lower):
        return False
    if re.match(r'^[0-9a-f]{8}-', text_lower):
        return False
    if re.match(r'^[A-Z]{2,6}-\d+$', text_stripped):
        return False
    if re.match(r'^FC2-?PPV', text_stripped, re.IGNORECASE):
        return False
    if re.match(r'^(mp4|avi|mkv|mov|wmv|flv|webm)$', text_lower):
        return False
    return True


def clean_blogger_name(name):
    name = name.strip()
    for prefix in PREFIX_CLEAN:
        if name.startswith(prefix):
            name = name[len(prefix):].strip()
    name = re.sub(r'^(?:_+|-+|\s+)', '', name)
    name = re.sub(r'(?:_+|-+|\s+)$', '', name)
    return name


def extract_blogger_from_folder(folder_name):
    folder_lower = folder_name.lower()
    for gen in GENERIC_FOLDERS:
        if gen.lower() in folder_lower:
            return None
    if re.match(r'^\d+V\d+[GM]$', folder_name):
        return None
    for pattern, ptype in FOLDER_PATTERNS:
        match = re.match(pattern, folder_name)
        if match:
            blogger = match.group(1).strip()
            blogger = clean_blogger_name(blogger)
            if is_valid_blogger_name(blogger):
                return blogger
    for pattern in FOLDER_PREFIX_PATTERNS:
        match = re.match(pattern, folder_name)
        if match:
            blogger = match.group(1).strip()
            blogger = clean_blogger_name(blogger)
            if is_valid_blogger_name(blogger):
                return blogger
    if is_valid_blogger_name(folder_name):
        return folder_name
    return None


def extract_blogger_from_file(filename):
    if filename.startswith('._'):
        return None
    if is_noise_file(filename):
        return None
    name_no_ext = os.path.splitext(filename)[0]
    cleaned = strip_source_prefix(name_no_ext)
    if not cleaned or len(cleaned) < 2:
        return None
    for pattern, ptype in FILE_BLOGGER_PATTERNS:
        match = re.match(pattern, cleaned)
        if match:
            blogger = match.group(1).strip()
            if is_valid_blogger_name(blogger):
                return blogger
    for pattern in FILE_BRACKET_PATTERNS:
        matches = re.findall(pattern, cleaned)
        for m in matches:
            m_clean = clean_blogger_name(m)
            if is_valid_blogger_name(m_clean):
                return m_clean
    for pattern in FILE_HASHTAG_PATTERNS:
        match = re.search(pattern, cleaned)
        if match:
            blogger = match.group(match.lastindex).strip()
            if is_valid_blogger_name(blogger):
                return blogger
    for pattern, ptype in FILE_CHINESE_BLOGGER_PATTERNS:
        matches = re.findall(pattern, cleaned)
        for m in matches:
            m_clean = clean_blogger_name(m)
            if is_valid_blogger_name(m_clean):
                return m_clean
    match = re.match(FILE_NANA_TAIPEI, cleaned, re.IGNORECASE)
    if match:
        return 'nana_taipei'
    match = re.match(FILE_YURI_PATTERN, cleaned, re.IGNORECASE)
    if match:
        return 'xoxo_yuri'
    match = re.match(FILE_JAPANESE_MYFANS, cleaned)
    if match:
        return match.group(1).strip()
    match = re.match(FILE_91_PATTERN, cleaned)
    if match:
        blogger = match.group(1).strip()
        if is_valid_blogger_name(blogger):
            return blogger
    return None


def get_video_stats_fast(folder_path):
    try:
        r = subprocess.run(
            f'find "{folder_path}" -type f \\( -iname "*.mp4" -o -iname "*.avi" '
            f'-o -iname "*.mov" -o -iname "*.mkv" \\) 2>/dev/null | wc -l',
            shell=True, capture_output=True, text=True, timeout=60
        )
        count = int(r.stdout.strip()) if r.stdout.strip().isdigit() else 0
        r2 = subprocess.run(f'du -sk "{folder_path}" 2>/dev/null | cut -f1',
            shell=True, capture_output=True, text=True, timeout=60)
        size_kb = int(r2.stdout.strip()) if r2.stdout.strip().isdigit() else 0
        size_gb = size_kb / (1024 * 1024)
        return count, size_gb
    except:
        return 0, 0


def build_blogger_database(base_path):
    blogger_db = {}
    try:
        items = os.listdir(base_path)
    except:
        return blogger_db
    for item in items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            blogger = extract_blogger_from_folder(item)
            if blogger:
                normalized = normalize_text(blogger)
                if normalized not in blogger_db:
                    blogger_db[normalized] = (blogger, item_path, item)
    return blogger_db


def analyze_all(base_path=BASE_PATH):
    print("=" * 80)
    print("Enhanced Video Matcher v2 - Deep Analysis Edition")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n[1/3] Building blogger database from existing folders...")
    blogger_db = build_blogger_database(base_path)
    print(f"  Found {len(blogger_db)} known bloggers")

    print("\n[2/3] Scanning files for blogger matches...")
    matches = defaultdict(list)
    video_exts = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    total_files = 0
    matched_files = 0

    try:
        result = subprocess.run(
            f'find "{base_path}" -maxdepth 3 -type f '
            f'\\( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" '
            f'-o -iname "*.mkv" \\) 2>/dev/null',
            shell=True, capture_output=True, text=True, timeout=600
        )
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    except:
        files = []

    for file_path in files:
        if not file_path:
            continue
        filename = os.path.basename(file_path)
        if filename.startswith('._'):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in video_exts:
            continue
        total_files += 1

        blogger = extract_blogger_from_file(filename)
        if blogger:
            matched_files += 1
            blogger_clean = clean_blogger_name(blogger)
            norm = normalize_text(blogger_clean)
            matched_name = blogger_clean
            for db_norm, (db_orig, db_path, db_folder) in blogger_db.items():
                if norm == db_norm or norm in db_norm or db_norm in norm:
                    matched_name = db_orig
                    break
            rel = os.path.relpath(file_path, base_path)
            matches[matched_name].append(('file', file_path, rel))

    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            blogger = extract_blogger_from_folder(item)
            if blogger:
                norm = normalize_text(blogger)
                matched_name = blogger
                for db_norm, (db_orig, db_path, db_folder) in blogger_db.items():
                    if norm == db_norm or norm in db_norm or db_norm in norm:
                        matched_name = db_orig
                        break
                matches[matched_name].append(('folder', item_path, item))

    print(f"  Scanned {total_files} files, matched {matched_files} ({matched_files*100//max(total_files,1)}%)")

    print("\n[3/3] Analysis Results:")
    print("=" * 80)

    for blogger in sorted(matches.keys()):
        items = matches[blogger]
        folders = [i for i in items if i[0] == 'folder']
        files = [i for i in items if i[0] == 'file']
        print(f"\n📁 {blogger}: {len(folders)} folders, {len(files)} files")
        for t, path, name in folders:
            print(f"   📂 {name}")
        if files:
            for t, path, name in files[:3]:
                print(f"   🎬 {name}")
            if len(files) > 3:
                print(f"   ... +{len(files) - 3} more files")

    print(f"\n{'=' * 80}")
    print(f"Summary: {len(matches)} bloggers, {total_files} files scanned, "
          f"{matched_files} matched ({matched_files * 100 // max(total_files, 1)}%)")
    print("=" * 80)

    return matches, blogger_db


def suggest_merge_plan(matches, blogger_db, base_path=BASE_PATH):
    print("\n" + "=" * 80)
    print("Suggested Merge Plan")
    print("=" * 80)

    merge_count = 0
    for blogger in sorted(matches.keys()):
        items = matches[blogger]
        folders = [i for i in items if i[0] == 'folder']

        if len(folders) > 1:
            main_folder = None
            for t, path, name in folders:
                if '【' in name:
                    main_folder = (path, name)
                    break
            if not main_folder:
                has_nvng = [f for f in folders if re.search(r'-\d+V\d+[GMK]', f[2])]
                if has_nvng:
                    main_folder = (has_nvng[0][1], has_nvng[0][2])
                else:
                    folders_sorted = sorted(folders, key=lambda x: len(x[2]), reverse=True)
                    main_folder = (folders_sorted[0][1], folders_sorted[0][2])

            merge_count += 1
            print(f"\n🔄 {blogger}:")
            print(f"   Target: {main_folder[1]}")
            print(f"   Merge from:")
            for t, path, name in folders:
                if path != main_folder[0]:
                    count, gb = get_video_stats_fast(path)
                    print(f"      - {name} ({count}V, {gb:.1f}G)")

    print(f"\nTotal merge candidates: {merge_count}")


def main():
    matches, blogger_db = analyze_all()
    suggest_merge_plan(matches, blogger_db)


if __name__ == "__main__":
    main()
