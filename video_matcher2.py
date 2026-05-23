#!/usr/bin/env python3
import os
import re

# Base path
base_path = "/Volumes/personal_folder-1/movie_a/中文"

# Folders to check (with their actual folder names on disk)
folders_to_check = [
    "其他",
    "合集", 
    "男友泄密⏩十月多位美女泄密合集-59V2G",
    "约炮大神Jxzeroc长腿美女合集12v8.56g-19V13G",
    "附大量聊天记录",
    "重磅核弹",
    "包-2-28V12G",
    "23-74V3G",
    "紫蛋合集.7z-34V16G",
    "5号系列视频-17V8G"
]

# Known blogger folders
blogger_names = [
    "Aheyanlz", "CandFans_烈", "Couplelove", "Enthus1asmm", "Green杰克", "ICICIS", "JOJO杰克", 
    "LUMA", "MyFans", "ROSEHJ", "ROSEWU", "RabbyjayCouple", "RinYu林语", "S-阿拉蕾秘语", 
    "sexy_yuki", "fKabuto", "hRQ百态", "henry_sera", "imladylinn小林女士", "jaacckk999235v13.9g", 
    "lananlanan", "luckydog7", "muchi_tina", "mympet", "nana_taipei", "newyearst6", 
    "onlyfans_xoxo_yuri", "pikpak1018", "pittyswg", "pupuv1", "sweetkk", "天野リリス", 
    "我的枪好长", "晨汐", "深海杀人鲸_小张历险记", "狗爹和小桃", "饭饭吖", "峰不二子", 
    "隔壁王某某", "你的娇妹妹", "跳跳羊", "噗噗通通", "小杏仁", "郑欣雯", "好大的爆米花", 
    "海绵宝宝", "武汉情侣", "罗芙S绿帽奴", "绿奴小猫", "噗噗", "北门玩家"
]

# Create lowercase versions for case-insensitive matching
blogger_names_lower = [name.lower() for name in blogger_names]

def check_folder(folder_name, max_files=None, check_subdirs=False):
    """Check a folder for video files and try to match blogger names"""
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Checking folder: {folder_name}")
    print(f"{'='*60}")
    
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    matches_found = []
    
    if check_subdirs:
        # For 合集 folder: check root and first level subdirectories
        files_checked = 0
        for root, dirs, files in os.walk(folder_path):
            # Only check root and first level subdirectories
            if root == folder_path or os.path.dirname(root) == folder_path:
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        if max_files and files_checked >= max_files:
                            break
                        
                        # Check if filename contains any blogger name
                        file_lower = file.lower()
                        for i, blogger_lower in enumerate(blogger_names_lower):
                            if blogger_lower in file_lower:
                                matches_found.append((file, blogger_names[i]))
                                break
                        
                        files_checked += 1
                
                if max_files and files_checked >= max_files:
                    break
    else:
        # For other folders: check all files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    # Check if filename contains any blogger name
                    file_lower = file.lower()
                    for i, blogger_lower in enumerate(blogger_names_lower):
                        if blogger_lower in file_lower:
                            matches_found.append((file, blogger_names[i]))
                            break
    
    # Print results
    if matches_found:
        print(f"\nFound {len(matches_found)} potential matches:")
        for file_name, blogger_name in matches_found:
            print(f"  {file_name} -> {blogger_name}")
    else:
        print("\nNo matches found in this folder.")
    
    # Also list some sample files for inspection
    print(f"\nSample files (first 10):")
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                print(f"  {file}")
                count += 1
                if count >= 10:
                    break
        if count >= 10:
            break

# Check each folder
for i, folder in enumerate(folders_to_check):
    if i == 1:  # 合集 folder
        check_folder(folder, max_files=100, check_subdirs=True)
    else:
        check_folder(folder)

print("\n" + "="*60)
print("Analysis complete. This was a DRY-RUN - no files were moved.")
print("="*60)