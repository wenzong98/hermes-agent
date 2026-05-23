#!/usr/bin/env python3
import os
import re

# Base path
base_path = "/Volumes/personal_folder-1/movie_a/中文"

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

def check_folder_more_thoroughly(folder_name):
    """Check a folder more thoroughly for video files and try to match blogger names"""
    folder_path = os.path.join(base_path, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"Checking folder more thoroughly: {folder_name}")
    print(f"{'='*60}")
    
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    matches_found = []
    all_files = []
    
    # Collect all video files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                all_files.append((root, file))
    
    print(f"Total video files found: {len(all_files)}")
    
    # Check each file for blogger name matches
    for root, file in all_files:
        file_lower = file.lower()
        for i, blogger_lower in enumerate(blogger_names_lower):
            if blogger_lower in file_lower:
                matches_found.append((file, blogger_names[i], root))
                break
    
    # Print results
    if matches_found:
        print(f"\nFound {len(matches_found)} potential matches:")
        for file_name, blogger_name, root_path in matches_found:
            rel_path = os.path.relpath(root_path, folder_path)
            if rel_path == ".":
                print(f"  {file_name} -> {blogger_name} (in root)")
            else:
                print(f"  {file_name} -> {blogger_name} (in subfolder: {rel_path})")
    else:
        print("\nNo matches found in this folder.")

# Check the "合集" folder more thoroughly
check_folder_more_thoroughly("合集")

print("\n" + "="*60)
print("Additional analysis complete.")
print("="*60)