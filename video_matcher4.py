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

def analyze_all_folders():
    """Analyze all folders to see if there are any other matches"""
    all_matches = {}
    
    # List all directories in the base path
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # Check if this is a blogger folder
            item_lower = item.lower()
            for i, blogger_lower in enumerate(blogger_names_lower):
                if blogger_lower in item_lower:
                    if blogger_names[i] not in all_matches:
                        all_matches[blogger_names[i]] = []
                    all_matches[blogger_names[i]].append(f"Folder exists: {item}")
                    break
            
            # Check for video files in this folder
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in video_extensions):
                        file_lower = file.lower()
                        for i, blogger_lower in enumerate(blogger_names_lower):
                            if blogger_lower in file_lower:
                                if blogger_names[i] not in all_matches:
                                    all_matches[blogger_names[i]] = []
                                rel_path = os.path.relpath(root, base_path)
                                all_matches[blogger_names[i]].append(f"File: {file} (in {rel_path})")
                                break
    
    # Print summary
    print(f"\n{'='*60}")
    print("Summary of all matches found:")
    print(f"{'='*60}")
    
    for blogger_name, matches in sorted(all_matches.items()):
        print(f"\n{blogger_name}:")
        for match in matches:
            print(f"  - {match}")
    
    print(f"\nTotal bloggers with matches: {len(all_matches)}")
    print(f"Total bloggers in list: {len(blogger_names)}")

# Run the analysis
analyze_all_folders()

print("\n" + "="*60)
print("Complete analysis finished.")
print("="*60)