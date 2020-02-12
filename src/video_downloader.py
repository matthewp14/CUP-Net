#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Author: Matthew Parker
ENGS 87/88 
File to download and format videos to be 30x30x30 (x,y,t)
Date: 2/12/20

Takes a list of youtube videos and downloads them.
Loads the videos again and reformats them

USAGE: video_downloader.py videos.txt 

"""

import os
import sys
from pytube import YouTube

vid_dir = "../data/videos/"


def check_args():
    if len(sys.argv) != 2:
        print("Too Few Args!")
        print("USAGE: video_downloader.py videos.txt")
        sys.exit(1)
    return 1

"""
Gets videos line by line from video file. Downloads each using YouTube Streamer
"""
def download_videos(videos_file):
    
    try:
        os.mkdir(vid_dir)
    except:
        pass
    
    with open(videos_file) as vids:
        
        for vid in vids:
            print("Trying to download: "+vid)
            try:
                YouTube(vid).streams.first().download(vid_dir)
                print("Success!")
            except: 
                print("Couldn't download " + vid)
            
            
if __name__ == "__main__":
        if (check_args()):
            download_videos(sys.argv[1])