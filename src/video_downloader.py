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
import cv2

i = 0 # variable for the video file save

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


"""
Function to read in full length videos from download_videos and save them 
"""
def parse_video(video):
    frames = 1
    video_num = 1
    cap = cv2.VideoCapture(video)
    prefix = "vid_" + str(i) + "_"
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(prefix + str(video_num) + ".avi" ,fourcc, 30, (30,30))
    
    success, frame = cap.read()
    
    while success:
        if frames == 30:
            out = cv2.VideoWriter(prefix+str(video_num)+".avi", fourcc, 30, (30,30))
            frames = 1
            video_num+=1
        resized = cv2.resize(frame, (30,30), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        out.write(gray)
        frames +=1
            
            
if __name__ == "__main__":
        if (check_args()):
            download_videos(sys.argv[1])