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
import numpy as np
import h5py
from pathlib import Path

youtube_vid_dir = "../data/videos/"
resized_vid_dir = "../data/resized_vids/"
hdf5_dir = Path("../data/hdf5/")


def check_args():
    if len(sys.argv) != 2:
        print("Too Few Args!")
        print("USAGE: video_downloader.py videos.txt")
        sys.exit(1)
    return 1

"""
Gets videos line by line from video file. Downloads each using YouTube Streamer

PARAMS:
    videos_file: text file with urls to youtube videos

RETURN: 
    vid_dictionary: dictionary mapping video title -> total frames
"""
def download_videos(videos_file):
    vid_dictionary  = {}
    
    try:
        os.mkdir(youtube_vid_dir)
    except:
        pass
    
    with open(videos_file) as vids:
        
        for vid in vids:
            print("Trying to download: "+vid)
            try:
                yt = YouTube(vid)
                frames = yt.length * yt.streams.first().fps
                title = yt.streams.first().download(youtube_vid_dir)
                titles = title.split("videos/")
                vid_dictionary[titles[1]] = frames
                print("Success!")
            except: 
                print("Couldn't download " + vid)
    return vid_dictionary


"""
Loop through all downloaded youtube vids and split them into smaller (30,30,30) vids
Store everything in video_arr : numpy array for storing later

PARAMS:
    vid_dictionary: dictionary mapping video title -> frame count
    
RETURN:
    video_arr: numpy array of size (total_vids,30,30,30,1)
"""

def split_all_vids(vid_dictionary):
    total_vids = calc_total_mini_vids(vid_dictionary)
    
    video_arr = np.zeros((total_vids,30,30,30))
    curr_index = 0
    
    try:
        os.mkdir(resized_vid_dir)
    except:
        pass
    youtube_vids = get_files(youtube_vid_dir)
    
    for youtube_vid in youtube_vids: 
        print("Splitting : " +str(youtube_vid))
        
        num_mini_vids = vid_dictionary[youtube_vid] // 30
        
        
        t_arr, num_videos = parse_video(youtube_vid,num_mini_vids)
        final_index = curr_index + num_videos
        
        video_arr[curr_index:final_index] = t_arr[:num_videos]
        curr_index = final_index

    video_arr = np.delete(video_arr,np.s_[curr_index-1:],0) ## ADD ANOTHER 10 VIDEO BUFFER
    print(np.shape(video_arr))
    return video_arr

"""
Function to read in full length videos from download_videos and save them 

PARAMS:
    video: filename for video. 
        NOTE: file expects videos to be located in ../data/videos directory
    
    num_vids: integer specifying the total number of mini videos expected to be created
        NOTE: videos must be size (30,30,30)
    
RETURN: 
    t_arr: numpy array of size (num_vids,30,30,30)
"""
def parse_video(video,num_vids):
    t_arr = np.zeros((num_vids, 30,30,30))
    mini_vid = np.zeros((30,30,30))
    vid_num = 0
    
    dim = (30,30)
    video = "../data/videos/" + video

    frames = 0
    cap = cv2.VideoCapture(video)    
    success, frame = cap.read()
    while success:
        frames+=1
        image = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mini_vid[frames-1] = gray
        if frames == 30:
            t_arr[vid_num] = mini_vid
            vid_num +=1
            frames = 0
        success,frame = cap.read()
            
    return t_arr, vid_num - 10 # return 10 videos less as safety buffer 

""" 
Stores an array of images to HDF5.

PARAMS:
    images: images array, (N, 30, 30, 30, 1) to be stored
"""
def store_many_hdf5(images):

    num_images = len(images)
    try:
        os.mkdir("../data/hdf5")
    except: 
        pass

    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_vids.h5", "w")

    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )    
    
    file.close()
  
    
"""
Calculate the total number of mini videos after splitting down.
Used to create numpy array later 

PARAMS:
    vid_dictionary: dictionary mapping video titles -> total frames
    
RETURN: 
    total: integer of total mini videos that can be made
"""    
def calc_total_mini_vids(vid_dictionary):
    frames = vid_dictionary.values()
    total = 0
    for frame in frames:
        total += frame // 30
    print("total vids = " + str(total))
    return total 

"""
Get all files in directory

PARAMS:
    directory: path to directory
    
RETURN: 
    list of files in directory
"""    
def get_files(directory):
    return os.listdir(directory)


          
            
if __name__ == "__main__":
       if (check_args()):
            vid_dictionary = download_videos(sys.argv[1])
            vid_arr = split_all_vids(vid_dictionary)
            check = vid_arr[10][0]
            store_many_hdf5(vid_arr)
            
        
            