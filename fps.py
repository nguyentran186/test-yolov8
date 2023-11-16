import os
from glob import glob

os.chdir('D:/NGUYEN/VD/.Code/ultralytics/ultralytics/assets/videos')
clip = glob('*mp4')[0]
clip_name = clip.split(".")[0] + '1'

command = "ffmpeg -i {0} -filter: fps=1 {1}.mov".format(clip, clip_name)
os.system(command)