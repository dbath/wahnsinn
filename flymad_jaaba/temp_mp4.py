import subprocess as sp
import os
import glob
import argparse
import cv2

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--videodir', type=str, required=True,
                        help='directory of fmf and JAABA csv files')  
args = parser.parse_args()

VIDEO_DIR = args.videodir



    
    
    

    
    

for x in glob.glob(VIDEO_DIR + '*costim*' + '*zoom*'):
    
    if os.path.exists(x + 'flymad_annotated.mp4'):  ###should use complete mp4 file here
        print 'skipping already finished file:', x
        continue
    
    
    command = [ 'ffmpeg',
                '-f', 'image2',
                '-r', '15',
                '-i', '-',
                '-vcodec', 'mpeg4',
                '-b', '8000k',
                '-y',
                (x + 'flymad_annotated.mp4')
                ]
                
    pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
    
    for frame in glob.glob((x + '/temp_png/*.png')):
        frame_array = cv2.imread(frame)
        pipe.stdin.write(frame_array)
        
