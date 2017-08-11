import numpy as np
from PIL import ImageGrab
import cv2
import time, os
from getKeys import key_check
from grabScreen import grab_screen

def keys_to_output(keys):
    #[A,W,D,S]
    output = [0,0,0,0]
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    elif 'W' in keys:
        output[1] = 1
    else:
        output[3] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    last_time = time.time()

    while True:
        screen = grab_screen(region=(0,40,800,640))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(80,60))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()

        #cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name,training_data)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        #    break

main()
