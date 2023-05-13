import subprocess
from Scene_Detection.run_placesCNN_unified import *
import streamlit as st
import tempfile
import os
from yolov7.yolov7_detect_txt import *
from mmaction2.analyse_vid import *
from mmaction2.analyse_vid2 import *

import sys
import re
import time

script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join( script_dir, './')
sys.path.append(mymodule_dir)
working_dir = "C:/Users/Tee/Desktop/FYP/GitFYP/Crime_Annotation"

#function to predict the scene
def scene_prediction(tfile, network_name): 
    with st.spinner("Analysis in progress ... "):
        time.sleep(5)
        result = scene_predict(tfile.name, network_name) #type:ignore
        st.write("This is an " + result["environment"] + " environment")
        st.write("This is a " + result["attribute_1"] + ", " + result["attribute_3"]+ " structure")
        st.write("The top 5 predicted categories for scene are: ")
        for i in range(len(result["scene_category"])):
            st.write(str(i+1) + ". " + result["scene_category"][i])

def object_prediction(vid_file):
    with st.spinner("Analysis in progress ... "):
        time.sleep(5)
        rpath, result = detect_final(vid_file)
        rpath.replace("\\", "/")
        # st_video = open(rpath, 'rb')
        # video_bytes = st_video.read()
        st.write("Objects Detected")
        for key,value in result.items():
            if key == "person":
                st.write("There are " + str(value) + " people seen in the CCTV footage")
            elif value == 1 : 
                st.write("There is " + str(value) + " " + key + " in the video")
            elif value > 1 : 
                st.write("There are " + str(value) + " " + key + "s in the video")
        
        st.write("Object Detection Video has been created at the location: " + rpath)


def action_detection(vid_file):
    with st.spinner("Analysis in progress ... "):
        time.sleep(5)
        current_dir = os.getcwd()
        #st.write("Current directory is " + current_dir)
        if os.path.abspath(current_dir) != os.path.abspath('C:/Users/Tee/Desktop/FYP/GitFYP/Crime_Annotation/mmaction2'):
            os.chdir('C:/Users/Tee/Desktop/FYP/GitFYP/Crime_Annotation/mmaction2')

        changed_dir = os.getcwd()
        #st.write("changed directory is " + changed_dir)
        # need to make sure that the env is correctly set
        the_cmd = 'conda run -n lastfyp python analyse_vid.py {}'.format(vid_file)
        #the_cmd = "python analyse_vid.py '{}'".format(vid_file)
        output = subprocess.run(the_cmd, shell=True, capture_output=True)
        output_str = output.stdout.decode('utf-8').replace(the_cmd, '')
        

        # Convert string representation of list into list
        output_lst = eval(output_str)
        #st.write("output is :" + str(output))
        #st.write("output_str is :" + str(output_str))
        for i in range(len(output_lst)):
            count = i + 1
            st.write("The top {} action detected in the video uploaded is {}".format(count, output_lst[i]))

        # Changing back the directory to the original
        os.chdir(working_dir)
        # last_dir = os.getcwd()
        # st.write("last current directory is " + last_dir)

def main():
    #main function
    st.markdown("<h1 style='text-align: center;'>CCTV Crime Footage Annotation</h1>", unsafe_allow_html=True)
    
    current_dir = os.getcwd()
    if os.path.abspath(current_dir) != os.path.abspath(working_dir):
        os.chdir(working_dir)
        
    #upload file
    uploaded_file = st.file_uploader("Choose a video file (.mp4 format)", type=['mp4', 'mpeg', 'mov'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read()) #type: ignore
        # st.write("tfile is :" + str(tfile))
        vidpath = working_dir + "/" + os.path.join("data/uploads/", uploaded_file.name)
        #st.write("imgpath is :" + str(imgpath))

        with open(vidpath, mode='wb') as f:
            f.write(uploaded_file.getbuffer())  # save video to disk

        st_video = open(vidpath, 'rb')
        video_bytes = st_video.read()
        st.write("Uploaded Video")
        st.video(video_bytes)

    res_network = st.selectbox(
        'Select a super resolution network to use for scene prediction: ',
        ('psnr-small', 'psnr-large', 'rrdn', 'noise-cancel', 'none')
    )

    
    model = st.selectbox(
        'Select what type of prediction you want to make:' ,
        ('scene', 'object', 'action', 'all') 
    )
    assigned_1 = st.button("Predict") #type: ignore
    
    progress_text = "Analysis in progress ..."
    # the code that calls the models
    if assigned_1:
        if uploaded_file is not None:
            if model=="scene":
                st.markdown("<h2 style='text-align: center;'>Scene Prediction</h1>", unsafe_allow_html=True)
                # start_time = time.time()
                # initial_time = start_time
                scene_prediction(tfile, res_network)
                # end_time = time.time()
                # time_took = end_time - start_time
                # with st.spinner("Analysis in progress .... "):
                # time.sleep(5)
                # st.write(result)         
            elif model=="object":
                st.markdown("<h2 style='text-align: center;'>Object Prediction</h1>", unsafe_allow_html=True)
                object_prediction(vidpath)
            elif model=="action":
                st.markdown("<h2 style='text-align: center;'>Action Prediction</h1>", unsafe_allow_html=True)
                action_detection(vidpath)
            else:
                with st.expander('Scene Prediction', expanded=False):
                    # start_time = time.time()
                    # initial_time = start_time
                    scene_prediction(tfile, res_network)
                    # end_time = time.time()
                    # time_took = end_time - start_time
                    # st.write("Time it takes for scene model to detect the video: {:.2f} seconds".format(time_took))
                with st.expander('Object Prediction', expanded=False):
                    # start_time = time.time()
                    object_prediction(vidpath)
                    # end_time = time.time()
                    # time_took = end_time - start_time
                    # st.write("Time it takes for object model to detect the video: {:.2f} seconds".format(time_took))
                with st.expander('Action Detection', expanded=False):
                    # start_time = time.time()
                    action_detection(vidpath)
                    # end_time = time.time()
                    # time_took = end_time - start_time
                    # total_time_took = end_time - initial_time
                    # st.write("Time it takes for action model to detect the video: {:.2f} seconds".format(time_took))
                    # st.write ("Time it takes for all 3 models to detect the video: {:.2f} seconds".format(total_time_took))            
    
        else:
            st.error("Upload file for prediction", icon="🚨")

if __name__ == "__main__":
    main()


