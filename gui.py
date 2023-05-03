from Scene_Detection.run_placesCNN_unified import *
import streamlit as st
import tempfile

#function to predict the scene
def scene_prediction(tfile, network_name): 
    result = scene_predict(tfile.name, network_name) #type:ignore
    st.write("This is an " + result["environment"] + " environment")
    st.write("This is a " + result["attribute_1"] + " " + result["attribute_3"]+ " structure")
    st.write("The scene category is " + result["scene_category"])

def main():
    #main function
    st.markdown("<h1 style='text-align: center;'>CCTV Crime Footage Annotation</h1>", unsafe_allow_html=True)

    #upload file
    uploaded_file = st.file_uploader("Choose a video file (.mp4 format)")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read()) #type: ignore

    res_network = st.selectbox(
        'Select a super resolution network to use for scene prediction: ',
        ('psnr-small', 'psnr-large', 'rrdn', 'noise-cancel', 'none')
    )

    assigned_1 = st.button("Predict") #type: ignore
    
    if assigned_1:
        with st.expander('Scene Prediction', expanded=False):
            scene_prediction(tfile, res_network)

if __name__ == "__main__":
    main()
# st.write(predict) # type: ignore

