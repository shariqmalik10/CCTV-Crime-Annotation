# CCTV-Crime-Annotation
## This is a repo containing code for CCTV Crime Annotation Model

This is a app made using Streamlit which has 3 models in the backend mainly scene, object 
and action detection models. 

### Setup instructions 
1. Make a new conda environment :
```console
foo@mac:~$ conda create --name myenv
```

2. Install the setup.py file after navigating to the Scene_Detection folder. Use python (or python3) as per 
what you are using. 
```console
foo@mac:~$ cd image-super-resolution
foo@mac:~$ python setup.py install
```

3. Next install the requirements.txt file 
```console
foo@mac:~$ pip3 install -r requirement.txt
```

x. Final Step: In the terminal run the command as follows:
```console
foo@mac:~$ python3 -m streamlit run gui.py
```
