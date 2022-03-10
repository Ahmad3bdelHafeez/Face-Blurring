# Face-Blurring
Privacy Protection is a real-world application by detection the faces in the input images and blurred these faces.
- - - -
![Face-Blurring](https://github.com/Ahmad3bdelHafeez/Face-Blurring/blob/main/output%206.PNG "Face-Blurring")
# High level pseudo-code for input image called it ‘im’:
1.	Get faces in ‘im’ by used ‘CaffeeModel’ Pre-trained model for face detection.
2.	Use OpenCV for applying smoothing filter (blur) on ‘im’ called it ‘blur-im’.
3.	For each face in ‘im’:
    1.	Calculate the boundary box of the face.
    2.	Get area of the boundary box from ‘blur-im’.
    3.	Replace the area into ‘im’. (To blur the face only)
# Demo:
Run the Colab Notebook and see results.
# Code:
Download it from this repository.
# How to run:
1.	Install the dependencies packages from ‘requirements.txt’ file.
2.	Write your test image path in ‘main.py’ file.
3.	Run ‘main.py’ file and see the results.
# References:
1.	https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ 
2.	https://colab.research.google.com/github/goodboychan/chans_jupyter/blob/main/_notebooks/2020-08-02-03-Advanced-Operations-Detecting-Faces-and-Features.ipynb#scrollTo=MApLLAYcGX31
