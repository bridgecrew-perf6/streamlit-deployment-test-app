
# This libraries are for testing purposes
import streamlit as st
import mediapipe as mp
from PIL import Image
import numpy as np
import tempfile
import time
import cv2


# Variables created for ease of typing
MPDrawing = mp.solutions.drawing_utils
MPFaceMesh = mp.solutions.face_mesh


time.localtime()
TmpFile = tempfile


@st.cache()
def FrameResize(Frame, FrameWidth=None, FrameHeight=None, InterpolationMtd=cv2.INTER_AREA):
    FrameDimensions = None
    height, width, _ = Frame.shape

    # We simply return the Image or Video Frame if both the Width and Height are None
    if FrameWidth is None and FrameHeight is None:
        return Frame

    # This statement executes if only the FrameWidth is None
    if FrameWidth is None:
        result = FrameWidth / float(width)
        FrameDimensions = (int(width * result), FrameHeight)

    else:
        result = FrameWidth / float(width)
        FrameDimensions = (FrameWidth, int(height * result))

    # Here, we resize the frame using the calculated values above using opencv
    ResizedFrame = cv2.resize(Frame, FrameDimensions,
                              interpolation=InterpolationMtd)

    # Return the new Resized Frame
    return ResizedFrame


st.set_page_config(layout="wide")

st.title("Streamlit Test Deploy Repository Page")

st.sidebar.title("Test Sidebar")

st.sidebar.markdown('---')

# This allows Users to import an Image file from their local Machine
UploadImageFile = st.sidebar.file_uploader(
    'Upload an Image', type=["png", "jpg", "jpeg"])

# This statement executes when the file upload buffer is not empty
if UploadImageFile is not None:
    ImageFile = np.array(Image.open(UploadImageFile))

    # These next 2 lines display the original image imported by the User on the Sidebar
    st.sidebar.text('Original Image Uploaded')
    st.sidebar.image(ImageFile)

# If the User Upload file is empty, then we use a stock image
else:
    pass

    st.sidebar.text('Demo Image Provided')
    # st.sidebar.image(ImageFile)

st.sidebar.markdown('---')


NumFaces = st.sidebar.number_input(
    'Select the number of faces you want to detect', value=2, min_value=1)

st.sidebar.markdown('---')

# Creates a Slider on the Sidebar for the User to set the Detection Confidence of the Model
DetectionConfidence = st.sidebar.slider(
    'Minimum Detection Confidence', min_value=0.0, max_value=1.0, value=0.5)

MeshThickness = st.sidebar.slider(
    'Mesh Drawing Thickness', min_value=1, max_value=10, value=2)

MeshCircleRadius = st.sidebar.slider(
    'Mesh Draawing Circle Radius', min_value=1, max_value=10, value=1)

DrawingSpec = MPDrawing.DrawingSpec(
    thickness=MeshThickness, circle_radius=MeshCircleRadius)
