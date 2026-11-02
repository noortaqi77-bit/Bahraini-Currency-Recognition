import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Bahraini Currency Classifier")

# load the model :
model= YOLO("best (1).pt")

st.title("Bahraini Currency Classifier")

# camera and upload images:
source=st.radio("",["Upload Image", "Camera"],horizontal=True)


image=None
if source == "Camera":
    image_money= st.camera_input("Take picture of the money")

    if image_money is not None:
        # change it to RGb
        image=Image.open(image_money).convert("RGB")
else:
    # upload the money images :
    uploaded=st.file_uploader("Choose an image",type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image=Image.open(uploaded).convert("RGB")



## the results :
if image is not None:
    st.image(image,width=400)

    if st.button("Detect Currency",type="primary"):
        results=model.predict(image,verbose=False)
        result=results[0]

        if result.boxes is None or len(result.boxes) == 0:
            st.warning("No currency detected..")

        else:
            st.subheader("Detection Results")

            annotated=result.plot()
            # make the image with its real color
            st.image(annotated[:, :, ::-1],caption="Detected Currency",width=600)

            # print the results 
            for i,box in enumerate(result.boxes):
                class_id= int(box.cls.item())
                confidence= float(box.conf.item())
                class_money=result.names[class_id]

                st.success(f"Prediction {i + 1}: {class_money}")
                st.write(f"Confidence: {confidence:.2%}")
