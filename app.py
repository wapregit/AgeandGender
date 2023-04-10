import cv2 
import streamlit as st 
import pandas as pd 
import numpy as np 
from PIL import Image
from datetime import datetime
from deta import Deta


TOKEN = "DFkZoYTZ_9o6Tz58g3xESNSnz7PnWY6kSEKKT8NWH"
deta = Deta(TOKEN)
db = deta.Base("imagesdata") 

def insert_imagesdata(add_name, gender, age, user_time): 
    """Returns the report on a successful creation, otherwise raises an error"""
    return db.put({"key": add_name, "gender": gender, "age": age, "datetime": user_time})


def fetch_all_imagesdata():
    """Returns a dict of all imagesdata""" 
    res = db.fetch() 
    return res.items


def get_imagesdata(imagesdata):
    """If not found, the function will return None"""
    return db.get(imagesdata)

class Detectface():
    def get_face_box(net, frame, conf_threshold=0.7):
        opencv_dnn_frame = frame.copy()
        frame_height = opencv_dnn_frame.shape[0]
        frame_width = opencv_dnn_frame.shape[1]
        blob_img = cv2.dnn.blobFromImage(opencv_dnn_frame, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob_img)
        detections = net.forward()
        b_boxes_detect = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                b_boxes_detect.append([x1, y1, x2, y2])
                cv2.rectangle(opencv_dnn_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
        return opencv_dnn_frame, b_boxes_detect

st.write("# Age and Gender prediction "+":robot_face:") 
st.write("## Upload a picture that contains a face "+":camera:")

bytes_data = None 
input_mode = st.radio("# Input mode", ["Camera", "File upload"]) 

if input_mode == "Camera":
    img_file_buffer = st.camera_input("Take a picture")
    photo = img_file_buffer 
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
else:
    uploaded_file = st.file_uploader("อัพโหลดรูปภาพ", type=['png', 'jpg'])
    photo = uploaded_file
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
if bytes_data is None:
    st.stop()
    
if photo:
    image = Image.open(photo) #เปิดใช้ photo
    cap = np.array(image)
    cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY))
    cap=cv2.imread('temp.jpg')

    face_txt_path="opencv_face_detector.pbtxt"
    face_model_path="opencv_face_detector_uint8.pb"

    age_txt_path="age_deploy.prototxt" 
    age_model_path="age_net.caffemodel"

    gender_txt_path="gender_deploy.prototxt" 
    gender_model_path="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    age_classes=['Age: ~1-2', 'Age: ~3-5', 'Age: ~6-14', 'Age: ~16-22', 'Age: ~23-30', 'Age: ~31-40', 'Age: ~41-50', 'Age: Greater than 50']
    gender_classes = ['Male', 'Female']

    age_net = cv2.dnn.readNet(age_model_path, age_txt_path) 
    gender_net = cv2.dnn.readNet(gender_model_path, gender_txt_path) 
    face_net = cv2.dnn.readNet(face_model_path, face_txt_path)

    padding = 20 
    frameFace, b_boxes = Detectface.get_face_box(face_net, cap)
    
    if not b_boxes:
        st.write("ไม่พบใบหน้าที่กำลังตรวจสอบ")

    for bbox in b_boxes:
        face = cap[max(0, bbox[1] - padding): min(bbox[3] + padding, cap.shape[0] - 1), max(0, bbox[0] - padding): min(bbox[2] + padding, cap.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_pred_list = gender_net.forward()
        gender = gender_classes[gender_pred_list[0].argmax()]
        st.write(
            f"Gender : {gender}, confidence = {gender_pred_list[0].max() * 100}%")

        age_net.setInput(blob)
        age_pred_list = age_net.forward()
        age = age_classes[age_pred_list[0].argmax()]
        st.write(f"Age : {age}, confidence = {age_pred_list[0].max() * 100}%")

        label = "{},{}".format(gender, age)
        cv2.putText(
            frameFace,
            label,
            (bbox[0],
            bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            cv2.LINE_AA)

        st.image(frameFace)
        st.sidebar.header("บันทึกข้อมูลลงในฐานข้อมูล")

        df = pd.read_csv("data/test.csv")
        st.sidebar.table(df)
        options_form = st.sidebar.form("options_form")
        add_name = options_form.text_input("โปรดใส่ชื่อของคุณ")
        now = datetime.now()
        user_time = now.strftime("%H:%M:%S")
        add_data = options_form.form_submit_button("บันทึกข้อมูล")

        if add_data:
            new_data = {"name": add_name , "gender": gender ,"age": age ,"time": user_time}
            df = df.append(new_data, ignore_index = True)
            df.to_csv("data/test.csv" , index = False)
            insert_imagesdata(add_name,gender,age,user_time)
            st.sidebar.header("บันทึกข้อมูลสำเร็จ")
            
        im_pil = Image.fromarray(frameFace)
        im_pil.save('Result.jpeg')
        with open("Result.jpeg", "rb") as file:
                btn = st.download_button( 
                        label="ดาวน์โหลดรูปภาพ",
                        data=file,
                        file_name="image_test.jpeg",
                    )
                if btn :
                    st.write('ดาวน์โหลดสำเร็จ')
                else :
                    st.write("คุณยังไม่ได้ทำการดาวน์โหลดรูปภาพ")
                    
        if add_data is not None :
            st.write('กราฟเพศและอายุของผู้ใช้งานทั้งหมด')
            df2 = df.filter(regex= '(gender|age)', axis=1)
            st.line_chart(df2)