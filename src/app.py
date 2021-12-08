from flask import Flask, render_template, url_for, redirect, request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pygame import mixer
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/mask_frames')
def mask_frames():
    facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel') #얼굴 인지모델(face detection) 미리 학습해둔 파일을 받아옴
    model = load_model('models/mask_detector.model') #마스크 모델은 keras모델 -> 설명 사이트 https://hazel-developer.tistory.com/97

    cap = cv2.VideoCapture(0) # videocapture 객체 생성 (0 = 노트북 내장 웹캠)
    ret, img = cap.read() # 프레임별로 이미지 캡쳐 ret = 프레임을 정상적으로 읽었는가 확인 

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # 비디오 캡쳐 저장 객체
    out = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (img.shape[1], img.shape[0]))
    #cv2.VideoWriter(파일이름, fourcc, fps초당프레임수, frameSize(width, height))
    mixer.init()
    sound = mixer.Sound('alarm.mp3')

    while cap.isOpened(): # cap이 초기화 되었나 확인
        ret, img = cap.read()
        if not ret:
            break

        h, w = img.shape[:2] #이미지의 높이와 너비 저장

        blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(300, 300), mean=(104., 177., 123.)) #네트워크 입력 블롭(blob)으로 이미지 만들기
        #(입력영상, 입력영상-픽셀에 곱할 값으로 기본값은 1, 출력영상크기, 입력 영상 각 채널에서 뺄 평균 값)
        facenet.setInput(blob) #모델에 들어가는 input
        dets = facenet.forward() #결과 추론

        result_img = img.copy() #결과값 저장

        for i in range(dets.shape[2]): # 사진속 얼굴 개수가 여러 개 있을 수 있으니 반복문 사용
            confidence = dets[0, 0, i, 2] #결과값이 얼마나 자신있느냐
            if confidence < 0.5: #값을 0.5로 정함
                continue # 0.5 미만인 값들은 모두 넘김

            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h) # x와 y의 바운딩 박스 값 구해줌
        
            face = img[y1:y2, x1:x2] # 얼굴 사진만 자르기
            # 전처리 구간
            face_input = cv2.resize(face, dsize=(224, 224)) #이미지 크기 변경
            face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB) #컬러 시스템을 BGR에서 RGB로 변환
            face_input = preprocess_input(face_input) #전처리 저장값 shape(224,224,3)
            face_input = np.expand_dims(face_input, axis=0) # 모델에 넣을때 (1,224,224,3)이므로 차원 추가
        
            mask, nomask = model.predict(face_input).squeeze() #predicr method를 사용해 아웃풋 출력 (마스크 있음, 마스크 없음)
        # 마스크 확률 출력
            if mask > (nomask*0.9):
                color = (0, 255, 0)
                label = 'Mask %d%%' % (mask * 100)
            else:
                color = (0, 0, 255)
                label = 'No Mask %d%%' % (nomask * 100)
                sound.play()
                time.sleep(1)
                cv2.putText(result_img, text='Wear Your Mask', org=(x1, y1 - 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)

            cv2.rectangle(result_img, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=color, lineType=cv2.LINE_AA) 
            #사각형 표시 (그림 그릴 이미지, 시작좌표, 종료좌표, 선두께)
            cv2.putText(result_img, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2, lineType=cv2.LINE_AA)
            #문자 표시 (그릴 이미지, 출력문자, 출력문자 시작위치 좌표, 출력 문자 시작위치 좌표, 폰트, 폰트 크기, 색상, 두께, 선종류)
        out.write(result_img) 
        cv2.imshow('result', result_img) #이미지를 모니터에 보여줌
        if cv2.waitKey(1) == ord('q'): #아스키 값으로 어떤 키를 눌렀는지 보여주는 값, q값을 누르면 영상 종료
            break

    out.release() # out 객체 해제
    cap.release() # cap 객체 해제

    return mask_frames


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')