# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# with open('data/names.pkl', 'rb') as w:
#     LABELS=pickle.load(w)
# with open('data/faces_data.pkl', 'rb') as f:
#     FACES=pickle.load(f)

# print('Shape of Faces matrix --> ', FACES.shape)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# imgBackground=cv2.imread("background.png")

# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
#         output=knn.predict(resized_img)
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#         cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
#         attendance=[str(output[0]), str(timestamp)]
#     imgBackground[162:162 + 480, 55:55 + 640] = frame
#     cv2.imshow("Frame",imgBackground)
#     k=cv2.waitKey(1)
#     if k==ord('o'):
#         speak("Attendance Taken..")
#         time.sleep(5)
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(attendance)
#             csvfile.close()
#         else:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
#             csvfile.close()
#     if k==ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()

from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(message):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(message)

# Load face data and labels
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Match lengths
if len(LABELS) != len(FACES):
    min_len = min(len(LABELS), len(FACES))
    LABELS = LABELS[:min_len]
    FACES = FACES[:min_len]

print('Shape of Faces matrix -->', FACES.shape)
print('Number of Labels -->', len(LABELS))

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load UI
imgBackground = cv2.imread("background.png")
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance = [str(output[0]), timestamp]

        exist = os.path.isfile(f"Attendance/Attendance_{date}.csv")

        # Draw face and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (0, 255, 0), -1)
        cv2.putText(frame, str(output[0]), (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Attendance action on key press
        if cv2.waitKey(1) & 0xFF == ord('o'):
            speak("Attendance Taken")
            os.makedirs("Attendance", exist_ok=True)
            with open(f"Attendance/Attendance_{date}.csv", 'a', newline='') as f:
                writer = csv.writer(f)
                if not exist:
                    writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            time.sleep(3)

    # Show frame on background
    if imgBackground is not None and frame is not None:
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow("Frame", imgBackground)
    else:
        cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


