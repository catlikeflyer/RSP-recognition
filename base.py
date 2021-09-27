import cv2
import mediapipe as mp
import time

# Setup first webcam as capture input
cap = cv2.VideoCapture(0)

# Reads hand landmark model from mediapipe
mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # To visually track landmarks

# Time for framerates
pTime = 0 # Prev time
cTime = 0 # Curr time

while True:
    success, img = cap.read() # Reads image
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Color set to RGB
    results = hands.process(imgRGB) # Results to extract info
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark): # Find the position of each landmark
                # print(id, lm)
                h, w, c = img.shape # Width, height, and channel of the image
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                # if id == 0: # Draws a purple circle in the base of the hand (id = 0)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED) 
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_ITALIC, 3, (255, 0, 255), 2) # Shows fps in the window

    cv2.imshow("Image", img) # Opens window with webcam capture
    cv2.waitKey(1) # Waits before window pops
