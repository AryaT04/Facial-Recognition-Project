# A program that uses facial recognition to capture an image using
# the webcam and compares the image to a database of images to find a match.
# Once a match is found, the program displays the captured image witht he name
# of the matched person. 

import cv2
import face_recognition as fr
import os
import numpy


# create database
path = 'imageDatabase'
rosterImages = []
rosterNames = []
roster = os.listdir(path)

for name in roster:
    thisImage = cv2.imread(f"{path}\\{name}")
    rosterImages.append(thisImage)
    rosterNames.append(os.path.splitext(name)[0])



# encode images 

def encode(images):

    encodedList = []

    # convert all images to rgb

    for image in images:

        # transform images into color structure

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # encode

        encoded = fr.face_encodings(image)[0]

        # add to list

        encodedList.append(encoded)

    # return the list of encoded images

    return encodedList

encodedRosterImages = encode(rosterImages)

# take a picture using webcam

capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# read the captured image

success, image = capture.read()

if not success:
    print("No capture.")

else:

    # check to see if there is a face in the capture

    capturedFace = fr.face_locations(image)

    # encode the captured face

    encodedCapturedFace = fr.face_encodings(image, capturedFace)

    # search for match

    for face , location_face in zip(encodedCapturedFace, capturedFace):
       
       # look for distances between captured face and faces from the database

        matches = fr.compare_faces(encodedRosterImages, face)
        distances = fr.face_distance(encodedRosterImages, face)

        

        # compare all distances, smallest distance is the match

        matchIndex = numpy.argmin(distances)

        # show coincidences 

        if distances[matchIndex] > 0.6:
            print("No match found.")
        else:

            # search for name

            rosterName = rosterNames[matchIndex]

            # frame face and add text to display name of match 

            y1, x2, y2, x1 =  location_face
            cv2.rectangle(image,
                          (x1, y1),
                          (x2, y2),
                          (127,0,255), 
                          2)
            cv2.rectangle(image,
                          (x1,y2-35),
                          (x2,y2),
                          (127,0,255), 
                          cv2.FILLED)
            cv2.putText(image,
                        rosterName,
                        (x1+6, y2-6),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255,255,255),
                        1)
            cv2.putText(image,
                        "Match Found",
                        (x1, y1-15),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (127,0,255),
                        2)


            # show the image obtained

            cv2.imshow('Captured Image', image)

            # keep window open

            cv2.waitKey(0)