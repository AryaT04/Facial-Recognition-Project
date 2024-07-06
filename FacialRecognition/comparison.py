# A program that uses facial recognition to compare any 2 given
# images and returns whether or not both pictures show the same person.
# The program displays both images, "True" if both images are the same person,
# "False" if both images are different people, and the distance(difference) between both images.
import cv2
import face_recognition as fr

# load images

controlPicture = fr.load_image_file('PictureA.jpg')
testPicture = fr.load_image_file('PictureB.jpg')

# transform images into color structure

controlPicture = cv2.cvtColor(controlPicture, cv2.COLOR_BGR2RGB)
testPicture = cv2.cvtColor(testPicture, cv2.COLOR_BGR2RGB)


# locate control face (returns tuple of face location)

faceAlocation = fr.face_locations(controlPicture)[0]
codedFaceA = fr.face_encodings(controlPicture)[0]


# locate test face (returns tuple of face location)

faceBlocation = fr.face_locations(testPicture)[0]
codedFaceB = fr.face_encodings(testPicture)[0]

# Frame face in image (using tuple returned by faceAlocation)

cv2.rectangle(controlPicture,
              (faceAlocation[3], faceAlocation[0]),
              (faceAlocation[1], faceAlocation[2]),
              (127,0,255), 2)

# Frame face in image (using tuple returned by faceBlocation)

cv2.rectangle(testPicture,
              (faceBlocation[3], faceBlocation[0]),
              (faceBlocation[1], faceBlocation[2]),
              (127,0,255), 2)


# compare both faces

result = fr.compare_faces([codedFaceA],codedFaceB)

print(result)



# measurement of distances

distance = fr.face_distance([codedFaceA], codedFaceB)

# display results 
cv2.putText(testPicture,
            f"{result} {distance.round(2)}",
            (50,50),
            cv2.FONT_HERSHEY_COMPLEX,
            0.5,
            (127,0,255),
            1)


# show images

cv2.imshow("Control picture", controlPicture)
cv2.imshow("Test Picture", testPicture)

# keep program running

cv2.waitKey(0)