import os
import faceRecognition as fr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "trainingImages")


faces,faceID=fr.get_labels_for_images(image_dir)
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('TrainedModel.yml')







