import roboflow

rf = roboflow.Roboflow(api_key='pX0IvaBYECIfBG2d043G')
project = rf.workspace().project("farag-v-elshorbagy-white-ball")

#can specify weights_filename, default is "weights/best.pt"
version = project.version(1)
version.deploy("yolov11", "trained-models", "g-ball2(white_latest).pt")

#example1 - directory path is "training1/model1.pt" for yolov8 model
version.deploy("yolov8", "trained-models", "model1.pt")
