import os
import re
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
import zmq
import speech_recognition as sr

from queue import Queue
from threading import Thread
from multiprocessing import Process, Queue, Pool
from utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels
from object_detection.utils import label_map_util

CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
PATH_TO_CKPT = os.path.join(CWD_PATH, 'object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def is_string_same(str1, str2):
 
    if str1 != str2:
        return 0
    else:
        return 1
    
def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)


# Voice activation using SpeechRecognition
def listen_for_activation_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for activation command...")
        audio = recognizer.listen(source)
    
    try:
        command = recognizer.recognize_google(audio)
        print("You said: " + command)
        return "start detection" in command.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return False
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return False

def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()

def detect_objects_diff(image_np, sess, detection_graph, target_object):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    # Run detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    detected_classes = []
    min_score_thresh = 0.5  # Threshold for detection confidence

    for i in range(int(num_detections[0])):
        if scores[0][i] > min_score_thresh:
            class_id = int(classes[0][i])
            class_name = category_index[class_id]['name']
            if class_name == target_object:
                detected_classes.append(class_name)
                break  # Break after detecting the first instance if only one is needed

    return detected_classes

def listen_for_object_name():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say the object you want to detect...")
        audio = recognizer.listen(source)
    
    try:
        # Using Google's speech recognition to convert audio to text
        speech_output = recognizer.recognize_google(audio)
        print("Recognized speech: {0}".format(speech_output))
        return speech_output.lower()
    except sr.UnknownValueError:
        print("Could not understand the audio")
        return None
    except sr.RequestError as e:
        print("Speech recognition error; {0}".format(e))
        return None
    

#publish information about detected objects via zmq
def publish_detected_object():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    addr = '127.0.0.1'  # remote ip or localhost
    port = "5556"  # same as in the pupil remote gui
    socket.bind("tcp://{}:{}".format(addr, port))
    time.sleep(1)

    while True:
        #publish the label only if there is a fixation and label    
        label_conf = label_q.get()
        print('label',label_conf.split())

        if label_conf:
            #print(self.label, self.fixation_norm_pos)
            topic = 'detected_object'
            # this only works for one and 2 word objects for now 
            if len(label_conf.split())==2:
                label = label_conf.split()[0][:-1]
                confidence = label_conf.split()[1][:-1]
            if len(label_conf.split())==3:
                label = label_conf.split()[0] + ' ' + label_conf.split()[1][:-1]
                confidence = label_conf.split()[2][:-1]

            print ('%s %s %s' % (topic, label, confidence))
            try:
                socket.send_string('%s %s %s' % (topic, label, confidence))
            except TypeError:
                socket.send('%s %s' % (topic, label, confidence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int, default=-1, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int, default=1280, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=720, help='Height of the frames in the video stream.')
    args = parser.parse_args()

    input_q = Queue(2)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    label_q = Queue()

    # Start worker and publishing processes
    t = Thread(target=worker, args=(input_q, output_q))
    t.daemon = True
    t.start()

    p = Process(target=publish_detected_object, args=(label_q,))
    p.daemon = True
    p.start()

    video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()
    fps = FPS().start()

    try:
        while True:  # Main loop
            if listen_for_activation_command():
                target_object = listen_for_object_name()
                if target_object:
                    print("Looking for {0}...".format(target_object))
                    while True:
                        frame = video_capture.read()
                        input_q.put(frame)

                        if not output_q.empty():
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            data = output_q.get()
                            rec_points = data['rect_points']
                            class_names = data['class_names']
                            class_colors = data['class_colors']
                            for point, name, color in zip(rec_points, class_names, class_colors):
                                print("name[0]", name[0])
                                res2 = " ".join(re.findall("[a-zA-Z]+", name[0]))
                                result = is_string_same(target_object, res2)
                                if result==1:
                                    label_q.put(target_object)
                                    print('Object detected: {}'.format(name[0]))
                                      # Send detection label
                                    break  # Exit the inner loop after sending the label
                                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                                        (int(point['xmax'] * args.width), int(point['ymax'] * args.height)), color, 3)
                                cv2.rectangle(frame, (int(point['xmin'] * args.width), int(point['ymin'] * args.height)),
                                        (int(point['xmin'] * args.width) + len(name[0]) * 6,
                                        int(point['ymin'] * args.height) - 10), color, -1, cv2.LINE_AA)
                                cv2.putText(frame, name[0], (int(point['xmin'] * args.width), int(point['ymin'] * args.height)), font,
                                    0.3, (0, 0, 0), 1)
                            print('blajh')
                            cv2.imshow('Video', frame)

                            if result:
                                break  # Exit to wait for new "start detection" command after a detection

                        fps.update()

                    print("Waiting for new 'start detection' command...")
                else:
                    print("No valid object name recognized. Please try again.")

    except KeyboardInterrupt:
        print("Keyboard Interrupt received, exiting.")

    finally:
        fps.stop()
        print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
        print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))
        video_capture.stop()
        cv2.destroyAllWindows()
        input_q.put(None)  # Signal the worker to terminate
        t.join()
        p.terminate()
        p.join()

