from controller import Robot
from controller import DistanceSensor
from controller import Keyboard
from controller import Camera
import playsound
from PIL import Image
from gtts import gTTS
import random
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np


Turtle = Robot()
camera = Camera("CAM")
keyboard = Keyboard()
voiced_text = gTTS("впереди препятствие",lang='ru')
file = "audio"+str(random.randint(0,10))+".mp3"
voiced_text.save(file)
playsound.playsound(file)

#Подключаем нейросеть YOLO
net = cv2.dnn.readNet('C:/Users/user/yolov3.weights','C:/Users/user/yolov3.cfg') #Загружаем нейросеть YOLO
with open ('C:/Users/user/Desktop/ДИПЛОМА/yolo/coco.names') as f:
    labels = f.read().strip().split('\n')
 

def run_robot(robot):
    time_step = 32
    max_speed = 6.28
    camera.enable(time_step)
    keyboard.enable(time_step)
    k_obstacles = 0 
    
    motor_cmd = {
                ord('S'): (max_speed,max_speed),
                ord('W'): (-max_speed,-max_speed),
                ord('A'): (-0.5*max_speed,0.5*max_speed),
                ord('D'): (0.5*max_speed,-0.5*max_speed),
                
                }

    left_motor = robot.getDevice( 'left wheel motor')
    right_motor = robot.getDevice( 'right wheel motor')
    left_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setPosition(float('inf'))
    right_motor.setVelocity(0.0)
    
    ds_right=robot.getDevice('ds_right')
    ds_right.enable(time_step)
    ds_left=robot.getDevice('ds_left')
    ds_left.enable(time_step)
    
    def command_motors(cmd):
        left_motor.setVelocity(cmd[0])
        right_motor.setVelocity(cmd[1])    
    
    while robot.step(time_step) != -1:
        
        Camera.getImage(camera)
        Camera.saveImage(camera,'C:/Users/user/Pictures/camera.png',1)
        
        frame = cv2.imread('C:/Users/user/Pictures/camera.png')
        height,width,_=frame.shape
        blob = cv2.dnn.blobFromImage(frame,1/255,(608,608),(0,0,0),swapRB=True)
        net.setInput(blob) #Подготовка нейросети к запуску

        #получение выхода нейросети на каждом слое
        layers = net.getLayerNames()
        out_layer_indexes_arr=net.getUnconnectedOutLayers()
        out_layer_indexes = []
        out_layer_names =[]
        for el in out_layer_indexes_arr:
            out_layer_indexes.append(el-1)
        #out_layer_indexes = [index[0]-1 for index in out_layer_indexes_arr ]
        #out_layer_indexes  
        for index in out_layer_indexes:
            out_layer_names.append(layers[index])

        #Функция для рисования объекта на изображении
        def draw_object(x,y,w,h,img,obj):
            #x,y - центр полученного объекта на изображении
            width = 2
            color = [250,0,0]
            img=cv2.rectangle(img,(x-w//2,y-h//2),(x+w//2,y+h//2),color,width)
            
            font_size = 2
            
            text = obj
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img,text,(x-w//2,y-h//2-10),font,font_size,color,width)
            return img
                
        out_layers = net.forward(out_layer_names) #Выход нейросети. Вероятность нахождения объектов на изображении
        
        object_boxes=[]
        object_probas=[]
        objects=[]
        
        for layer in out_layers:
            for result in layer:
                x,y,w,h = result[:4] 
                probas = result[5:]
                x=int(x*width)
                y=int(y*height)
                w=int(w*width)
                h=int(h*height)
                max_proba_index = np.argmax(probas)
                max_proba = probas[max_proba_index]
                if max_proba>0:
                    object_boxes.append([x,y,w,h])
                    object_probas.append(float(max_proba))
                    objects.append(labels[max_proba_index])
                    #print(x,y,w,h,labels[max_proba_index])
                    
            
        filtered_boxes_indexes = cv2.dnn.NMSBoxes(object_boxes,object_probas,0.0,0.2)
        
        for index_arr in filtered_boxes_indexes:
            index = index_arr
            box = object_boxes[index]
            object_todraw = objects[index]
            x,y,w,h = box
            img=draw_object(x,y,w,h,frame,object_todraw)
                
        cv2.imwrite('C:/Users/user/Pictures/frame1492.jpg',img)  
        plt.imshow(img)      
        
        
        key = keyboard.getKey()        
        ds_right_value = ds_right.getValue()
        ds_left_value = ds_left.getValue()
      
        left_motor.setVelocity(-max_speed)
        right_motor.setVelocity(-max_speed)
        if key in motor_cmd.keys():
            command_motors(motor_cmd[key])
            pass 
        
        if key == ord('E'): 
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        
                 
        if (ds_right_value < 1000 or ds_left_value < 1000 ):
            k_obstacles += 1
            left_motor.setVelocity(max_speed)
            right_motor.setVelocity(-max_speed)
            if k_obstacles == 1:
                playsound.playsound(file)
        else:
            k_obstacles = 0
               

if __name__ == "__main__":
    
    run_robot(Turtle)