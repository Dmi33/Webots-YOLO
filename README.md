# Webots-YOLO

Перед запуском скачать конфигурацию и веса нейросети для распознавания объектов на изображении - YOLO V3

https://pjreddie.com/darknet/yolo/

#### Данный программный код является контроллером для робота, созданного в среде разработки Webots.

![image](https://user-images.githubusercontent.com/108614519/177049664-4afe8b03-2248-485f-a495-1c9a4a607435.png)

#### Функционал:

# • Управление поворотом робота с помощью клавиатуры;

# • Получение данных с датчика расстояния. Когда робот находится в метре от препятствия - воспроизводится голосовое оповещение и происходит обход препятствий;

# • Передача показаний датчиков на ROS для дальнейшей реализации алгоритма ORB-SLAM;

https://www.ros.org/

https://github.com/raulmur/ORB_SLAM2

# • Применение нейронной сети Yolo для распознавания препятствий на изображении, получаемом с камеры в режиме реального времени;

![image](https://user-images.githubusercontent.com/108614519/177051211-d63ff748-41ad-4451-9117-75c779de3ee3.png)



