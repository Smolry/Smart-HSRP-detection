model used= yolov8n-cls.pt

Image classification is the simplest of the three tasks and involves classifying an entire image into one of a set of predefined classes.

The output of an image classifier is a single class label and a confidence score. Image classification is useful when you need to know only what class an image belongs to and don't need to know where objects of that class are located or what their exact shape i

used different parameter values:
1) 30 epochs, 224 imgsz, batch= 8 with accuracy= 97%
2) 30 epochs, 224 imgsz, batch= 16 with accuracy= 95%
3) 30 epochs, 320 imgsz, batch=16 with accuracy= 97%
4) 50 epochs, 240 imgsz, batch=32 with accuracy= 98%
5) epochs=50, imgsz=384, batch= 32 with accuracy= 99%

it appeared that because data was small no. of epochs and batch size made less dofference in result.
