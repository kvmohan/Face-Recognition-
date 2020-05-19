# Face-Recognition-

In this we have used the Transfer Learning for training the model and recognition the faces.

## TASK

To create a model which detect the face or make a face detection model using the transfer learning

### We have used the MobileNet to train the model with our images

### Step 1: We have import the pretrained model or load the mobilenet model

![](1.png)

### Step 2: Freeze all layers of the model expext the last layers as we have to make changes in that layer

![](2.png)

### Step 3: Make a function that return the FC Head. This is the layer creation to train our model

![](3.png)

### Step 4: Now add the layer which we have created to the MobileNet layers where we have freezed all the layers

![](4.png)

### Step 5: Now load the dataset which you have to train and this should be in the folder and the output depends on the folder name

![](5.png)

### Step 6: Now compile the model and we have use imagedatagenerator to generate our images for the training

![](6.png)

### Step 7: Now finally train the model and after the epochs completed then save that model

![](7.png)

### Step 8: Now load the saved model to test our model.

![](8.png)

### Step 9: Test our model with the testing images we have putted in the folder. Here we have made a small python programm to do so and used cv2 module to show the images.

* Here the python program is using the random photo from the test folder for testing.

![](9.png)

### Outputs:

![](out1.png)

![](out6.png)

![](out5.png)

![](out2.png)

![](out4.png)

![](out3.png)


## Author

[SAURAV PATEL](https://www.linkedin.com/in/saurav-patel-148539151/)
