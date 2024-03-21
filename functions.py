import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
import random

def split_data(train_dataset, test_dataset, val_dataset, RANDOM_SEED = 1, one_hot_encoded = True, num_classes = 2, image_size = 128):
    """_summary_

    Args:
        train_dataset (_type_): _description_
        test_dataset (_type_): _description_
        val_dataset (_type_): _description_
        RANDOM_SEED (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []

    for X, y in train_dataset:
        X_train.append(np.asarray(X))
        y_train.append(y[0])

    for X, y in test_dataset:
        X_test.append(np.asarray(X))
        y_test.append(y[0])

    for X, y in val_dataset:
        X_val.append(np.asarray(X))
        y_val.append(y[0])

    X_test_A, X_test_B, y_test_A, y_test_B = train_test_split(X_test,
    y_test, test_size=0.7, random_state=RANDOM_SEED)

    if one_hot_encoded:
        y_train  = utils.to_categorical(y_train ,dtype ="uint8", num_classes = num_classes)
        y_val    = utils.to_categorical(y_val   ,dtype ="uint8", num_classes = num_classes)
        y_test_A = utils.to_categorical(y_test_A,dtype ="uint8", num_classes = num_classes)
        y_test_B = utils.to_categorical(y_test_B,dtype ="uint8", num_classes = num_classes)

    X_train = np.array(X_train).reshape(-1, image_size, image_size, 1)
    X_val = np.array(X_val).reshape(-1, image_size, image_size, 1)
    X_test_A = np.array(X_test_A).reshape(-1, image_size, image_size, 1)
    X_test_B = np.array(X_test_B).reshape(-1, image_size, image_size, 1)


    return X_train, y_train, X_val, y_val, X_test_A, y_test_A, X_test_B, y_test_B


def get_label_counts(y_train,
                     y_val,
                     y_test_A,
                     y_test_B,
                     class_names = ["0","1"]):
    
    for dataset, dataset_name in zip([y_train,y_val,y_test_A,y_test_B],["Train","Val","Test A","Test B"]):
        
        unique_rows, count = np.unique(dataset, axis=0,return_counts=True)
        out = {tuple(i):j for i,j in zip(unique_rows,count)}
        print(f"\n{dataset_name}\n{out}  {class_names}")


def build_basic_model(input_shape = (128,128,1), num_classes = 2, final_activation='softmax', loss = "categorical_crossentropy"):

    model = Sequential()
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Flatten())
    model.add(Dense(units = 128 , activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes , activation = final_activation))
    model.compile(optimizer = "rmsprop" , loss = loss , metrics = ['accuracy'])
    model.summary()

    return model


def argmax_array(array):
    return([np.argmax(item) for item in array])

def generate_training_mixup_images(X_train, y_train, ITERS = 1):

    X_train_mixup = []
    y_train_mixup = []

    for X1, y1 in zip(X_train, y_train):

        for i in range(ITERS):
        
            # Sample from the beta function
            alpha = np.random.beta(0.2, 0.2, 1)

            new_label = False
            while new_label == False:
                id=random.randint(0,len(y_train)-1)
                # Check label is from other class
                if y_train[id][0] != y1[0]:
                    new_label = True
                    X2, y2 = X_train[id], y_train[id]

                    y_train_mixup.append( alpha*y1 + (1-alpha)*y2 )       
                    X_train_mixup.append( alpha*X1 + (1-alpha)*X2 )      

    return X_train_mixup, y_train_mixup
        