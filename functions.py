import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

def split_data(train_dataset, test_dataset, val_dataset, RANDOM_SEED = 1):
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

    return X_train, y_train, X_val, y_val, X_test_A, y_test_A, X_test_B, y_test_B



def get_label_counts(y_train,
                     y_val,
                     y_test_A,
                     y_test_B,
                     class_labels = [0,1],
                     class_names = ["0","1"]):
    
    count = {}
    
    for dataset, dataset_name in zip([y_train,y_val,y_test_A,y_test_B],["Train","Val","Test A","Test B"]):
        print("")
        for label, name in zip(class_labels,class_names):
            c = dataset.count(label)
            print(dataset_name, name, c)


def build_basic_model(input_shape = (128,128,1), num_classes = 2):

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
    model.add(Dense(units = 1 if num_classes < 3 else num_classes , activation = 'sigmoid'))
    model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    model.summary()

    return model