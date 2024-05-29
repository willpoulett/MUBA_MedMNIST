import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import utils
import random
import pandas as pd

def split_data(train_dataset, test_dataset, val_dataset, RANDOM_SEED = 1, one_hot_encoded = True, num_classes = 2, image_size = 128, three_d = False):
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

    X_train = np.array(X_train).reshape(-1, image_size, image_size, image_size if three_d else 1)
    X_val = np.array(X_val).reshape(-1, image_size, image_size, image_size if three_d else 1)
    X_test_A = np.array(X_test_A).reshape(-1, image_size, image_size, image_size if three_d else 1)
    X_test_B = np.array(X_test_B).reshape(-1, image_size, image_size, image_size if three_d else 1)


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
                if np.argmax(y_train[id]) != np.argmax(y1):
                    new_label = True
                    X2, y2 = X_train[id], y_train[id]

                    y_train_mixup.append( alpha*y1 + (1-alpha)*y2 )       
                    X_train_mixup.append( alpha*X1 + (1-alpha)*X2 )      

    return X_train_mixup, y_train_mixup
        

def generate_testing_mixup_images(test_set_A_df, X_test_A, MUBA_ITERS, classes = 2):
    # Appending to arrays is more efficient than appending a row at a time to a df.
    alphas = {}
    class_indexes = {}
    mixup_image = []
    labels = []

    for class_num in range(classes):
        alphas[f"alpha_class_{str(class_num)}"] = []
        class_indexes[f"class_{str(class_num)}_index"] = []

    for class_num in range(classes-1): # Iterate through the class numbers
        for index_0, row_0 in test_set_A_df[test_set_A_df["class"]==class_num].iterrows(): # For each class X image
            for index_1, row_1 in test_set_A_df[test_set_A_df["class"]>class_num].iterrows(): # For each class > X image
                class_row_1 = row_1["class"]
                for i in range(MUBA_ITERS):
                    alpha = (1/MUBA_ITERS) * np.random.rand() + ( (i) / MUBA_ITERS) # Create an alpha value inside a bin of width 1/MUBA_ITERS
                    new_img = alpha * X_test_A[int(row_0["image_index"])] + (1 - alpha) * X_test_A[int(row_1["image_index"])]
                    
                    label = class_num if alpha > 0.5 else class_row_1 # If alpha is greater than 0.5, there is a higher proportion of a class 0 image.
                    
                    for c in range(classes):
                        if c == class_num:
                            alphas[f"alpha_class_{str(c)}"].append(alpha)
                            class_indexes[f"class_{str(c)}_index"].append(int(index_0))
                        elif c == class_row_1:
                            alphas[f"alpha_class_{str(c)}"].append(float(1-alpha))
                            class_indexes[f"class_{str(c)}_index"].append(int(index_1))
                        else:
                            alphas[f"alpha_class_{str(c)}"].append(0)
                            class_indexes[f"class_{str(c)}_index"].append(None)

                    mixup_image.append(new_img)
                    labels.append(label)

    images_types_labels = {"image": mixup_image,
                         "type":["mix" for i in range(len(labels))],
                         "label":labels}
    merged_dict = alphas | class_indexes | images_types_labels

    return pd.DataFrame(merged_dict)

def find_boundary_points(muba_df,
                         X_test_A,
                         MUBA_ITERS: int = 60,
                         classes=2):
    """Generates new images with alpha values between those at which a model changes it's prediction

    Args:
        muba_df (DataFrame): DataFrame containing all mixed up images
        MUBA_ITERS (int, optional): _description_. Defaults to 60.

    Returns:
        boundary_points_df (DataFrame): Contains all boundary points
    """

    # Appending to arrays is more efficient than appending a row at a time to a df.
    alphas = {}
    class_indexes = {}
    mixup_image = []
    labels = []

    for class_num in range(classes):
        alphas[f"alpha_class_{str(class_num)}"] = []
        class_indexes[f"class_{str(class_num)}_index"] = []

    for i in range(int((len(muba_df))/MUBA_ITERS)):

        # Create a mask to split df in to blocks of MUBA_ITERS
        mask = (muba_df.index >= MUBA_ITERS*i) & (muba_df.index < MUBA_ITERS*i + MUBA_ITERS)
        window_df = muba_df.loc[mask]

        # Find the index in which the prediction changes
        changing_pred_index = (window_df["argmax_pred"].diff()[window_df["argmax_pred"].diff() != 0].index.values)
        for index, row in window_df.iterrows():
            
            if index in changing_pred_index[1:]:
                
                row0 = window_df.loc[[index]]
                row1 = window_df.loc[[index-1]]
                
                row_0_class_0 = row0["label"][index]
                
                # Get id for both images
                image_0_id = int(row0[f"class_{row_0_class_0}_index"]) # Find id of image with largest alpha
                index_columns = list(class_indexes.keys())
                for id_col in index_columns:
                    id = row0[id_col][index]
                    if not np.isnan(id) and id != image_0_id:
                        image_1_id = int(id)
                        row_0_class_1 = [int(s) for s in id_col.split("_") if s.isdigit()][0]
                        
                # Work out alpha class 0 as the difference between alpha class 0 in both images
                alpha_class_0 = ( float(row0[f"alpha_class_{row_0_class_0}"]) + float(row1[f"alpha_class_{row_0_class_0}"])) / 2
                alpha_class_1 = 1 - alpha_class_0
                image = alpha_class_0 * X_test_A[image_0_id] + alpha_class_1 * X_test_A[image_1_id]
                
                label = row_0_class_0 if alpha_class_0 > 0.5 else row_0_class_1
                
                for c in range(classes):
                    if c == row_0_class_0:
                        alphas[f"alpha_class_{str(c)}"].append(alpha_class_0)
                        class_indexes[f"class_{str(c)}_index"].append(image_0_id)
                    elif c == row_0_class_1:
                        alphas[f"alpha_class_{str(c)}"].append(float(alpha_class_1))
                        class_indexes[f"class_{str(c)}_index"].append(image_1_id)
                    else:
                        alphas[f"alpha_class_{str(c)}"].append(0)
                        class_indexes[f"class_{str(c)}_index"].append(None)

                mixup_image.append(image)
                labels.append(label)

    images_types_labels = {"image": mixup_image,
                         "type":["boundary" for i in range(len(labels))],
                         "label":labels}
    merged_dict = alphas | class_indexes | images_types_labels

    return pd.DataFrame(merged_dict)