from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

num_classes = 2
resnet_weights_path = "<YOUR PATH>/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))


my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = data_generator.flow_from_directory('<YOUR PATH>/rural_and_urban_photos/train',
                                                     target_size=(image_size, image_size),
                                                     batch_size=12,
                                                     class_mode='categorical')


validation_generator = data_generator.flow_from_directory('<YOUR PATH>/rural_and_urban_photos/val',
                                                          target_size=(image_size,image_size),
                                                          class_mode='categorical')


my_new_model.fit_generator(train_generator,
                           steps_per_epoch=6,
                           validation_data=validation_generator,
                           validation_steps=1)


my_new_model.save_weights('urban_model.h5')


def read_and_prep_images(img_paths, img_height=image_size, img_width=image_size):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    return preprocess_input(img_array)


from os.path import join

image_dir = '<YOUR PATH>/Pictures/'
img_paths = [join(image_dir, filename) for filename in
             ['urban1.jpg']]


import numpy as np
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

test_data = read_and_prep_images(img_paths)
preds = my_new_model.predict(test_data)

print(preds)

print(preds[0, 0])




