import random #to randomize the image order in random generator function
import glob #to read images from file
import subprocess #to run the LINUX command
import os #to check if path exists
from PIL import Image
import numpy as np #using numpy arrays
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import wandb #to link the code with wandb
from wandb.keras import WandbCallback

run = wandb.init(project='superres') #connects to your project
config = run.config #the configurations for the run

config.num_epochs = 50 #number of times the model will cycle through the data
config.batch_size = 32
#size of input image
config.input_height = 32
config.input_width = 32
#size of output image
config.output_height = 256
config.output_width = 256

#address for validation and train datasets
val_dir = 'data/test'
train_dir = 'data/train'

# automatically get the data if it doesn't exist
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xzf - -C data", shell=True)

#used to limit the iterations during an epoch
config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size

#it is used to return the training images following call in line 114
def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    #The function takes in the batch processing size and processes batch_size images at a time
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    #input files contain the list of all files
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        #allocate space for batch_size of small and large images
        if counter+batch_size >= len(input_filenames):
            counter = 0
        for i in range(batch_size): #iterates through the images
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size
        #this keeps producing images to the fit_generator function in batches of batch_size


def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))


val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)


class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)
"""
#we are defining a sequential model
model = Sequential()
#first layer contains 3 nodes with filter size (3,3) and activation, padding and input shape are defined
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same',
                        input_shape=(config.input_width, config.input_height, 3)))
#we will get an output of size (config.input_width x config.input_height x 3)
model.add(layers.UpSampling2D())
#repeats the image and increases each dimension size by 2 => size is ((config.input_width x 2) x (config.input_height x 2) x 3)
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
#size is ((config.input_width x 4) x (config.input_height x 4) x 3)
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D())
model.add(layers.Conv2D(3, (3, 3), activation='relu', padding='same'))
#size is ((config.input_width x 8) x (config.input_height x 8) x 3)

# DONT ALTER metrics=[perceptual_distance]
model.compile(optimizer='adam', loss='mse',
              metrics=[perceptual_distance])
#we are defining the adam optimizer to control learning rate, loss as mse and perceptual_distance as a metric
"""
model = Sequential()
model.add(Conv2D(nb_filter=128, nb_row=9, nb_col=9, init='glorot_uniform',
                 activation='relu', border_mode='valid', bias=True, input_shape=(None, None, 1)))
model.add(Conv2D(nb_filter=64, nb_row=3, nb_col=3, init='glorot_uniform',
                 activation='relu', border_mode='same', bias=True))
# SRCNN.add(BatchNormalization())
model.add(Conv2D(nb_filter=1, nb_row=5, nb_col=5, init='glorot_uniform',
                 activation='linear', border_mode='valid', bias=True))
adam = Adam(lr=0.0003)
model.compile(optimizer=adam, loss='mse', metrics=[perceptual_distance])


#fit_generator is an advanced version of fit
#image data can be augmented on the fly using functions
#epoch is the number of times we go through the training set
model.fit_generator(image_generator(config.batch_size, train_dir),
                    steps_per_epoch=config.steps_per_epoch,
                    epochs=config.num_epochs, callbacks=[
                        ImageLogger(), WandbCallback()],
                    validation_steps=config.val_steps_per_epoch,
                    validation_data=val_generator)
