from __future__ import print_function, division
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding3D
from keras.layers import MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling3D, Conv3D, Deconv3D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from keras.models import model_from_json
from keras.models import load_model

class EncoderDecoderGAN():
    def __init__(self):
        self.vol_rows = 64
        self.vol_cols = 64
        self.vol_height = 64
        self.mask_height = 32
        self.mask_width = 32
        self.mask_length = 32

        self.channels = 1
        self.num_classes = 2
        self.vol_shape = (self.vol_rows, self.vol_cols, self.vol_height, self.channels)
        self.missing_shape = (self.mask_height, self.mask_width, self.mask_length, self.channels)

        optimizer = Adam(0.0002, 0.5)

        try: 
            self.discriminator = load_model('saved_model/discriminator.h5')
            self.generator = load_model('saved_model/generator.h5')

            print("Loaded checkpoints")
        except:
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator() 
            print("No checkpoints found")   

        # discriminator
        
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # generator
        

        # The generator takes noise as input and generates the missing part
        masked_vol = Input(shape=self.vol_shape)
        gen_missing = self.generator(masked_vol)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated voxels as input and determines
        # if it is generated or if it is a real voxels
        valid = self.discriminator(gen_missing)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(masked_vol , [gen_missing, valid])
        self.combined.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        # Encoder
        model.add(Conv3D(32, kernel_size=5, strides=2, input_shape=self.vol_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(64, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(128, kernel_size=5, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Conv3D(512, kernel_size=1, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        # Decoder
        model.add(UpSampling3D())
        model.add(Deconv3D(256, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Deconv3D(128, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling3D())
        model.add(Deconv3D(64, kernel_size=5, padding="same"))
        model.add(Activation('relu'))
        model.add(BatchNormalization(momentum=0.8))
        # 
        model.add(UpSampling3D())
        #
        model.add(Deconv3D(self.channels, kernel_size=5, padding="same"))
        model.add(Activation('tanh'))

        model.summary()

        masked_vol = Input(shape=self.vol_shape)
        gen_missing = model(masked_vol)

        return Model(masked_vol, gen_missing)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv3D(64, kernel_size=3, strides=2, input_shape=self.missing_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv3D(256, kernel_size=3, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        vol = Input(shape=self.missing_shape)
        validity = model(vol)

        return Model(vol, validity)

    def generateWall(self):

        x, y, z = np.indices((64, 64, 64))
        voxel = (x < 56)&(x>10) & (y>10) & (y < 56) & (z>20)&(z < 50)
        
        # add channel 
        voxel = voxel[...,np.newaxis].astype(np.float)
        # repeat 1000 times
        voxels = list()
        for i in range(1000):
            voxels.append(voxel)
        voxels = np.asarray(voxels)
        return voxels

    def mask_randomly(self, vols):
        y1 = np.random.randint(0, self.vol_rows - self.mask_height, vols.shape[0])
        y2 = y1 + self.mask_height
        x1 = np.random.randint(0, self.vol_cols - self.mask_width, vols.shape[0])
        x2 = x1 + self.mask_width
        z1 = np.random.randint(0, self.vol_height - self.mask_length, vols.shape[0])
        z2 = z1 + self.mask_length


        masked_vols = np.empty_like(vols)
        missing_parts = np.empty((vols.shape[0], self.mask_height, self.mask_width, self.mask_length, self.channels))
        for i, vol in enumerate(vols):
            masked_vol = vol.copy()
            _y1, _y2, _x1, _x2, _z1, _z2 = y1[i], y2[i], x1[i], x2[i], z1[i], z2[i]
            missing_parts[i] = masked_vol[_y1:_y2, _x1:_x2, _z1:_z2, :].copy()
            masked_vol[_y1:_y2, _x1:_x2, _z1:_z2, :] = 0
            masked_vols[i] = masked_vol

        return masked_vols, missing_parts, (y1, y2, x1, x2, z1, z2)

    def train(self, epochs, batch_size=16, sample_interval=50):

        X_train = self.generateWall()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            vols = X_train[idx]

            masked_vols, missing_parts, _ = self.mask_randomly(vols)

            # Generate a batch 
            gen_missing = self.generator.predict(masked_vols)
            # print(gen_missing.shape)

            d_loss_real = self.discriminator.train_on_batch(missing_parts, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_missing, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train Generator
            g_loss = self.combined.train_on_batch(masked_vols, [missing_parts, valid])

            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # save generated samples
            if epoch % sample_interval == 0:
                idx = np.random.randint(0, X_train.shape[0], 2)
                vols = X_train[idx]
                self.sample_images(epoch, vols)
                self.save_model()
    
    def sample_images(self, epoch, vols):
        r, c = 2, 2

        masked_vols, missing_parts, (y1, y2, x1, x2, z1, z2) = self.mask_randomly(vols)
        gen_missing = self.generator.predict(masked_vols)
        gen_missing = np.where(gen_missing > 0.5, 1, 0)
        fig = plt.figure(figsize=plt.figaspect(0.5), dpi=300)

        vols = 0.5 * vols + 0.5

        for i in range(2):
            masked_vol = masked_vols[i]
            masked_vol = masked_vol[:, :, :, 0].astype(np.bool)
            colors1 = np.empty(masked_vol.shape, dtype=object)
            colors1[masked_vol] = 'red'
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.voxels(masked_vol, facecolors=colors1, edgecolor='black', linewidth=0.2)
            
            filled_in = np.zeros_like(masked_vol)
            # filled_in = vols[i].copy()            
            one_gen_missing = gen_missing[i]
            one_gen_missing = one_gen_missing[:, :, :, 0].astype(np.bool)
            filled_in[y1[i]:y2[i], x1[i]:x2[i], z1[i]:z2[i]] = one_gen_missing  
            fill = filled_in       
            combine_voxels = masked_vol | fill
            
            colors2 = np.empty(combine_voxels.shape, dtype=object)
            colors2[masked_vol] = 'red'
            colors2[fill] = 'blue'

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.voxels(combine_voxels, facecolors=colors2, edgecolor='black', linewidth=0.2)
            # ax.voxels(masked_vol, facecolors=colors1, edgecolor='k')
                     
        # plt.show()
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.h5" % model_name
            model.save(model_path)

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")

if __name__ == '__main__':
    context_encoder = EncoderDecoderGAN()
    context_encoder.train(epochs=3000, batch_size=32, sample_interval=200)