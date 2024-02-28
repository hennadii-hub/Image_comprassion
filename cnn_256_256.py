import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import History
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Check if GPU is available and set memory growth
physical_devices = tf.config.list_physical_devices('GPU')
print(len(physical_devices))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

# Define the directory containing the images
data_dir = 'data2'

# Get the list of image files
image_files = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]

# Load and format images with grayscale conversion
images = [rgb2gray(img_to_array(load_img(os.path.join(data_dir, file), target_size=(256, 256)))) for file in image_files]
images = np.array(images)

# Check the number of images
if len(images) < 1:
    raise ValueError("There are no images in the dataset.")

# Normalize pixels to the range [0, 1]
images = images.astype('float32') / 255.0

# Split into training and testing sets
x_train, x_test = train_test_split(images, test_size=0.3, random_state=42)

# Reshape images to match the expected input shape
x_train = np.reshape(x_train, (len(x_train), 256, 256, 1))
x_test = np.reshape(x_test, (len(x_test), 256, 256, 1))

def create_deep_conv():
    input_img = Input(shape=(256, 256, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    coded = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x_decoded = Conv2D(64, (3, 3), activation='relu', padding='same')(coded)
    x_decoded = UpSampling2D((2, 2))(x_decoded)
    x_decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(x_decoded)
    x_decoded = UpSampling2D((2, 2))(x_decoded)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x_decoded)
    coder = Model(input_img, coded, name="coder")
    decoder = Model(coded, decoded, name="decoder")
    CNN = Model(input_img, decoder(coder(input_img)), name="CNN")
    return coder, decoder, CNN

# Create the deep convolutional autoencoder models
c_coder, c_decoder, CNN = create_deep_conv()
CNN.compile(optimizer='adam', loss='binary_crossentropy')

# Display summary of the CNN model
CNN.summary()

# Train the CNN model
history = History()
CNN.fit(x_train, x_train,
                epochs=10,
                batch_size=10,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[history])

n = 10

# Display the reconstructed images
imgs = x_test[:n]
coded_imgs = c_coder.predict(imgs, batch_size=n)
decoded_imgs = c_decoder.predict(coded_imgs, batch_size=n)

plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(imgs[i].reshape(256, 256), cmap='gray')  # Adjusted to the image size
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Display compressed images
for i in range(n):
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(coded_imgs[i].reshape(64, 64, 128)[..., 0], cmap='gray')  # Adjusted to the encoder output size
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Display reconstructed images
for i in range(n):
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].reshape(256, 256), cmap='gray')  # Adjusted to the image size
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# Calculate and plot metrics for each image
mse_values = []
mae_values = []
psnr_values = []

for i in range(n):
    mse = np.mean((imgs[i] - decoded_imgs[i]) ** 2)
    mae = np.mean(np.abs(imgs[i] - decoded_imgs[i]))
    psnr_value = psnr(imgs[i], decoded_imgs[i], data_range=imgs[i].max() - imgs[i].min())
    
    mse_values.append(mse)
    mae_values.append(mae)
    psnr_values.append(psnr_value)
    
    print(f"Image {i + 1}: MSE = {mse}, MAE = {mae}, PSNR = {psnr_value}")

# Plot MSE
plt.figure(figsize=(10, 5))
plt.plot(range(n), mse_values, marker='o', color='r')
plt.xlabel('Image Index')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE for Reconstructed Images')
plt.xticks(range(n))
plt.grid(axis='x')
plt.show()

# Plot MAE
plt.figure(figsize=(10, 5))
plt.plot(range(n), mae_values, marker='o', color='g')
plt.xlabel('Image Index')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE for Reconstructed Images')
plt.xticks(range(n))
plt.grid(axis='x')
plt.show()

# Plot PSNR
plt.figure(figsize=(10, 5))
plt.plot(range(n), psnr_values, marker='o', color='b')
plt.xlabel('Image Index')
plt.ylabel('PSNR (dB)')
plt.title('PSNR for Reconstructed Images')
plt.xticks(range(n))
plt.grid(axis='x')
plt.show()

# Plot training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the models
c_coder.save('c_coder_model.h5')
c_decoder.save('c_decoder_model.h5')
CNN.save('CNN_model.h5')
