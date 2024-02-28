from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.callbacks import History
from skimage.metrics import peak_signal_noise_ratio as psnr

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype('float32') / 255.
x_test  = x_test .astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test  = np.reshape(x_test,  (len(x_test),  28, 28, 1))

# Function to create deep convolutional autoencoder
def create_deep_conv():
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(128, (7, 7), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (2, 2), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x_coded = Conv2D(1, (7, 7), activation='relu', padding='same')(x)
    x_input = Input(shape=(7, 7, 1))
    x = Conv2D(32, (7, 7), activation='relu', padding='same')(x_input)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (2, 2), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x_decoded = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)
    coder = Model(input_img, x_coded, name="coder")
    decoder = Model(x_input, x_decoded, name="decoder")
    CNN = Model(input_img, decoder(coder(input_img)), name="CNN")
    return coder, decoder, CNN

# Create the deep convolutional autoencoder
c_coder, c_decoder, CNN = create_deep_conv()

# Compile the model
CNN.compile(optimizer='adam', loss='binary_crossentropy')
CNN.summary()

history = History()

# Train the autoencoder
CNN.fit(x_train, x_train,
        epochs=10,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test),
        callbacks=[history])

n = 10

imgs = x_test[:n]
coded_imgs = c_coder.predict(imgs, batch_size=n)
decoded_imgs = c_decoder.predict(coded_imgs, batch_size=n)

# Display the original, coded, and decoded images
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Display the coded images
for i in range(n):
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(coded_imgs[i].reshape(7, 7))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Display the decoded images
for i in range(n):
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# Compute and display metrics for each image
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
