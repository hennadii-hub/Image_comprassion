from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from skimage.color import rgb2gray  
from sklearn.model_selection import train_test_split
from keras.callbacks import History
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.metrics import peak_signal_noise_ratio as psnr

# Define the directory containing the images
data_dir = 'data2'

# Get the list of image files
image_files = [file for file in os.listdir(data_dir) if file.endswith('.jpg')]

# Load and format the images with color conversion
images = [rgb2gray(img_to_array(load_img(os.path.join(data_dir, file), target_size=(256, 256, 3)))) for file in image_files]
images = np.array(images)

# Normalize pixel values to the range [0, 1]
images = images.astype('float32') / 255.0

# Split into training and test sets
x_train, x_test = train_test_split(images, test_size=0.3, random_state=42)

# Reshape images if necessary to match expected format
x_train = np.reshape(x_train, (len(x_train), 256, 256, 1))  # Single channel for grayscale
x_test = np.reshape(x_test, (len(x_test), 256, 256, 1))

def create_deep_dense_ae():
    encoding_dim = 49
    
    # Encoder
    input_img = Input(shape=(256, 256, 1))  
    x = Flatten()(input_img)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='linear')(x)

    # Decoder
    input_encoded = Input(shape=(encoding_dim,))
    x = Dense(128, activation='relu')(input_encoded)
    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    flat_decoded = Dense(256*256, activation='sigmoid')(x)
    decoded = Reshape((256, 256, 1))(flat_decoded)

    # Models
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    
    return encoder, decoder, autoencoder

def plot_digits(*args):
    args = [x.squeeze() for x in args]
    n = min([x.shape[0] for x in args])
    
    plt.figure(figsize=(2*n, 2*len(args)))
    for j in range(n):
        for i in range(len(args)):
            ax = plt.subplot(len(args), n, i*n + j + 1)
            plt.imshow(args[i][j])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()

# Create and compile the deep dense autoencoder
d_encoder, d_decoder, d_autoencoder = create_deep_dense_ae()
d_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = History()

# Train the autoencoder
d_autoencoder.fit(x_train, x_train,
                  epochs=50,
                  batch_size=10,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  callbacks=[history])

n = 10

# Display summary of the autoencoder
d_autoencoder.summary()

imgs = x_test[:n]
encoded_imgs = d_encoder.predict(imgs, batch_size=n)
decoded_imgs = d_decoder.predict(encoded_imgs, batch_size=n)

plot_digits(imgs, decoded_imgs)

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