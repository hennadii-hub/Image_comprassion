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

# Function to plot images
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

# Function to create deep dense autoencoder
def create_deep_dense_ae():
    # Dimension of the encoded representation
    encoding_dim = 49

    # Encoder
    input_img = Input(shape=(28, 28, 1))
    flat_img = Flatten()(input_img)
    x = Dense(encoding_dim*3, activation='relu')(flat_img)
    x = Dense(encoding_dim*2, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='linear')(x)
    
    # Decoder
    input_encoded = Input(shape=(encoding_dim,))
    x = Dense(encoding_dim*2, activation='relu')(input_encoded)
    x = Dense(encoding_dim*3, activation='relu')(x)
    flat_decoded = Dense(28*28, activation='sigmoid')(x)
    decoded = Reshape((28, 28, 1))(flat_decoded)
    
    # Models
    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder

# Create the deep dense autoencoder
d_encoder, d_decoder, d_autoencoder = create_deep_dense_ae()
d_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

history = History()

# Train the autoencoder
d_autoencoder.fit(x_train, x_train,
                  epochs=10,
                  batch_size=256,
                  shuffle=True,
                  validation_data=(x_test, x_test),
                  callbacks=[history])

n = 10

imgs = x_test[:n]
encoded_imgs = d_encoder.predict(imgs, batch_size=n)
encoded_imgs[0]

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