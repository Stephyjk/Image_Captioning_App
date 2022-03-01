# Import necessary libraries
from pickle import load
from numpy import argmax
from flask import Flask, render_template, request
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
import os

# Create flask instance
app = Flask(__name__)

# Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def init():
#     global graph
#     graph = tf.get_default_graph()

# Function to load and prepare the image in right shape


# def read_image(filename):
#     # Load the image
#     img = load_img(filename, color_mode="grayscale", target_size=(28, 28))
#     # Convert the image to array
#     img = img_to_array(img)
#     # Reshape the image into a sample of 1 channel
#     img = img.reshape(1, 28, 28, 1)
#     # Prepare it as pixel data
#     img = img.astype('float32')
#     img = img / 255.0
#     # print(img)
#     return img


@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

# generate a description for an image

# map an integer to a word


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text


def extract_features(filename):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # load the photo
    image = load_img(filename, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # print(image)
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        try:
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join('static/images', filename)
                file.save(file_path)
                # print(file_path)

                # load the tokenizer
                tokenizer = load(open('tokenizer.pkl', 'rb'))
                # pre-define the max sequence length (from training)
                max_length = 34
                # load the model
                model = load_model('model_19.h5')
                # print(model)
                # # load and prepare the photograph
                photo = extract_features(file_path)
                # print(photo)
                # generate description
                description = generate_desc(
                    model, tokenizer, photo, max_length)
                print(description)

                # Remove startseq and endseq
                query = description
                stopwords = ['startseq', 'endseq']
                querywords = query.split()

                resultwords = [
                    word for word in querywords if word.lower() not in stopwords]
                result = ' '.join(resultwords)
                return render_template('predict.html', result=result, user_image=file_path)
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')


if __name__ == "__main__":
    # init()
    app.run()
