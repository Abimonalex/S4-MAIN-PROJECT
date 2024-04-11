from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from keras.preprocessing import image as keras_image
from keras.applications.inception_v3 import preprocess_input


app = Flask(__name__)

# Load the trained AlexNet model
alexnet_model = load_model('alexnet_final_scratch.h5')
alexnet_class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                       'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                       'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                       'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                       'Tomato___healthy']

# Loading trained vgg16
vgg16_model = load_model('vgg16_final.h5')
vgg16_class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                     'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                     'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                     'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                     'Tomato___healthy']

# Load the trained GoogLeNet model
googlenet_model = load_model('model_inception.h5')
googlenet_class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                         'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                         'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                         'Tomato___healthy']

allmodel_class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                         'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                         'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
                         'Tomato___healthy']

# Define functions for AlexNet prediction
def alexnet_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(227, 227))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def alexnet_predict_disease(image_path):
    img_array = alexnet_preprocess_image(image_path)
    prediction = alexnet_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = alexnet_class_names[predicted_class_index]
    confidence = round(100 * np.max(prediction), 2)
    return predicted_class, confidence




# Define function to preprocess image for VGG16
def vgg16_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Define function to make prediction using VGG16
def vgg16_predict_disease(img_array):
    prediction = vgg16_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = vgg16_class_names[predicted_class_index]
    confidence = round(100 * np.max(prediction), 2)
    return predicted_class, confidence


# Function to preprocess and predict using GoogLeNet
def predict_tomato_disease(image_file):
    try:
        # Read the image file using PIL
        img = Image.open(image_file)
        img = img.resize((224, 224))  # Resize image to match model's expected sizing
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model's expected input shape
        img_array = preprocess_input(img_array)  # Preprocess the image

        # Make prediction using the model
        prediction = googlenet_model.predict(img_array)

        # Get the predicted class and confidence
        predicted_class = np.argmax(prediction)
        confidence = round(100 * np.max(prediction), 2)

        # Get class name
        predicted_class_name = googlenet_class_names[predicted_class]

        return predicted_class_name, confidence
    except Exception as e:
        return None, str(e)


# functions start for all model******************************************************************************************************************


# Define function to preprocess image for VGG16
def vgg16_preprocess_image_all(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

# Define function to make prediction using VGG16
def vgg16_predict_disease_all(img_array):
    img_array = vgg16_preprocess_image_all(img_array)
    prediction = vgg16_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_vggall = vgg16_class_names[predicted_class_index]
    confidence_vggall = round(100 * np.max(prediction), 2)
    return predicted_class_vggall, confidence_vggall




# Define functions for AlexNet prediction for all model
def alexnet_preprocess_image_all(image_path):
    img = image.load_img(image_path, target_size=(227, 227))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    return img_array

def alexnet_predict_disease_all(image_path):
    img_array = alexnet_preprocess_image_all(image_path)
    prediction = alexnet_model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_alnetall = allmodel_class_names[predicted_class_index]
    confidence_alnetall = round(100 * np.max(prediction), 2)
    return predicted_class_alnetall, confidence_alnetall






# Function to preprocess and predict using GoogLeNet
def predict_tomato_disease_all(image_file):
    try:
        # Read the image file using PIL
        img = Image.open(image_file)
        img = img.resize((224, 224))  # Resize image to match model's expected sizing
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model's expected input shape
        img_array = preprocess_input(img_array)  # Preprocess the image

        # Make prediction using the model
        prediction = googlenet_model.predict(img_array)

        # Get the predicted class and confidence
        predicted_class = np.argmax(prediction)
        confidence_glnetall = round(100 * np.max(prediction), 2)

        # Get class name
        predicted_class_name_glnetall = allmodel_class_names[predicted_class]

        return predicted_class_name_glnetall, confidence_glnetall
    except Exception as e:
        return None, str(e)







#function end for all model*********************************************************************************************************************
 

# Routes for home page and model selection
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/indexalex.html')
def indexalex():
    return render_template('indexalex.html')

@app.route('/indexvgg.html')
def indexvgg():
    return render_template('indexvgg.html')

@app.route('/indexgoogle.html')
def indexgoogle():
    return render_template('indexgoogle.html')


#Routing start for all model****************************************************************************************************************

@app.route('/indexall.html')
def indexall():
    return render_template('indexall.html')



#Routing end for all model****************************************************************************************************************

# Route for predicting with AlexNet
@app.route('/predict_alexnet', methods=['POST'])
def predict_alexnet():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)
        predicted_class, confidence = alexnet_predict_disease(file_path)
        if predicted_class == 'Tomato___healthy':
            return redirect(url_for('healthy', predicted_class=predicted_class, confidence=confidence))
        else:
            return redirect(url_for('diseased', predicted_class=predicted_class, confidence=confidence))

# Route for predicting with VGG16
@app.route('/predict_vgg16', methods=['POST'])
def predict_vgg16():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)
        img_array = vgg16_preprocess_image(file_path)
        predicted_class, confidence = vgg16_predict_disease(img_array)

#handling unknown img start*************************************************************************************************************************        
        threshold = 70
        if confidence <= threshold:
            return redirect(url_for('diseased', predicted_class="This image not in specified class", confidence=confidence))
#handling unknown img end*************************************************************************************************************************        

        if predicted_class == 'Tomato___healthy':
            return redirect(url_for('healthy', predicted_class=predicted_class, confidence=confidence))
        else:
            return redirect(url_for('diseased', predicted_class=predicted_class, confidence=confidence))
        


# Route for predicting with GoogLeNet
@app.route('/predict_googlenet', methods=['POST'])
def predict_googlenet():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)
        predicted_class, confidence = predict_tomato_disease(file_path)
        if predicted_class == 'Tomato___healthy':
            return redirect(url_for('healthy', predicted_class=predicted_class, confidence=confidence))
        else:
            return redirect(url_for('diseased', predicted_class=predicted_class, confidence=confidence))

#Route for prediction start for all model****************************************************************************************************************

@app.route('/predict_all', methods=['POST'])
def predict_allmodel():
    if request.method == 'POST':
        file = request.files['image']
        file_path = 'static/uploads/' + file.filename
        file.save(file_path)
        predicted_class__vggall, confidence_vggall=vgg16_predict_disease_all(file_path)
        predicted_class_alnetall,confidence_alnetall=alexnet_predict_disease_all(file_path)
        predicted_class_name_glnetall, confidence_glnetall=predict_tomato_disease_all(file_path)

        return redirect(url_for('allinone', alexnet_predicted_class=predicted_class_alnetall, alexnet_confidence=confidence_alnetall,
                                            vgg16_predicted_class=predicted_class__vggall, vgg16_confidence=confidence_vggall,
                                            googlenet_predicted_class= predicted_class_name_glnetall,googlenet_confidence=confidence_glnetall))




#Route for prediction end for all model****************************************************************************************************************



# Routes for result pages
@app.route('/healthy')
def healthy():
    predicted_class = request.args.get('predicted_class')
    confidence = request.args.get('confidence')
    return render_template('healthy.html', predicted_class=predicted_class, confidence=confidence)

@app.route('/diseased')
def diseased():
    predicted_class = request.args.get('predicted_class')
    confidence = request.args.get('confidence')
    return render_template('diseased.html', predicted_class=predicted_class, confidence=confidence)


#Route for prediction start for all model****************************************************************************************************************

@app.route('/allinone')
def allinone():
    predicted_class_alnetall = request.args.get('alexnet_predicted_class')
    confidence_alnetall = request.args.get('alexnet_confidence')
    predicted_class_vggall = request.args.get('vgg16_predicted_class')
    confidence_vggall = request.args.get('vgg16_confidence')
    predicted_class_name_glnetall = request.args.get('googlenet_predicted_class')
    confidence_glnetall = request.args.get('googlenet_confidence')
    return render_template('allinone.html',alexnet_predicted_class=predicted_class_alnetall, alexnet_confidence=confidence_alnetall,
                                            vgg16_predicted_class=predicted_class_vggall, vgg16_confidence=confidence_vggall,
                                            googlenet_predicted_class= predicted_class_name_glnetall,googlenet_confidence=confidence_glnetall)


#Route for prediction end for all model****************************************************************************************************************



if __name__ == '__main__':
    app.run(debug=True)
