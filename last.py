import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential  # Specific imports
import tensorflow.keras.models as keras_models  # Import entire module
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
import numpy as np
from tensorflow.keras import Input

#model loading

model1 = tf.keras.models.load_model('skin.h5')
model2 = tf.keras.models.load_model('braincase.h5')
# model2.batch_size = [1, 128, 128, 3] 
# #batch_input_shape: [1, 128, 128, 3]

image_path = "tumorimg.jpg"



def preprocess_image(image_path, target_size=(170, 170)):
 
  # Load the image with resizing to the target size
  img = load_img(image_path, target_size=target_size)

  # Convert the PIL image to a NumPy array
  img_array = img_to_array(img)

  # Normalize pixel values to the range [0, 1]
  img_array = img_array / 255.0

  # Add a batch dimension (None allows flexibility for batch size)
  img_array = np.expand_dims(img_array, axis=0)

  return img_array


preprocessed_image = preprocess_image(image_path)

# #--------------------------------------------------------------------
while True:
  model_choice = input("Enter 1 for skin  or 2 for brain  Identification: ")
  if model_choice in ('1', '2'):
    break
  else:
    print("Invalid choice. Please enter 1 or 2.")




# Make prediction based on the chosen model & docotor reccomendation


if model_choice == '1':
  predictions = model1.predict(preprocessed_image)
  predicted_class = predictions[0] > 0.5  # assuming the first element represents cancer probability
  if predicted_class:
    print("Skin Model Prediction: Cancer")
    see_doctor = input("Would you like to see a doctor (yes/no)? ").lower()
    if see_doctor == 'yes':
        doctors = {
            "skin_cancer": ["Dr. Hiro Nakamura","Dr. Li Wei","Dr. Kim Sun-woo","Dr. Priya Desai","Dr. Nguyen Minh Anh","Dr. Aishah Hassan","Dr. Benigno Santos","Dr. Pradeep Singh","Dr. Rinchen Dorji","Dr. Anand Patel"]

        }
    for doctor in doctors["skin_cancer"]:
            print(doctor)
  else:
    print("Skin Model Prediction: No Cancer")

#------------------------------------------------------------------------------

elif model_choice == '2':
  predictions = model2.predict(preprocessed_image)
  predicted_class = predictions[0] > 0.5  # assuming the first element represents cancer probability
  if predicted_class:
    print("Brain Model Prediction: Cancer")
    see_doctor = input("Would you like to see a doctor (yes/no)? ").lower()
    if see_doctor == 'yes':
      print("Recommended Brain Cancer Doctors:")
      doctors = {
            "brain_cancer": ["Dr. Li Wei", "Dr. Akiko Sato", "Dr. Kim Sun-Young", "Dr. Nguyen Minh Hien", 
                "Dr. Priya Sharma", "Dr. Aisha Khan", "Dr. Benita Prasad", "Dr. Angeline Santos",
                "Dr. Chen Lin", "Dr. Ahmad Yusof"]
        }
    for doctor in doctors["brain_cancer"]:
            print(doctor)
  else:
    print("Brain Model Prediction: No Cancer")




