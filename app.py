import streamlit as st
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

import warnings
warnings.filterwarnings("ignore")


# Charger le modèle
model = load_model('mon_modele.h5')

st.title("Classification de mélanom")

st.write("Téléchargez une image de peau pour déterminer si elle est maligne ou bénigne.")


uploaded_file = st.file_uploader("veuillez introduire votre image stp", type=["jpg","jpeg","png"])

if uploaded_file:
    st.write("image telechargé avec succées")
    img = image.load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption='Image téléchargée', use_column_width=True)

    # Prétraitement de l'image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalisation

    # Prédiction
    predictions = model.predict(img_array)
    score = predictions[0][0]

    # Affichage du résultat
    if score > 0.5:
        st.write(f"**Probabilité de mélanome (maligne)** : {score:.2f}")
        st.write("Cette image est probablement maligne.")
    else:
        st.write(f"**Probabilité de mélanome (benigne)** : {1-score:.2f}")
        st.write("Cette image est probablement bénigne.")
