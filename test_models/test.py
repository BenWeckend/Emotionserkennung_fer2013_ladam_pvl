from tensorflow.keras.models import load_model


loaded_model = load_model("model.h5")

prediction = loaded_model.predict()
