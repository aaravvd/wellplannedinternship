import pickle

# Load the saved model
def load_model():
    with open('Model/model_1724287063.pkl', 'rb') as file:
        model = pickle.load(file)
        return model

