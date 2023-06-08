from keras.models import Sequential
from keras.models import model_from_json

def load_model(filename='model'):
    model = Sequential()

    # START LOADING MODEL
    # Loading model and weight from saved data
    # - load model
    json_file = open("model/" + filename + ".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # - load weights
    model.load_weights("model/" + filename + ".h5")
    print("Loaded model from disk")
    # END LOADING MODEL

    model_loaded = model
    return model_loaded

model = load_model()
test = "influencer ini adalah influencer yang beres amanah jujur"
print (test)
print (predict_probability(model, [tfidf_data.transform(test)]))