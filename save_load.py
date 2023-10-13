import pickle
def save_model(model,filename):
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Successfully saved model in models")

def load_model(model):
    return pickle.load(open(model, 'rb'))

def save_feature(feature, filename):
    with open(filename + '.pkl', 'wb') as file:
        pickle.dump(feature, file)
    print("Successfully saved features")

def load_feature(feature):
    return pickle.load(open(feature, 'rb'))