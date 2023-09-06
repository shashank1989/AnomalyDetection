import pickle

def Model_pickle(model,model_name):
    # Define the path to save the .pkl file
    pkl_filename = model_name + '.pkl'

    # Serialize and save the model to a .pkl file
    with open(pkl_filename, 'wb') as pkl_file:
        pickle.dump(model, pkl_file)

    print(f"Model saved as {pkl_filename}")