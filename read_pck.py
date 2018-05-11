import pickle

if __name__ == '__main__':
    with open('song/all_songs', 'rb') as fin:
        a = pickle.load(fin)
        pass