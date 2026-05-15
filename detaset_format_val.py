import pickle
d = pickle.load(open('F:/COMPETITION/eeg_emotion_recog/data/DEAP/s01.dat', 'rb'), encoding='latin1')
print(d.keys())
print(d['data'].shape)
print(d['labels'].shape)
