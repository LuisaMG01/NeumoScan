import numpy as np

IMG_SIZE = (224, 224)
CLASS_LABELS = ['NORMAL', 'BACTERIAL', 'VIRAL']
DATASET_EMBEDDINGS = np.load("./embedding/dataset_embeddings.npy")
IMAGE_PATHS = np.load("./path/image_paths.npy", allow_pickle=True)
