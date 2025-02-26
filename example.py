import os
import pandas as pd

data_dir = 'classification_proyect/code/images'

for img in os.listdir(data_dir):
    images = [os.path.join(data_dir, img)]
    labels = [1 if 'disease' in img else 0 for img in images]
    df = pd.DataFrame({'image_path': images, 'label': labels})
    print(df.head())