import os
import cv2
import numpy as np


def prepare_test_data(data_directory_path):
    
    file_directory = os.listdir(data_directory_path)
    print('step 1: get cats and dogs images and label them')
    for file in file_directory:
        if file == 'cats':
            cats_directory = os.path.join(data_directory_path,file)
            print('processing cats images ...')
            cats_imgs = os.listdir(cats_directory)
            cats_imgs.remove('.DS_Store')
            cats_imgs.remove(cats_imgs[-1])
            cats_train_arrays = [cv2.resize(cv2.imread(os.path.join(cats_directory,cat),1),(128,128)) for cat in cats_imgs]
            cats_train_labels = np.zeros(shape = len(cats_train_arrays),dtype = 'float32')
            print('cats processing finished')
            
        elif file == 'dogs':
            dogs_directory = os.path.join(data_directory_path,file)
            print('processing dogs images ...')
            dogs_imgs = os.listdir(dogs_directory)
            dogs_imgs.remove('.DS_Store')
            dogs_imgs.remove(dogs_imgs[-1])
            dogs_train_arrays = [cv2.resize(cv2.imread(os.path.join(dogs_directory,dog),1),(128,128)) for dog in dogs_imgs]
            dogs_train_labels = np.ones(shape = len(dogs_train_arrays),dtype = 'float32')
            print('dogs processing finished')
    print('step 1: Done!')
    
    
    print('step 2: grouping dataset')
    
    merged_train_dogs_cats = np.concatenate((cats_train_arrays,dogs_train_arrays))
    merged_labels_dogs_cats = np.concatenate((cats_train_labels,dogs_train_labels))
    permutations = np.arange(merged_labels_dogs_cats.shape[0])
    np.random.shuffle(permutations)
    np.random.shuffle(permutations)
    np.random.shuffle(permutations)
    np.random.shuffle(permutations)
    merged_train_dogs_cats = merged_train_dogs_cats[permutations]
    merged_train_dogs_cats = ((merged_train_dogs_cats-255)/255).astype('float32')
    merged_labels_dogs_cats = merged_labels_dogs_cats[permutations]
    merged_labels_dogs_cats = merged_labels_dogs_cats.reshape(-1, 1)
    print('step 2: Done!')
    


    
    return merged_train_dogs_cats,merged_labels_dogs_cats
    

    
    