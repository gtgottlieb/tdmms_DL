"""Module that contains functions."""

import json
import os
import random
import shutil

data_types = ['images', 'annotations']
data_sets = ['train', 'val']

def check_dir_setup(ROOT_DIR, train_size):    
    for dt in data_types:
        for ds in data_sets:
            extension = ''
            if dt == data_types[1]:
                extension = '.ndjson'
            
            path = os.path.join(ROOT_DIR, 'data', dt, ds+extension)
            if not os.path.exists(path):
                print(f'{path} did not exist')
                create_dir_setup(ROOT_DIR, train_size)
                return None

def create_dir_setup(ROOT_DIR, train_size):
    print('Creating directories from batches..')

    batches = [i for i in os.listdir(os.path.join(ROOT_DIR, 'data', 'images')) if 'batch' in  i]
    print('Found batches:',','.join(batches))
    
    reset_dirs(ROOT_DIR)
    data_split_images(batches, ROOT_DIR, train_size)
    data_split_annotations(batches, ROOT_DIR)
    
def reset_dirs(ROOT_DIR):
    for dt in data_types:
        for ds in data_sets:
            extension = ''
            if dt == data_types[1]:
                extension = '.ndjson'
                
            path = os.path.join(ROOT_DIR, 'data', dt, ds+extension)
            if os.path.exists(path):
                if extension == '':                    
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                        os.mkdir(os.path.join(ROOT_DIR, 'data', 'images', ds))
                    except OSError as e:
                        print('Error in deleting and creating directory:',e)
                else:
                    try:
                        shutil.rmtree(path, ignore_errors=True)
                    except OSError as e:
                        print('Error in deleting file:',e)
                        
def data_split_images(batches, ROOT_DIR, train_size):
    imgs_batches = []
    
    for batch in batches:
        imgs_batches.append((batch, os.listdir(os.path.join(ROOT_DIR, 'data', 'images', batch))))    
    imgs_batches = [[(i[0], j) for j in i[1]] for i in imgs_batches]
    imgs_batches = [i for j in imgs_batches for i in j]
    
    random.shuffle(imgs_batches)
    
    img_count = len(imgs_batches)    
    print(f'Total image count: {img_count}')
    
    train_amount = round(train_size*img_count)
    
    print('Copying images..')
    for i in imgs_batches[:train_amount]:
        shutil.copy(
            os.path.join(ROOT_DIR, 'data', 'images', i[0], i[1]),
            os.path.join(ROOT_DIR, 'data', 'images', 'train', i[1])
        )
    
    for i in imgs_batches[train_amount:]:
        shutil.copy(
            os.path.join(ROOT_DIR, 'data', 'images', i[0], i[1]),
            os.path.join(ROOT_DIR, 'data', 'images', 'val', i[1])
        )

    return None

def data_split_annotations(batches, ROOT_DIR):
    rows = []
    for batch in batches:
        with open(os.path.join(ROOT_DIR, 'data', 'annotations', batch+'.ndjson')) as f:
            rows += [json.loads(l) for l in f.readlines()]
    
    train_imgs = os.listdir(os.path.join(ROOT_DIR, 'data', 'images', 'train'))
    val_imgs = os.listdir(os.path.join(ROOT_DIR, 'data', 'images', 'val'))
    
    print('Writing annotation files..')
    with open(os.path.join(ROOT_DIR, 'data', 'annotations', 'train.ndjson'), "w+") as f:
        for row in rows:
            if row['data_row']['external_id'] in train_imgs:
                f.write(str(row)+'\n')
                
    with open(os.path.join(ROOT_DIR, 'data', 'annotations', 'val.ndjson'), "w+") as f:
        for row in rows:
            if row['data_row']['external_id'] in val_imgs:
                f.write(str(row)+'\n')