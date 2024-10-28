"""Module to load datarows with pre-labels into Labelbox.com"""

import os
import sys
import uuid

import labelbox as lb

ROOT_DIR = os.path.abspath("../../")
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

import config

API_KEY = config.LABELBOX_API_KEY
client = lb.Client(API_KEY)

# dataset = client.get_dataset("<dataset_id>")
project = client.get_project('cm167pqz802tq07023jfr2abh')

data_path = os.path.join(ROOT_DIR, 'data', 'images', 'batch1', '19_sio2_NbSe2_Cleanroom_f4_img.png')
global_key = '19_sio2_NbSe2_Cleanroom_f4_img.png'

annotations = [{"name": "Thick", "polygon": [{"x": 1325.173, "y": 1046.883}, {"x": 1332.173, "y": 1067.883}, {"x": 1226.173, "y": 1048.883}, {"x": 1219.173, "y": 1039.383}, {"x": 1221.673, "y": 1034.883}, {"x": 1325.173, "y": 1046.883}]}]

# Create a dataset in Labelbox
# Each dataset is a batch from the data/ folder
def upload_dataset():
    test_img_url = {
        "row_data": data_path,
        "global_key": global_key
    }

    dataset = client.create_dataset(name="test-dataset")
    task = dataset.create_data_rows([test_img_url])
    task.wait_till_done()

    print(f"Failed data rows: {task.failed_data_rows}")
    print(f"Errors: {task.errors}")

    if task.errors:
        for error in task.errors:
            if 'Duplicate global key' in error['message'] and dataset.row_count == 0:
                # If the global key already  exists in the workspace the dataset will be created empty, so we can delete it.
                print(f"Deleting empty dataset: {dataset}")
                dataset.delete()

# Add batch to labeling project
def add_batch_to_project():
    batch = project.create_batch(
        "batch-test",  # each batch in a project must have a unique name
        global_keys=[global_key], # paginated collection of data row objects, list of data row ids or global keys
        priority=5  # priority between 1(highest) - 5(lowest)
    )

    print(f"Batch: {batch}")

# Convert annotations to labels attached to datarows
def link_annotations_to_datarows():
    label_ndjson = []

    for annotation in annotations:
        annotation.update({
            "dataRow": {
                "globalKey": global_key
            },
        })
        label_ndjson.append(annotation)

    return label_ndjson
# Upload annotations
def upload_annotations():
    upload_job = lb.MALPredictionImport.create_from_objects(
        client = client, 
        project_id = project.uid, 
        name="mal_job"+str(uuid.uuid4()), 
        predictions=link_annotations_to_datarows()
    )

    print(f"Errors: {upload_job.errors}")
    print(f"Status of uploads: {upload_job.statuses}")

if __name__ == '__main__':
    # upload_dataset()
    # add_batch_to_project()
    upload_annotations()
