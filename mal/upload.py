"""Module to load datarows with pre-labels into Labelbox.com"""

import os
import sys
import uuid
import json
import labelbox as lb

ROOT_DIR = os.path.abspath(os.path.join(__file__, '../../../'))
print('Root directory:',ROOT_DIR)
sys.path.append(ROOT_DIR)

import config

API_KEY = config.LABELBOX_API_KEY
client = lb.Client(API_KEY)

data_pre_fix = 'mal_test_'

# Create a dataset in Labelbox
# Each dataset is a batch from the data/ folder
def upload_dataset(data: str):
    dataset_name = data_pre_fix + data

    print(f"Uploading '{data}' data under '{dataset_name}'..")

    data_rows = []
    data_dir = os.path.join(ROOT_DIR, 'data', 'images', data)
    images = os.listdir(data_dir)
    for i in images:
        data_rows.append({
            "row_data": os.path.join(data_dir, i),
            "global_key": data_pre_fix+i
        })

    dataset = client.create_dataset(name=dataset_name)
    task = dataset.create_data_rows(data_rows)
    task.wait_till_done()

    print(f"Failed data rows: {task.failed_data_rows}")
    print(f"Errors: {task.errors}")

    if task.errors:
        for error in task.errors:
            if 'Duplicate global key' in error['message'] and dataset.row_count == 0:
                # If the global key already  exists in the workspace the dataset will be created empty, so we can delete it.
                print(f"Deleting empty dataset: {dataset}")
                dataset.delete()

    return data_rows

def link_dataset_to_project(data: str, project_name: str, data_rows: list):
    print(f"Linking '{data}' to '{project_name}' project..")
    if project_name == 'test':
        project_id = config.LABELBOX_TEST_ANNOTATIONS_ID
    elif project_name == 'NbSe2':
        project_id = config.LABELBOX_NBSE2_SIO2_ID

    project = client.get_project(project_id)
    dataset_name = data_pre_fix + data

    global_keys = [i['global_key'] for i in data_rows]

    add_batch_to_project(project, dataset_name, global_keys)
    upload_annotations(project, global_keys)

def add_batch_to_project(project, batch, global_keys):
    batch = project.create_batch(
        batch,  # each batch in a project must have a unique name
        global_keys=global_keys, # paginated collection of data row objects, list of data row ids or global keys
        priority=5  # priority between 1(highest) - 5(lowest)
    )

    print(f"Batch: {batch}")

def link_annotations_to_datarows(global_keys):
    print('Adding global keys to data rows..')
    label_ndjson = []

    data_dir = os.path.join(ROOT_DIR, 'mal', data)

    for global_key in global_keys:
        file_name = global_key.replace(data_pre_fix, '').split('.')[0] + '.json' 

        with open(os.path.join(data_dir, file_name)) as f:
            annotations = json.load(f)

        for annotation in annotations:
            annotation.update({
                "dataRow": {
                    "globalKey": global_key
                },
            })
            label_ndjson.append(annotation)

    return label_ndjson

def upload_annotations(project, global_keys):
    print('Uploading annotations..')
    upload_job = lb.MALPredictionImport.create_from_objects(
        client = client, 
        project_id = project.uid, 
        name="mal_job"+str(uuid.uuid4()), 
        predictions=link_annotations_to_datarows(global_keys)
    )

    print(f"Errors: {upload_job.errors}")
    print(f"Status of uploads: {upload_job.statuses}")

if __name__ == '__main__':
    data = 'batch4'

    data_rows = upload_dataset(data)
    link_dataset_to_project(data, 'test', data_rows)
