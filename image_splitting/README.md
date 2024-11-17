## Image splitting directory

This directory contains the code for image splitting, also known as copy-paste augmentation. Some images have many annotations. Therefore it can be valuable to split these images into more images. So to extract the flakes from an image and paste them on a background.

### What this algorithm roughly does
- Extract mages from the training dataset that have more annotations than a certain threshold.
- Loop through all the annotations of each image.
- While looping through the annotations a slightly larger bounding box is drawn around the annotation. This bounding box is used to check for overlap with other annotations. Overlapping annotations are added and the bounding box is extended.
- When no overlapping annotations are found the latest extended bounding box is cut out from the original image and pasted into a background.


### Usage

To run this algorithm, run the following:
`python image_splitting/split.py`
With optional arguments:
- `--annotation_threshold <integer>`, default is `15`
- `--border <integer>`, default is `15`

This will create a new folder in the `data/image/` directory called `batchsplit/` and a new annotations file in the `data/annotations/` directory called `batchsplit.json`.

The `load_train_val_datasets` function from `bep.utils` automatically tries to load the `batchsplit` data for the training set. To disable this set the `load_split` argument to `False`.