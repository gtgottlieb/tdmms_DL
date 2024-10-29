**Model Assisted Labeling Directory**

Contains code to use a (trained) model to make initial predictions, extract the annotations in polygon format, and upload the images with annotations to Labelbox.com.

 :exclamation: **Note that this `mal` module uses a different python environment than the rest of the project. So create a separate environment from `mal/requirements.txt`.** :exclamation:

Follow these steps to perform model assisted labeling:
 1. Create a `config.py` file in the root directory of the project that contains the following:
 ```
 LABELBOX_API_KEY  =  'your api key'
```
2. **In your AI environment** run `python mal/predict.py <dataset> --weights <weights file>`.  `<dataset>` is the directory name of the image directory in `data/images/` that contains all the images that will be pre-labeled. `<weights file>` is the filename of the weights file in the `weights/` folder that will be used for predictions.
3. Create another python environment, the Labelbox environment, with the requirements found in the `mal/requirements.txt` file.
4. **In your Labelbox environment** run `python mal/upload.py <dataset> --project_id <labelbox project id>`.  `<labelbox project id>` is the id of the Labelbox project to link the data to.

> The `<dataset>` argument must be the consistent.

The dataset and batch names in Labelbox must all be unique. So running the exact same code twice will result in errors.