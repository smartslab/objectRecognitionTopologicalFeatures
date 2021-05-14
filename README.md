# Vision-Based Object Recognition in Indoor Environments using Topologically Persistent Features
Code repository for methods proposed in 'Vision-Based Object Recognition in Indoor Environments using Topologically Persistent Features'

## Dataset

## Requirements

## Usage
* #### Segmentation map generation steps:
1. Install the Deeplab implementation available through Tensorflow models following the installation instructions [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md) 
2. Under the tensorflow/models/resarch/deeplab directory create the following recommended directory structure. (The files train_uwis.sh, export_uwis.sh, and convert_uwis.sh can be found under segMapUtils fold in this repository)

```
+ deeplab
  - train_uwis.sh
  - export_uwis.sh
  + datasets (Note: merge this folder with the pre-existing datasets folder)
      - convert_uwis.sh
      + uwis
      + init_models
      + data
        + JPEGImage_livingroom
        + foreground_livingroom
        + ImageSet_livingroom
```   
  
3. Modify the datasets/build_voc2012_data.py and datasets/data_generator.py for the UW-IS dataset.
4. Use segMapUtils/convertToRaw.py to convert binary segmentation maps to raw annotations (pixel value indicates class labels). Then use segMapUtils/createTrainValSets.py to generate the training and validation sets for training the DeepLabv3+ model. Run convert_uwis.sh to convert annotations into tensorflow records for the model
5. Place appropriate initial checkpoint available from [here](https://github.com/tensorflow/models/tree/master/research/deeplab) in the init_models folder
6. Use train_uwis.sh followed by export_uwis.sh to train a DeepLabv3+ model and to export the trained model.
7. Use segMapUtils/loadmodel_inference.py to generate segmentation maps for scene images using the trained model.
8. Use segMapUtils/cropPredsObjectWise.py to obtain cropped object images from the scene segmentation map.
9. Use segMapUtils/loadmodel_inference.py again (using the same trained model) to generate object segmentatipn maps for all the cropped object images.


* #### Persistent feature extraction and recognition steps:
