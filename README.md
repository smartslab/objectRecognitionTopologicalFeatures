# Vision-Based Object Recognition in Indoor Environments using Topologically Persistent Features
Code repository for methods proposed in 'Vision-Based Object Recognition in Indoor Environments using Topologically Persistent Features' [[Pre-print]](https://arxiv.org/abs/2010.03196)

## Dataset
The UW Indoor Scenes (UW-IS) dataset used in the above paper can be found [here](https://data.mendeley.com/datasets/dxzf29ttyh/).

## Requirements
* [Tensorflow Models DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)
* Keras
* giotto-tda=0.2.2
* persim=0.1.2
* Python 3.6


## Usage
* ### Segmentation map generation: 
<p align="center">
    <img src="https://github.com/smartslab/objectRecognitionTopologicalFeatures/blob/181f88f2f8fce88cc5bf6410580394a80e461c2d/segmentationMapGeneration.png" width="840"> <br />
    <em> Pipeline of segmentation map generation module</em>
</p>



1. Install the Deeplab implementation available through Tensorflow models following the installation instructions. [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md) 
2. Under the tensorflow/models/resarch/deeplab directory create the following recommended directory structure. (The files train_uwis.sh, export_uwis.sh, and convert_uwis.sh can be found under the segMapUtils folder in this repository.)

```
+ deeplab
  - train_uwis.sh
  - export_uwis.sh
  - loadmodel_inference.py
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
4. Use segMapUtils/convertToRaw.py to convert binary segmentation maps to raw annotations (pixel value indicates class labels). Then use segMapUtils/createTrainValSets.py to generate the training and validation sets for training the DeepLabv3+ model. Run convert_uwis.sh from within the deeplab/datasets directory to convert annotations into tensorflow records for the model.
5. Place appropriate initial checkpoint available from [here](https://github.com/tensorflow/models/tree/master/research/deeplab) in the init_models folder.
6. Use train_uwis.sh followed by export_uwis.sh from within the deeplab directory to train a DeepLabv3+ model and to export the trained model, respectively.
7. Run loadmodel_inference.py from within the deeplab directory to generate segmentation maps for scene images using the trained model.
8. Use segMapUtils/cropPredsObjectWise.py to obtain cropped object images from the scene segmentation map.
9. Run loadmodel_inference.py again (using the same trained model) to generate object segmentatipn maps for all the cropped object images.

At this stage, object segmentation maps would have the following filename structure `<sceneImageName>_<cropId>_cropped.png`. Before moving to the next step, all the object segmentation maps are to labeled with appropriate category id for training the recognition networks. The steps in persistent feature extraction and recognition assume the following filename structure for object segmentation maps:

`<sceneImageName>_<cropId>_cropped_obj<category id>.png`.

* ### Persistent feature extraction and recognition:

<p align="center">
    <img src="https://github.com/smartslab/objectRecognitionTopologicalFeatures/blob/181f88f2f8fce88cc5bf6410580394a80e461c2d/recognitionUsingPersistenceFeatures.png" width="840"> <br />
    <em> Pipeline for object recognition using persistent features </em>
</p>

  All the steps below refer to code files under persistentFeaturesRecognition.
1. Generate persistence diagrams for the object segmentation maps using generatePDs.py 
2. To generate sparse PI features from the persistence diagrams run generatePIs.py followed by sparseSamplingPIs.py. Alternatively generate amplitude features using generateAmplitude.py
3. Train recognition network for sparse PI features using trainRecognitSparsePI.py. Alternatively, train recognition network for amplitude features using trainRecognitAmplitude.py
4. To test the performance of the recognition networks in the same environment that they are trained on use predictFromSparsePIs_test_trainEnv.py or predictFromAmplitude_test_trainEnv.py as appropriate.
5. To test the performance of the recognition networks in unseen environments use predictFromSparsePIs_test_testEnv.py or predictFromAmplitude_test_testEnv.py as appropriate.
