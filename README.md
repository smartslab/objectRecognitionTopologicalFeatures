# Vision-Based Object Recognition in Indoor Environments using Topologically Persistent Features
Code repository for methods proposed in 'Vision-Based Object Recognition in Indoor Environments using Topologically Persistent Features'. [[Pre-print]](https://arxiv.org/abs/2010.03196)

## Dataset
The UW Indoor Scenes (UW-IS) dataset used in the above paper can be found [here](https://data.mendeley.com/datasets/dxzf29ttyh/). The dataset consists of scenes from two different environments, namely, a living room and a mock warehouse. The scenes are captured using varying camera poses under different illumination conditions and include up to five different objects from a given set of fourteen objects.

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



1. Install the DeepLabv3+ implementation available through Tensorflow models following the installation instructions [here.](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md) 
2. Under the `tensorflow/models/resarch/deeplab` directory create the following recommended directory structure for training a DeepLabv3+ model in the living room environment of the UW-IS dataset. (The files `train_uwis.sh`, `export_uwis.sh`, and `convert_uwis.sh` can be found under the `segMapUtils` folder in this repository.)

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
        + JPEGImages_livingroom
        + foreground_livingroom
        + ImageSets_livingroom
```   
  
3. Place living room scene images to be used for training the DeepLabv3+ model under `JPEGImages_livingroom` and corresponding ground truth segmentation maps under `foreground_livingroom`.
4. Modify the existing files `datasets/build_voc2012_data.py` and `datasets/data_generator.py` appropriately for the UW-IS dataset.
5. Use `segMapUtils/convertToRaw.py` to convert binary segmentation maps to raw annotations (pixel value indicates class labels). Then use `segMapUtils/createTrainValSets.py` to generate the training and validation sets for training the DeepLabv3+ model. Run `convert_uwis.sh` from within the `deeplab/datasets` directory to convert annotations into tensorflow records for the model.
6. Place appropriate initial checkpoint available from [here](https://github.com/tensorflow/models/tree/master/research/deeplab) in the `init_models` folder.
7. Use `train_uwis.sh` followed by `export_uwis.sh` from within the `deeplab` directory to train a DeepLabv3+ model and to export the trained model, respectively.
8. Run `loadmodel_inference.py` from within the `deeplab` directory to generate segmentation maps for scene images using the trained model.
9. Use `segMapUtils/cropPredsObjectWise.py` to obtain cropped object images from the scene segmentation map.
10. Run `loadmodel_inference.py` again (using the same trained model) to generate object segmentatipn maps for all the cropped object images.

At this stage, object segmentation maps would have the following filename structure `<sceneImageName>_<cropId>_cropped.png`. Before moving to the next step, all the object segmentation maps are to be labeled with appropriate category id for training the recognition networks. The steps in persistent features extraction and recognition assume the following filename structure for object segmentation maps:

`<sceneImageName>_<cropId>_cropped_obj<category id>.png`.

* ### Persistent feature extraction and recognition:

<p align="center">
    <img src="https://github.com/smartslab/objectRecognitionTopologicalFeatures/blob/181f88f2f8fce88cc5bf6410580394a80e461c2d/recognitionUsingPersistenceFeatures.png" width="840"> <br />
    <em> Pipeline for object recognition using sparse PI features </em>
</p>

  All the steps below refer to code files under the `persistentFeatRecognit` folder in this repository.
1. Generate persistence diagrams for the object segmentation maps using `generatePDs.py`
2. To generate sparse PI features from the persistence diagrams, run `generatePIs.py` to obtain persistence images (PIs) followed by `sparseSamplingPIs.py`. The script `sparseSamplingPIs.py` generates optimal pixel locations for the PIs that can be used to obtain sparse PIs. To generate amplitude features, use `generateAmplitude.py`
3. Use `trainRecognitSparsePI.py` to train a recognition network using sparse PI features. The file loads generated PIs and obtains sparse PIs using the optimal pixel locations generated in the previous step. To train a recognition network using amplitude features, use `trainRecognitAmplitude.py`.
4. To test the performance of the recognition networks in the same environment that they are trained on (i.e., living room in the default case), use `predictFromSparsePIs_test_trainEnv.py` or `predictFromAmplitude_test_trainEnv.py` as appropriate.
5. To test the performance of the recognition networks in unseen environments (i.e., mock warehouse in the default case), generate object segmentation maps from the warehouse images as described above. Then, obtain persitence diagrams from the object segmentation maps. From the persistence diagrams generate PIs and amplitude for the object segmentation maps. 
      1. To test the sparse PI recognition network's performance use `predictFromSparsePIs_test_testEnv.py`. It uses the same optimal pixel locations obtained at the time of training to obtain sparse PIs, and makes predictions using the trained model.
      2. To test the amplitude recognition network's performance use `predictFromAmplitude_test_testEnv.py `.
