## nanodet module

The *nanodet* module contains the *NanodetLearner* class, which inherits from the abstract class *Learner*.

### Class NanodetLearner
Bases: `engine.learners.Learner`

The *NanodetLearner* class is a wrapper of the Nanodet object detection algorithms based on the original
[Nanodet implementation](https://github.com/RangiLyu/nanodet).
It can be used to perform object detection on images (inference) and train All predefined Nanodet object detection models and new modular models from the user.

The [NanodetLearner](../../src/opendr/perception/object_detection_2d/nanodet/nanodet_learner.py) class has the
following public methods:

#### `NanodetLearner` constructor
```python
NanodetLearner(self, model_to_use, iters, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device,
               weight_decay, warmup_steps, warmup_ratio, lr_schedule_T_max, lr_schedule_eta_min, grad_clip)
```

Constructor parameters:

- **model_to_use**: *{"EfficientNet_Lite0_320", "EfficientNet_Lite1_416", "EfficientNet_Lite2_512", "RepVGG_A0_416",
  "t", "g", "m", "m_416", "m_0.5x", "m_1.5x", "m_1.5x_416", "plus_m_320", "plus_m_1.5x_320", "plus_m_416",
  "plus_m_1.5x_416", "custom"}, default=m*\
  Specifies the model to use and the config file that contains all hyperparameters for training, evaluation and inference as the original
  [Nanodet implementation](https://github.com/RangiLyu/nanodet). If you want to overwrite some of the parameters you can
  put them as parameters in the learner.
- **iters**: *int, default=None*\
  Specifies the number of epochs the training should run for.
- **lr**: *float, default=None*\
  Specifies the initial learning rate to be used during training.
- **batch_size**: *int, default=None*\
  Specifies number of images to be bundled up in a batch during training.
  This heavily affects memory usage, adjust according to your system.
- **checkpoint_after_iter**: *int, default=None*\
  Specifies per how many training iterations a checkpoint should be saved.
  If it is set to 0 no checkpoints will be saved.
- **checkpoint_load_iter**: *int, default=None*\
  Specifies which checkpoint should be loaded.
  If it is set to 0, no checkpoints will be loaded.
- **temp_path**: *str, default=''*\
  Specifies a path where the algorithm looks for saving the checkpoints along with the logging files. If *''* the `cfg.save_dir` will be used instead.
- **device**: *{'cpu', 'cuda'}, default='cuda'*\
  Specifies the device to be used.
- **weight_decay**: *float, default=None*\
- **warmup_steps**: *int, default=None*\
- **warmup_ratio**: *float, default=None*\
- **lr_schedule_T_max**: *int, default=None*\
- **lr_schedule_eta_min**: *float, default=None*\
- **grad_clip**: *int, default=None*\

#### `NanodetLearner.fit`
```python
NanodetLearner.fit(self, dataset, val_dataset, logging_path, verbose, logging, seed, local_rank)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:

- **dataset**: *ExternalDataset*\
  Object that holds the training dataset.
  Can be of type `ExternalDataset`.
- **val_dataset** : *ExternalDataset, default=None*\
  Object that holds the validation dataset.
  Can be of type `ExternalDataset`.
- **logging_path** : *str, default=''*\
  Subdirectory in temp_path to save log files and TensorBoard.
- **verbose** : *bool, default=True*\
  Enables verbosity.
- **logging** : *bool, default=True*\
  Enables the maximum verbosity and the logger.
- **seed** : *int, default=123*\
  Seed for repeatability.
- **local_rank** : *int, default=1*\
  Needed if training on multiple machines.

#### `NanodetLearner.eval`
```python
NanodetLearner.eval(self, dataset, verbose, logging, local_rank)
```

This method is used to evaluate a trained model on an evaluation dataset.
Saves a txt logger file containing stats regarding evaluation.

Parameters:

- **dataset** : *ExternalDataset*\
  Object that holds the evaluation dataset.
- **verbose**: *bool, default=True*\
  Enables verbosity.
- **logging**: *bool, default=False*\
  Enables the maximum verbosity and logger.
- **local_rank** : *int, default=1*\
  Needed if evaluating on multiple machines.

#### `NanodetLearner.infer`
```python
NanodetLearner.infer(self, input, thershold)
```

This method is used to perform object detection on an image.
Returns an `engine.target.BoundingBoxList` object, which contains bounding boxes that are described by the left-top corner and
its width and height, or returns an empty list if no detections were made of the image in input.

Parameters:
- **input** : *Image*\
  Image type object to perform inference on it.
- **threshold**: *float, default=0.35*\
  Specifies the threshold for object detection inference.
  An object is detected if the confidence of the output is higher than the specified threshold.

#### `NanodetLearner.optimize`
```python
NanodetLearner.optimize(self, export_path, initial_img, verbose, optimization)
```

This method is used to perform JIT or ONNX optimizations and save a trained model with its metadata.
If a model is not present in the location specified by *export_path*, the optimizer will save it there.
If a model is already present, it will load it instead.
Inside this folder, the model is saved as *nanodet_{model_name}.pth* for JIT models or *nanodet_{model_name}.onnx* for ONNX and a metadata file *nanodet_{model_name}.json*.

Note: In ONNX optimization, the output model executes the original model's feed forward method.
The user must create their own pre- and post-processes in order to use the ONNX model in the C API.
In JIT optimization the output model performs the feed forward pass and post-processing.
To use the C API, it is recommended to use JIT optimization as shown in the [example of OpenDR's C API](../../projects/c_api/samples/object_detection/nanodet/nanodet_jit_demo.c).

Parameters:

- **export_path**: *str*\
  Path to save or load the optimized model.
- **initial_img**: *Image*, default=None\
  If optimize is called for the first time a dummy OpenDR image is needed as input.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity.
- **optimization**: *str, default="jit"*\
  It determines what kind of optimization is used, possible values are *jit* or *onnx*.

#### `NanodetLearner.save`
```python
NanodetLearner.save(self, path, verbose)
```

This method is used to save a trained model with its metadata.
Provided with the path, it creates the *path* directory, if it does not already exist.
Inside this folder, the model is saved as *nanodet_{model_name}.pth* and a metadata file *nanodet_{model_name}.json*.
If the directory already exists, the *nanodet_{model_name}.pth* and *nanodet_{model_name}.json* files are overwritten.
If optimization is performed, the optimized model is saved instead.

Parameters:

- **path**: *str, default=None*\
  Path to save the model, if None it will be the `"temp_folder"` or the `"cfg.save_dir"` from learner.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity and logger.

#### `NanodetLearner.load`
```python
NanodetLearner.load(self, path, verbose)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.
If optimization is performed, the optimized model is loaded instead.

Parameters:

- **path**: *str, default=None*\
  Path of the model to be loaded.
- **verbose**: *bool, default=True*\
  Enables the maximum verbosity.

#### `NanodetLearner.download`
```python
NanodetLearner.download(self, path, mode, model, verbose, url)
```

Downloads data needed for the various functions of the learner, e.g., pretrained models as well as test data.

Parameters:

- **path**: *str, default=None*\
  Specifies the folder where data will be downloaded. If *None*, the *self.temp_path* directory is used instead.
- **mode**: *{'pretrained', 'images', 'test_data'}, default='pretrained'*\
  If *'pretrained'*, downloads a pretrained detector model from the *model_to_use* architecture which was chosen at learner initialization.
  If *'images'*, downloads an image to perform inference on. If *'test_data'* downloads a dummy dataset for testing purposes.
- **verbose**: *bool, default=False*\
  Enables the maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.


#### Tutorials and Demos

A tutorial on performing inference is available.
Furthermore, demos on performing [training](../../projects/perception/object_detection_2d/nanodet/train_demo.py),
[evaluation](../../projects/perception/object_detection_2d/nanodet/eval_demo.py) and
[inference](../../projects/perception/object_detection_2d/nanodet/inference_demo.py) are also available.



#### Examples

* **Training example using an `ExternalDataset`**

  To train properly, the architecture weights must be downloaded in a predefined directory before fit is called, in this case the directory name is "predefined_examples".
  Default architecture is *'m'*.
  The training and evaluation dataset root should be present in the path provided, along with the annotation files.
  The default COCO 2017 training data can be found [here](https://cocodataset.org/#download) (train, val, annotations).
  All training parameters (optimizer, lr schedule, losses, model parameters etc.) can be changed in the model config file
  in [config directory](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/config).
  You can find more information in [corresponding documentation](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/config/config_file_detail.md).
  For easier usage of the NanodetLearner, the user can overwrite the following parameters:
  (iters, lr, batch_size, checkpoint_after_iter, checkpoint_load_iter, temp_path, device, weight_decay, warmup_steps,
  warmup_ratio, lr_schedule_T_max, lr_schedule_eta_min, grad_clip)

  **Note**

  The Nanodet tool can be used with any PASCAL VOC- or COCO-like dataset, by providing the correct root and dataset type.

  If *'voc'* is chosen for *dataset*, the directory must look like this:

  - root folder
    - train
      - Annotations
        - image1.xml
        - image2.xml
        - ...
      - JPEGImages
        - image1.jpg
        - image2.jpg
        - ...
    - val
      - Annotations
        - image1.xml
        - image2.xml
        - ...
      - JPEGImages
        - image1.jpg
        - image2.jpg
        - ...

  On the other hand, if *'coco'* is chosen for *dataset*, the directory must look like this:

  - root folder
    - train2017
      - image1.jpg
      - image2.jpg
      - ...
    - val2017
      - image1.jpg
      - image2.jpg
      - ...
    - annotations
      - instances_train2017.json
      - instances_val2017.json

  You can change the default annotation and image directories in [the *build_dataset* function](../../src/opendr/perception/object_detection_2d/nanodet/algorithm/nanodet/data/dataset/__init__.py).
  This example assumes the data has been downloaded and placed in the directory referenced by `data_root`.
  ```python
  from opendr.engine.datasets import ExternalDataset
  from opendr.perception.object_detection_2d import NanodetLearner


  if __name__ == '__main__':
    dataset = ExternalDataset(data_root, 'voc')
    val_dataset = ExternalDataset(data_root, 'voc')

    nanodet = NanodetLearner(model_to_use='m', iters=300, lr=5e-4, batch_size=8,
                             checkpoint_after_iter=50, checkpoint_load_iter=0,
                             device="cpu")

    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet_m", verbose=True)
    nanodet.fit(dataset, val_dataset)
    nanodet.save()

  ```

* **Inference and result drawing example on a test image**

  This example shows how to perform inference on an image and draw the resulting bounding boxes using a nanodet model that is pretrained on the COCO dataset.
  In this example, a pre-trained model is downloaded and inference is performed on an image that can be specified with the *path* parameter.

  ```python
  from opendr.perception.object_detection_2d import NanodetLearner
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import draw_bounding_boxes

  if __name__ == '__main__':
    nanodet = NanodetLearner(model_to_use='m', device="cpu")
    nanodet.download("./predefined_examples", mode="pretrained")
    nanodet.load("./predefined_examples/nanodet_m", verbose=True)
    nanodet.download("./predefined_examples", mode="images")
    img = Image.open("./predefined_examples/000000000036.jpg")
    boxes = nanodet.infer(input=img)

    draw_bounding_boxes(img.opencv(), boxes, class_names=nanodet.classes, show=True)
  ```

* **Optimization framework with Inference and result drawing example on a test image**

  This example shows how to perform optimization on a pretrained model, then run inference on an image and finally draw the resulting bounding boxes, using a nanodet model that is pretrained on the COCO dataset.
  In this example we use ONNX optimization, but JIT can also be used by changing *optimization* to *jit*.
  With the *path* parameter you can define the image file to be used as dummy input for the optimization and inference.
  The optimized model will be saved in the `./optimization_models` folder
  ```python
  from opendr.engine.data import Image
  from opendr.perception.object_detection_2d import NanodetLearner, draw_bounding_boxes


  if __name__ == '__main__':
    nanodet = NanodetLearner(model_to_use='m', device="cpu")
    nanodet.load("./predefined_examples/nanodet_m", verbose=True)

    # First read an OpenDR image from your dataset and run the optimizer:
    img = Image.open("./predefined_examples/000000000036.jpg")
    nanodet.optimize("./onnx/nanodet_m/", img, optimization="onnx")

    boxes = nanodet.infer(input=img)

    draw_bounding_boxes(img.opencv(), boxes, class_names=nanodet.classes, show=True)
  ```


#### Performance Evaluation

In terms of speed, the performance of Nanodet is summarized in the table below (in FPS).
The speed is measured from the start of the forward pass until the end of post-processing.

For PyTorch inference.

| Method              {intput} | RTX 2070 | TX2   | NX    |
|------------------------------|----------|-------|-------|
| Efficient Lite0     {320}    | 48.63    | 9.38  | 14.48 |
| Efficient Lite1     {416}    | 43.88    | 7.93  | 11.07 |
| Efficient Lite2     {512}    | 40.51    | 6.44  | 8.84  |
| RepVGG A0           {416}    | 33.4     | 9.21  | 12.3  |
| Nanodet-g           {416}    | 51.32    | 9.57  | 15.75 |
| Nanodet-m           {320}    | 48.36    | 8.56  | 14.08 |
| Nanodet-m 0.5x      {320}    | 46.94    | 7.97  | 12.84 |
| Nanodet-m 1.5x      {320}    | 47.41    | 8.8   | 13.98 |
| Nanodet-m           {416}    | 47.3     | 8.34  | 13.15 |
| Nanodet-m 1.5x      {416}    | 45.62    | 8.43  | 13.2  |
| Nanodet-plue m      {320}    | 41.9     | 7.45  | 12.01 |
| Nanodet-plue m 1.5x {320}    | 39.63    | 7.66  | 12.21 |
| Nanodet-plue m      {416}    | 40.16    | 7.24  | 11.58 |
| Nanodet-plue m 1.5x {416}    | 38.94    | 7.37  | 11.52 |

For JIT optimization inference.

| Method              {intput} | RTX 2070 | TX2   | NX    |
|------------------------------|----------|-------|-------|
| Efficient Lite0     {320}    | 69.06    | 12.94 | 17.78 |
| Efficient Lite1     {416}    | 62.94    | 9.27  | 12.94 |
| Efficient Lite2     {512}    | 65.46    | 7.46  | 10.32 |
| RepVGG A0           {416}    | 41.44    | 11.16 | 14.89 |
| Nanodet-g           {416}    | 76.3     | 12.94 | 20.52 |
| Nanodet-m           {320}    | 75.66    | 12.22 | 20.67 |
| Nanodet-m 0.5x      {320}    | 65.71    | 11.31 | 17.68 |
| Nanodet-m 1.5x      {320}    | 66.23    | 12.46 | 19.99 |
| Nanodet-m           {416}    | 79.91    | 12.08 | 19.28 |
| Nanodet-m 1.5x      {416}    | 69.44    | 12.3  | 18.6  |
| Nanodet-plue m      {320}    | 67.82    | 11.19 | 18.85 |
| Nanodet-plue m 1.5x {320}    | 64.12    | 11.57 | 18.26 |
| Nanodet-plue m      {416}    | 64.74    | 11.22 | 17.57 |
| Nanodet-plue m 1.5x {416}    | 56.77    | 10.39 | 14.81 |

For ONNX optimization inference.

In this case, the forward pass is performed in ONNX.
The pre-processing steps were implemented in PyTorch.
Results show that the performance on ONNX varies significantly among different architectures, with some achieving good performance while others performing poorly.
Additionally, it was observed that the performance of ONNX on a TX2 device was generally good, although it was observed to have occasional spikes of long run times that made it difficult to accurately measure.
Overall, the TX2 device demonstrated good performance with ONNX.

| Method              {intput} | RTX 2070  | TX2 | NX     |
|------------------------------|-----------|-----|--------|
| Efficient Lite0     {320}    | 33.12     |     | 34.03  |
| Efficient Lite1     {416}    | 16.78     |     | 17.35  |
| Efficient Lite2     {512}    | 10.35     |     | 12.14  |
| RepVGG A0           {416}    | 27.89     |     | 51.74  |
| Nanodet-g           {416}    | 103.22    |     | 87.40  |
| Nanodet-m           {320}    | 98.73     |     | 122.26 |
| Nanodet-m 0.5x      {320}    | 144.46    |     | 208.19 |
| Nanodet-m 1.5x      {320}    | 75.82     |     | 75.40  |
| Nanodet-m           {416}    | 73.09     |     | 72.78  |
| Nanodet-m 1.5x      {416}    | 51.30     |     | 51.78  |
| Nanodet-plue m      {320}    | 51.39     |     | 50.67  |
| Nanodet-plue m 1.5x {320}    | 39.65     |     | 40.62  |
| Nanodet-plue m      {416}    | 39.17     |     | 36.98  |
| Nanodet-plue m 1.5x {416}    | 28.55     |     | 27.20  |

Finally, we measure the performance on the COCO dataset, using the corresponding metrics.

| Method              {intput} | coco2017 mAP |
|------------------------------|--------------|
| Efficient Lite0     {320}    | 24.4         |
| Efficient Lite1     {416}    | 29.2         |
| Efficient Lite2     {512}    | 32.4         |
| RepVGG A0           {416}    | 25.5         |
| Nanodet-g           {416}    | 22.7         |
| Nanodet-m           {320}    | 20.2         |
| Nanodet-m 0.5x      {320}    | 13.1         |
| Nanodet-m 1.5x      {320}    | 23.1         |
| Nanodet-m           {416}    | 23.5         |
| Nanodet-m 1.5x      {416}    | 26.6         |
| Nanodet-plue m      {320}    | 27.0         |
| Nanodet-plue m 1.5x {320}    | 29.9         |
| Nanodet-plue m      {416}    | 30.3         |
| Nanodet-plue m 1.5x {416}    | 34.1         |
 