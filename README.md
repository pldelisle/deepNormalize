#  <img src="/icons/chip.png" width="60" vertical-align="bottom"> Adversarial normalization network for multi-task segmentation
> This project aims to develop a convolutionnal neural network (CNN) that automatically and intelligently normalize 
3D medical images for maximizing segmentation while preserving the medical plausibility of the image all along
the segmentation pipeline.


## Using

`python deepNormalize_main.py --data-dir=/path/to/tfrecords/folder/ --job-dir=./logs/`

List of arguments of this script :

* --data-dir: String. The directory where the deepNormalize input data is stored
	
* --job-dir: String. The directory where the model will be stored.

* --variable-strategy: choices=['CPU', 'GPU'];  Where to locate variable operations. Default to CPU.
		
* --num-gpus: Integer. The number of GPUs used. Uses only CPU if set to 0.

* --gpu-id: String. The GPU IDs on which to run the training.

* --sync: Boolean. If present when running in a distributed environment will run on sync mode.

* --num-intra-threads: Integer. Number of threads to use for intra-op parallelism. 
      When training on CPU set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.

* --num-inter-threads: Integer. Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.

* --data-format: choices=['channel_first', 'channel_last'] If not set, the data format best for the training device is used. 
      Allowed values: channels_first (NCHW) for GPU usage, channels_last (NHWC) for CPU usage.

*- -log-device-placement: Boolean. Whether to log device placement.


## Contributing

#### How to contribute ?
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

#### Branch naming

##### Feature branch
> feature/ [Short feature description] [Issue number]

##### Bug branch
> fix/ [Short fix description] [Issue number]

#### Commits syntax:

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]

##### Merging code:
> Y Merged [Short Description] [Issue Number]


Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>