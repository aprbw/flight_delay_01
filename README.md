# Predicting Flight Delay With Spatio-Temporal Trajectory Convolutional Network and Airport Situational Awareness Map

<p align="center">
  <img src=./trajcnn_arch.PNG>
</p>

```{python}
from Flight_Delay_experiment_5 import experiment
experiment()
```

* **FAA_DataPointLoader_Image_Only_2.py**
This is the dataloader for the TrajCNN features.

* **FAA_DataPointLoader_no_image_1.py**
This is the dataloader for all the other features (schedule and weather).

* **Flight_Delay_experiment_5.py**
This the main file. It contains all the configurations, the default hyperparamters, as well as the model.

* **ProgressIndicator.py**
This is a utility class for progress indicator in a loop.

* **hp.py**
This is for hyper-paramter search.

Note: the data are not yet provided.
