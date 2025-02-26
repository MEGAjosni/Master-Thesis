# Master-Thesis
Repository for all code produced as part of my Master Thesis on Universal Physics-Informed Neural Networks (UPINNs)


## Load a trained model
To load a model with name ```modelname``` at path ```path/to/models```, use the following code:
```python
UPINN.load(path='path/to/models', name='modelname')
```

Note! This requires instantiating a ```UPINN``` model object with the same architecture as the model to be loaded. Saving a model with ```UPINN.save()``` will create an associated ```modelname_architecture.txt``` file containing a printout of the model architecture.