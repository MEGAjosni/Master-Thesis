# Master-Thesis
Repository for all code produced as part of my Master Thesis on Universal Physics-Informed Neural Networks (UPINNs)


## Load a trained model
The model ```model.pt``` at path ```path/to/models``` can be loaded using the following code:
```python
upinn_model.load(path='path/to/models', name='model.pt')
```

Note! This requires instantiating a ```UPINN``` model object with the same architecture as the model to be loaded. Models saved with the ```UPINN.save()``` have an associated ```model_architecture.txt``` file containing a printout of the model architecture.