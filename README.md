# face_attributes_mxnet

### Five face attributes predication for a single face image.

Gender (0:woman, 1:man)  

Mask (0:no, 1:yes)

EyeGlasses (0:no, 1:EyeGlasses, 2:SunGlasses)

MouthOpen (0:no, 1:yes, 2:indeterminacy)),

EyesClose (0:no, 1:yes, 2:indeterminacy))

### Train

This project is implemented by mxnet and use the custom dataset.

`tools/prepare_dataset.py` is used to convert the pascal VOC format annotation to mxnet  lst file. Subsequently use `mxnet/tools/im2rec.py` create rec dataset.

### test

`python unit_test/fa_test.py`

### Result

2019-08-21 14:17:01,214 INFO "base_module.py" line:565: Epoch[132] Train-Gender=0.999282
2019-08-21 14:17:01,214 INFO "base_module.py" line:565: Epoch[132] Train-Mask=0.998361
2019-08-21 14:17:01,214 INFO "base_module.py" line:565: Epoch[132] Train-Glass=0.994949
2019-08-21 14:17:01,214 INFO "base_module.py" line:565: Epoch[132] Train-MouthOpen=0.997059
2019-08-21 14:17:01,214 INFO "base_module.py" line:565: Epoch[132] Train-EyesOpen=0.998002

2019-08-21 14:17:06,274 INFO "base_module.py" line:585: Epoch[132] Validation-Gender=0.863076
2019-08-21 14:17:06,274 INFO "base_module.py" line:585: Epoch[132] Validation-Mask=0.982936
2019-08-21 14:17:06,274 INFO "base_module.py" line:585: Epoch[132] Validation-Glass=0.954564
2019-08-21 14:17:06,274 INFO "base_module.py" line:585: Epoch[132] Validation-MouthOpen=0.841488
2019-08-21 14:17:06,274 INFO "base_module.py" line:585: Epoch[132] Validation-EyesOpen=0.902961