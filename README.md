# face_attributes_mxnet

### MobileNet for Five face attributes predication, the model no more than 70K.

Gender (0:woman, 1:man)  

Mask (0:no, 1:yes)

EyeGlasses (0:no, 1:EyeGlasses, 2:SunGlasses)

MouthOpen (0:no, 1:yes, 2:indeterminacy)),

EyesClose (0:no, 1:yes, 2:indeterminacy))

### Train

This project is implemented by mxnet and use the custom dataset.

`tools/prepare_dataset.py` is used to convert the pascal VOC format annotation to mxnet  lst file. Subsequently use `mxnet/tools/im2rec.py` create rec dataset.

run `python3 train/fa_train.py` to start training.

### test

`python demo/fa_test.py`

### Result

INFO "base_module.py" line:565: Epoch[299] Train-Gender=0.900570

INFO "base_module.py" line:565: Epoch[299] Train-Mask=0.985610

INFO "base_module.py" line:565: Epoch[299] Train-Glass=0.963070

INFO "base_module.py" line:565: Epoch[299] Train-MouthOpen=0.875516

INFO "base_module.py" line:565: Epoch[299] Train-EyesOpen=0.909168



INFO "base_module.py" line:585: Epoch[299] Validation-Gender=0.892064

INFO "base_module.py" line:585: Epoch[299] Validation-Mask=0.982730

INFO "base_module.py" line:585: Epoch[299] Validation-Glass=0.958676

INFO "base_module.py" line:585: Epoch[299] Validation-MouthOpen=0.870477

INFO "base_module.py" line:585: Epoch[299] Validation-EyesOpen=0.904400
