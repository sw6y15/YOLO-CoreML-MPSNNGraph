# Conversion script for tiny-yolo-voc to Core ML.
# Needs Python 2.7 and Keras 1.2.2

import coremltools

coreml_model = coremltools.converters.keras.convert(
    'yad2k/model_data/tiny-yolo-voc.h5',
    input_names='data',
    image_input_names='data',
    output_names='model_outputs0',
    image_scale=1/255.)

coreml_model.author = 'Original paper: Joseph Redmon, Ali Farhadi'
coreml_model.license = 'Public Domain'
coreml_model.short_description = "The Tiny YOLO network from the paper 'YOLO9000: Better, Faster, Stronger' (2016), arXiv:1612.08242"

coreml_model.input_description['data'] = 'Input image'
coreml_model.output_description['model_outputs0'] = 'The 13x13 grid with the bounding box data'

print(coreml_model)

coreml_model.save('../TinyYOLO-CoreML/TinyYOLO-CoreML/TinyYOLO.mlmodel')
