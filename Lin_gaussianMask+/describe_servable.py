"""
A template for creating a DLHub-compatible description of a model.

The places where you need to fill in information are marked with "TODO" comments
"""

from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from dlhub_sdk.utils.schemas import validate_against_dlhub_schema
from dlhub_sdk.models.servables.pytorch import TorchModel
import json
import pickle as pkl

# Read in model from disk
model_info = TorchModel.create_model('gaussianMask+.pth', (1, 1, None, None), (1, 1, None, None))

# Define the name and title for the model
model_info.set_title("Lin gaussianMask+ model")
model_info.set_name("Lin_gaussianMask+")

# Verify authors and affiliations
model_info.set_authors(["Lin, Ruoqian"], ["Chemistry Division-BNL"])

# Describe the scientific purpose of the model
model_info.set_domains(["general", "materials science", "microscopy"])
model_info.set_abstract("A model for atom localization in atomically-resolved STEM images. An encoder–decoder type U-net architectured CNN network is used.")

# Add references for the model
model_info.add_related_identifier("10.1038/s41598-021-84499-w", "DOI", "IsDescribedBy")  # Example: Paper describing the model
#model_info.add_related_identifier("https://github.com/xinhuolin/AtomSegNet", "Github", "IsDocumentedBy")  # Example: Github documenting the model

# Describe the computational environment
model_info.add_requirement('numpy', 'detect')
model_info.add_requirement('scipy', 'detect')
model_info.add_requirement('PIL', 'detect')
model_info.add_requirement('skimage', 'detect')
model_info.add_requirement('torch', 'detect')

# Describe the inputs and outputs of the model
model_info.set_inputs('ndarray', 'data of a gray image of any size', shape=([None, None]))
model_info.set_outputs('ndarray', 'the probability map of being an atom with the same shape as input', shape=([None, None]))

# Add the file describing the net of torch model
model_info.add_file('unet_sigmoid.py')
# changing the descriptions for the inputs and outputs from their default values
#model_info['servable']['methods']['run']['output']['description'] = 'Response'

# Check the schema against a DLHub Schema
validate_against_dlhub_schema(model_info.to_dict(), 'servable')

# Save the metadata
with open('dlhub.json', 'w') as fp:
    json.dump(model_info.to_dict(save_class_data=True), fp, indent=2)
