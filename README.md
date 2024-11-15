# ankle-fractures-llm
Classification model for ankle fractures with LLM-generated labels.

Classifier for x-ray images of the ankle. Images are classified with an accuracy of 89.5% as fracture or non-fracture. Training data were generated by interpreting radiology reports using the open-source Large Language Model Mixtral 8x7b. 
Not intented for diagnostic uses.

The LLM_classifier.py file shows the data acquisition process used to interpret radiology reports using the Large Language Model Mixtral 8x7b. Given a system_instruction as well as a specific shape for the answer, the model analyses the radiology report and gives an appropriate answer. If the answer does not satisfy a preset condition (being either 0 or 1) the Model gets its answer as well as an instruction to stick to the required answer shape. This function is recursive until it eventually gets the proper answer shape.

train_model.py file includes a Trainer class for both training and testing of a CNN for ankle fracture classification. The Trainer initialiser requires a saving_directory, which has to include a training and validation dataset for training or a test dataset and a pretrained model for testing. The datasets should be called "train_dataframe.csv", "val_dataframe.csv" and "test_dataframe.csv", which should be comma seperated values with two columns called "image" (=filepathes of the images) and "label" (0 or 1 for non-fracture and fracture respectively). 
A pretrained model is available for download on zenodo.org under https://zenodo.org/records/11495761

For demonstration purposes we created a gradio app, which can be freely accessed via: XXXXXXXXXXXXXXX. X-ray images can be uploaded and classified by the model. Please only upload images, to which you have the rights to upload to the website server. Not intented for diagnostic use.
<img width="960" alt="empty" src="https://github.com/FaresAlMohamad/ankle-fractures-llm/assets/141377568/98ca0bba-0356-4eb2-a914-1aedc229d28e">
<img width="960" alt="fracture" src="https://github.com/FaresAlMohamad/ankle-fractures-llm/assets/141377568/72ebfa50-eece-4a02-a3c6-390796dc9645">
<img width="960" alt="no_fracture" src="https://github.com/FaresAlMohamad/ankle-fractures-llm/assets/141377568/f460f8e0-9eb7-4e7e-9ac1-36e6f966bb91">

