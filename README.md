This is the hugging face implementation of the paper and project included within this repo: https://github.com/georgeretsi/smirk?tab=readme-ov-file.

#Instructions to Host On Hugging Face:
Download this github repo as a zip
download the wheel and pretrained model from the github releases attached
Download the Flame 2020 model here: https://flame.is.tue.mpg.de/index.html
Create a new hugging face space
Place the .pkl files within assets under the Flame2020 folder
Place the pretrained model under a pretrained_models folder and place the wheel within a third_party folder within the root
Ensure you are using a GPU
To upload/remove images/videos, enter the samples folder at the root and choose the files you want to add/delete and it will re run to show the 3d mesh
A Working version can be found here: https://huggingface.co/spaces/icampara/6166FinalWithRetrain?logs=container 
