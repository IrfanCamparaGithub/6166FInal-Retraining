# SMIRK Hugging Face Implementation

This is the hugging face implementation of the paper and project included within this repo: [https://github.com/georgeretsi/smirk?tab=readme-ov-file](https://github.com/georgeretsi/smirk?tab=readme-ov-file).

# Instructions to Host On Hugging Face:
Download this github repo as a zip  
download the wheel and pretrained model from the github releases attached  
Download the Flame 2020 model here: [https://flame.is.tue.mpg.de/index.html](https://flame.is.tue.mpg.de/index.html)  
Create a new hugging face space  
Place the .pkl files within assets under the Flame2020 folder  
Place the pretrained model under a pretrained_models folder and place the wheel within a third_party folder within the root  
Ensure you are using a GPU  
To upload/remove images/videos, enter the samples folder at the root and choose the files you want to add/delete and it will re run to show the 3d mesh  
A Working version can be found here: [https://huggingface.co/spaces/icampara/6166FinalWithRetrain?logs=container](https://huggingface.co/spaces/icampara/6166FinalWithRetrain?logs=container)  

To retrain and evaluate the model and see our version of this, pls go to the releases and go to the tar.gz link.  
Download the file which should take you to a google drive file.  
1. Launch ubunutu within your machine (from now on the powershell and naming will be as the my local machine)  
2.  
```bash
wsl -d Ubuntu-22.04
```  
3. Restore the conda enviroment:  
```bash
TAR="/mnt/c/Users/Irfan/Documents/GitHub/smirk6166Retrained/smirk118_env_backup.tar.gz"
mkdir -p ~/miniconda/envs
cd ~/miniconda/envs
tar -xzf "$TAR"
source ~/miniconda/etc/profile.d/conda.sh
conda activate smirk118
```  
4. Copy the project code:  
```bash
cp -r ~/smirk-work/smirk ~/smirk-retrained
cd ~/smirk-retrained
```  
5. Verify that all needed files are there:  
```bash
test -f assets/FLAME2020/generic_model.pkl && echo "FLAME model: OK" || echo "FLAME model: MISSING"
test -f assets/head_template.obj && echo "Head OBJ: OK" || echo "Head OBJ: MISSING"
test -f trained_models/SMIRK_em1.pt && echo "Retrained checkpoint: OK" || echo "Retrained checkpoint: MISSING"
test -d data_prepared/eval/images && echo "Eval images dir: OK" || echo "Eval images dir: MISSING"
```  
6. Run the evaluation:  
```bash
export PYTHONPATH=$(pwd)
python tools/eval_rendered.py --batch 8 --max_n 800
```  
7.You should see losses as were present within the image below  
<img width="188" height="32" alt="image" src="https://github.com/user-attachments/assets/a6612b62-da26-440f-a5c6-b4294784fd4d" />
