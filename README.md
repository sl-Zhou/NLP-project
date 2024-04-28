# IFT6758 Project
[Repo URL](https://github.com/sl-Zhou/NLP-project)

## Authors
- [@Shilong Zhou](https://github.com/sl-Zhou)
- [@Seyed Armin Hosseini](https://github.com/Arminhosseini)
- [@Yushun Cui](https://github.com/loongtop)


## Prerequisites
[requirements](https://github.com/sl-Zhou/NLP-project/blob/main/requirements.txt)

Of course the python environment is essential!

## Run the code
Because we are using the notebook format, you can clearly see the results of our run and then reproduce the results by running each cell in turn.


### Baseline and Finetuning
[Here](https://github.com/sl-Zhou/NLP-project/blob/main/pythia_finetuning.ipynb)


### Knowledge distillation with Pythia-160M as the teacher model and Pythia-70M as the student model
[Here](https://github.com/sl-Zhou/NLP-project/blob/main/DistilPythia_160M.ipynb)

### Knowledge distillation with Pythia-1B as the teacher model and Pythia-70M as the student model
[Here](https://github.com/sl-Zhou/NLP-project/blob/main/DistilPythia_1B.ipynb)

You can run these two notebooks to save the model and the csv file of model's output, then check the results(BLEU, ROUGE) at the end of notebook. 
### Evaluation metric(ERRANT)
[Here](https://github.com/sl-Zhou/NLP-project/blob/main/preprocess_eval.ipynb)

You can use the csv file from above notebooks in this notebook to see the results of ERRANT.
