
# Grammatical Error Correction using Knowledge Distillation

## Overview

This project focuses on Grammatical Error Correction (GEC) by leveraging the knowledge distillation approach. The aim is to create an efficient and accessible GEC model that can run on local devices without requiring an internet connection. We fine-tuned a smaller model (Pythia-70M) using the outputs of a larger teacher model (Pythia-1b) to achieve high performance with reduced computational requirements.

## Project Highlights

- **Knowledge Distillation**: We employed a technique where a smaller model (student) learns to replicate the performance of a larger model (teacher), enhancing efficiency while maintaining accuracy.
- **Enhanced Training Data**: Incorporation of additional Q&A datasets to improve the model's understanding and explanation of grammatical errors.
- **Local Usability**: The model is optimized to run on local devices, making it accessible in environments with limited or unreliable internet connectivity.
- **Dataset Augmentation:** Enhances the training dataset by incorporating explanations for grammatical corrections, improving the model's ability to generalize.

## Methodology

1. **Teacher-Student Model**: 
   - **Teacher Models**: Pythia-160M and Pythia-1b.
   - **Student Model**: Pythia-70M.
   - The teacher models provide outputs which the student model learns to mimic, thus transferring knowledge.

2. **Fine-Tuning**:
   - The student model is fine-tuned using datasets with explanations for grammatical errors.
   - This process helps the model understand the rationale behind corrections, improving its overall effectiveness.

3. **Datasets**:
   - **JFLEG**: Includes multiple corrected versions for each input sentence.
   - **TMU-GFM-Dataset**: Provides a substantial number of examples for training.

4. **Evaluation**:
   - Performance metrics such as ERRANT, BLEU, and ROUGE are used to evaluate the model's effectiveness.


## How to Use
Because we are using the notebook format, you can clearly see the results of our run and then reproduce the results by running each cell in turn.

1. **Clone the Repository**:.
   ```bash
   git clone https://github.com/sl-Zhou/NLP-project.git
   cd NLP-project
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Baseline and Finetuning**
   [Here](https://github.com/sl-Zhou/NLP-project/blob/main/pythia_finetuning.ipynb)

4. **Knowledge distillation with Pythia-160M as the teacher model and Pythia-70M as the student model**
   [Here](https://github.com/sl-Zhou/NLP-project/blob/main/DistilPythia_160M.ipynb)

5. **Knowledge distillation with Pythia-1B as the teacher model and Pythia-70M as the student model**
   [Here](https://github.com/sl-Zhou/NLP-project/blob/main/DistilPythia_1B.ipynb)

   You can run these two notebooks to save the model and the csv file of model's output, then check the results(BLEU, ROUGE) at the end of notebook.

6. **Knowledge distillation with Pythia-160M and explanation generated by ChatGPT4**
   [Here](https://github.com/sl-Zhou/NLP-project/blob/main/distillation_explanation.py)

   You can run this python script to fine-tune the pythia-160M model based on the **TMU-GFM-Dataset** with explanation and prompting from ChatGPT4 and get the output of the model on the testset.

7. **Evaluation metric(ERRANT)**
   [Here](https://github.com/sl-Zhou/NLP-project/blob/main/preprocess_eval.ipynb)

   You have to use this notebook to create 3 different text files for original input and output and the generated output by the model.

   Then you have to use
   [this](https://github.com/sl-Zhou/NLP-project/blob/main/eval.sh) bash script to get the results in the txt file. Make sure to adjust paths based on your files.

## Contact

For any questions or feedback, please contact:
- Seyed Armin Hosseini (seyed.armin.hosseini@umontreal.ca)
- Shilong Zhou (shilong.zhou@umontreal.ca)
- Yushun Cui (yushun.cui@umontreal.ca)
