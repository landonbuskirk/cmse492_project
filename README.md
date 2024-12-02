# Auto Claims Image Classification
## Landon Buskirk - Fall 2024 CMSE 492 project

Many insurance companies must manage thousands of claims each year. Furthermore, insurance fraud is a large obstacle for these companies, and so claims often have to be analyzed diligently by hand. This project aims to create a model to classify images of car damage as a common type of auto claim (dent, flat tire, broken glass, etc). Predictions can help speed up claims analysis and fraud detection by flagging mismatches or triaging claims. 


#### Project Objectives:
- effectively classify uncontrolled, client-submitted images of several auto damage types
- explore various architectures of convolutional neural networks
- create a report with a clear narrative


## How to Run the Code

### 1. Clone the Repository
First, clone this repository to your local machine:

    git clone https://github.com/your-username/your-project-name.git
    cd your-project-name

### 2. Set Up a Virtual Environment (Optional but Recommended)
Create a virtual environment to isolate the project's dependencies:

    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    venv\Scripts\activate

### 3. Install Dependencies

Install the required dependencies listed in the requirements.txt file:

    pip install -r requirements.txt

For the most part, this project uses the following python packages: sklearn, keras, numpy, pandas, matplotlib, and pillow.

### 4. Run the Code

The images under the directory data/processed/ are compressed and ready for model training, however, code that performed this preprocessing can be seen in src/data_loader.py. The src/ directory also includes train.py, which includes the functions necessary to define and compile the CNN architectures, as well as run cross-validation for a keras classification model.

CV scores calculated from the train_CV function in train.py are stored in results/cv_scores.csv. Models checkpoint are saved at results/models/, though any previously trained model files are excluded from the repo to save on storage space. 

Reports and all images used for reports can be found in the reports/ directory.

Finally, the notebooks/ directory contains all jupyter notebooks with the work of the project. EDA.ipynb includes data exploration, data preprocessing, and PCA. baseline_models.ipynb includes the fitting and evaluation of the 4 baseline models tested (MCC, MLR, MLP, shallow CNN). diagram.ipynb includes a few lines to generate visual representations of our CNN architectures. Running the main notebook, cnn_modeling.ipynb, will perform CV on several CNNs, and save CV performance metrics in the results folder, train the final model, and plot final results.


### 5. Deactivating the Virtual Environment

Once you are done, you can deactivate the virtual environment by running:

    deactivate


## Project Dependencies

In general, this project requires Python 3.12, as well as current versions of python packages sklearn, keras, numpy, pandas, matplotlib, and pillow. For more extensive package and version requirements, please refer to the requirements.txt file.