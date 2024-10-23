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

The images under the directory data/processed/ are compressed and ready for model training, however, code that performed this preprocessing can be seen in src/data_loader.py


Running the main script will train several CNNs, tune hyper parameters, and save performance metrics in the results folder.

    python src/main.py


### 5. Deactivating the Virtual Environment

Once you are done, you can deactivate the virtual environment by running:

    deactivate


## Project Dependencies

In general, this project requires Python 3.12, as well as current versions of python packages sklearn, keras, numpy, pandas, matplotlib, and pillow. For more extensive package and version requirements, please refer to the requirements.txt file.