import os
os.environ['KAGGLE_USERNAME'] = "matteoturla"
os.environ['KAGGLE_KEY'] = "f9ee66d5fb9648caf98f980b3108f94d"
    
import kaggle 

if __name__ == '__main__':

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('defileroff/comic-faces-paired-synthetic-v2', path='data', unzip=True)