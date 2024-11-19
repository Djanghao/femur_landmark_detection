1. Run `pip install -r req.txt` to install packages
2. Copy your data to `landmark_detection/dataset`, clean and extract data using `landmark_detection/dataset/femur/code/cleaning.ipynb`
3. Adjust scale distance in `landmark_detection/code/datasets/femur.py`
4. Adjust params in `landmark_detection/code/config/voting_femur.yaml`
5. For training and testing run `python -m scripts.Voting --dataset femur` in `landmark_detection/code`