# Face Recognition with a From-Scratch KNN

A K-nearest-neighbours classifier written in NumPy that identifies 10 people from 32x32 grayscale face images, compared against scikit-learn's SVM and Gaussian Naive Bayes.

**Interactive demo:** [Run it in your browser](https://laiba-khawar-portfolio.vercel.app/work/face-knn#demo)

## How it works

All code is in `i211697_Q2.ipynb`.

1. **Data.** `dataset.csv` has 1,700 rows and 1,024 columns with no header. Each row is one 32x32 face; rows are grouped by subject, 170 per subject, 10 subjects.
2. **Preprocessing.** Every row is divided by its L2 norm.
3. **Split.** For each subject, `train_test_split(train_size=150, test_size=20, random_state=42)`, giving 1,500 training and 200 test faces.
4. **KNN from scratch.** A `KNN` class stores the training set, computes the Euclidean distance from a test face to every training face, takes the 7 nearest (`k=7`) and returns the majority label. A cosine distance function is also defined but the recorded run uses Euclidean.
5. **Baselines.** scikit-learn `SVC()` (default RBF kernel and settings) and `GaussianNB()` trained on the same split.
6. **Visualisation.** PCA to 3 components and a 3D scatter plot of the training faces coloured by subject.

## Results

Recorded output of `i211697_Q2.ipynb`, cell 2 (200 test faces):

| Classifier | Test accuracy |
| --- | --- |
| KNN from scratch (k=7, Euclidean) | 0.96 |
| SVM (scikit-learn defaults) | 1.0 |
| Gaussian Naive Bayes | 0.85 |

These come from a single split with `random_state=42`; no cross-validation was run.

## Repository contents

- `i211697_Q2.ipynb`: the notebook (3 code cells: imports, functions, run).
- `dataset.csv`: face data (1,700 x 1,024).
- `DM-Ass03.pdf`: assignment document.
- `requirements.txt`: jupyter, matplotlib, numpy, pandas, scikit-learn.

## Running it

The notebook was last run with Python 3.10.

```bash
git clone https://github.com/LaibaKhawar/FaceRecognition_KNN.git
cd FaceRecognition_KNN
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook i211697_Q2.ipynb
```

Before running, fix the data path in cell 2. The notebook reads `'Q2 Dataset\dataset.csv'`, but in this repository `dataset.csv` sits at the repository root. Change the line to:

```python
filepath = 'dataset.csv'
```

The KNN predicts with Python loops over all 1,500 training faces for each test face, so it takes a few seconds rather than milliseconds.

## Known limitations

- The data path in the notebook does not match the repository layout (see above), so the notebook fails as committed.
- Results are from one train/test split and 200 test images, so small differences between classifiers are not statistically meaningful.
- Distances are computed one pair at a time in Python; the implementation is written for clarity, not speed.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laiba-k-00b2b1249/)
