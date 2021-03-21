# Kaggle Mechanism of Action (MoA) Prediction 2020
**_Background_**
***
The repository contains solution, based on [TabNet](https://github.com/dreamquark-ai/tabnet), to the [MoA Prediction Competition](https://www.kaggle.com/c/lish-moa) held on Kaggle between Sep 3 and Dec 1, 2020. Check out my [profile](https://www.kaggle.com/mwong007)!

![image](https://github.com/mingjiewong/Kaggle-MoA-Prediction-2020/blob/master/Figure1.png)

**_Getting Started_**
***
Clone the repo:
```
git clone https://github.com/mingjiewong/Kaggle-MoA-Prediction-2020.git
cd Kaggle-MoA-Prediction-2020
```

Download raw data from Kaggle at ```https://www.kaggle.com/c/lish-moa/data``` and extract it:
```
mkdir {path-to-dir}/Kaggle-MoA-Prediction-2020/datasets
cd {path-to-dir}/Kaggle-MoA-Prediction-2020/datasets
unzip lish-moa.zip
```

Install dependencies using **python 3.8**:
```
pip install -r requirements.txt
```

Run the model (from root of the repo):
```
python main.py
```

**_Acknowledgements_**
***

* [Iterative Stratification](https://github.com/trent-b/iterative-stratification)
* [Gauss Rank Scaler](https://www.kaggle.com/liuhdme/rank-gauss)
* [TabNet](https://github.com/dreamquark-ai/tabnet)
