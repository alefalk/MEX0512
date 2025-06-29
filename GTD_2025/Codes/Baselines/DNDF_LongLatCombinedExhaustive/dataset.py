import torch
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

class UCIAdult(Dataset):
    def __init__(self,root,train=True):
        super(UCIAdult,self).__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        # 65, Private, 198766, Masters, 14, Married-civ-spouse, Sales, Husband, White, Male, 20051, 0, 40, United-States, >50K
        self.fields = [{'type':'num'},
                       {'type':'cate','choices':['unk','Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']},
                       {'type': 'num'},
                       {'type': 'cate', 'choices': ['unk','Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']},
                       {'type': 'num'},
                       {'type': 'cate', 'choices': ['unk','Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse' ]},
                       {'type': 'cate', 'choices': ['unk','Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces' ]},
                       {'type': 'cate', 'choices': ['unk', 'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']},
                       {'type': 'cate', 'choices': ['unk', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']},
                       {'type': 'cate', 'choices': ['unk', 'Female', 'Male']},
                       {'type': 'num'},
                       {'type': 'num'},
                       {'type': 'num'},
                       {'type': 'cate', 'choices': ['unk', 'United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']},
                       ]

        for field in self.fields:
            if field['type'] == 'cate':
                field['w2i'] = dict([(w,i) for i,w in enumerate(field['choices'])])

        self.X, self.y = self.load_data(root,train)


    def __getitem__(self, index):
        return (self.X[index],self.y[index])

    def __len__(self):
        return len(self.X)


    def load_data(self,root,train):
        if train:
            data_path = os.path.join(root,"adult.data")
        else:
            data_path = os.path.join(root, "adult.test")

        with open(data_path) as f:
            rows = [ [ fv.strip() for fv in row.strip().split(",")] for row in f.readlines() if len(row.strip()) > 0 and not row.startswith("|")]
        n_datas = len(rows)
        X_dim = np.sum([ 1 if field['type']=='num' else len(field['choices']) for field in self.fields])
        X = np.zeros((n_datas, X_dim), dtype=np.float32)
        y = np.zeros(n_datas, dtype=np.int32)
        for i, row in enumerate(rows):
            assert len(row) == 15
            foffset = 0
            for j in range(14):
                if self.fields[j]['type'] == 'num':
                    fdim=1
                    X[i, foffset] = float(row[j])
                else:
                    fdim = len(self.fields[j]['choices'])
                    hit = self.fields[j]['w2i'].get(row[j],0)
                    X[i, foffset+hit] = 1
                foffset += fdim

            y[i] = 0 if row[-1].strip().startswith("<=50K") else 1

        X_min = np.array([[17.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12285.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        X_max = np.array([[90.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1484705.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 16.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 99999.0, 4356.0, 99.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        X = (X - X_min) / np.clip((X_max - X_min), a_min=1e-6, a_max=None)

        #print([float(i) for i in np.min(X, axis=0)])
        #print([float(i) for i in np.max(X, axis=0)])


        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.IntTensor)
        return X, y


class UCILetter(Dataset):
    def __init__(self,root,train=True):
        super(UCILetter,self).__init__()
        self.root = os.path.expanduser(root)
        self.train = train
        self.X, self.y = self.load_data(root,train)


    def __getitem__(self, index):
        return (self.X[index],self.y[index])

    def __len__(self):
        return len(self.X)

    def load_data(self,root,train):
        data_path = os.path.join(root, "letter-recognition.data")
        with open(data_path) as f:
            rows = [[ item.strip() for item in row.strip().split(',')] for row in f.readlines()]
        if train:
            rows = rows[:16000]
        else:
            rows = rows[16000:]

        n_datas = len(rows)
        X = np.zeros((n_datas, 16), dtype=np.float32)
        y = np.zeros(n_datas, dtype=np.int32)
        for i, row in enumerate(rows):
            X[i, :] = list(map(float, row[1:]))
            y[i] = ord(row[0]) - ord('A')

        X = torch.from_numpy(X).type(torch.FloatTensor)
        y = torch.from_numpy(y).type(torch.IntTensor)
        return X,y


class UCIgtd(Dataset):
    def __init__(self, csv_path, target_col='gname'):
        super(UCIgtd, self).__init__()

        df = pd.read_csv(csv_path)

        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_col]

        # Encode target labels to integers
        self.labels = sorted(df[target_col].unique())
        self.labels2idx = {label: i for i, label in enumerate(self.labels)}
        df['label_idx'] = df[target_col].map(self.labels2idx)

        df[feature_columns] = df[feature_columns].map(
        lambda x: int(x) if isinstance(x, bool) else x
        )

        # Save tensors
        self.X = torch.tensor(df[feature_columns].values, dtype=torch.float32)
        self.y = torch.tensor(df['label_idx'].values, dtype=torch.long)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def get_data(self):
        """Returns the full dataset as X, y tensors."""
        return self.X, self.y

    @property
    def targets(self):
        return self.y.numpy()