# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# # --------------------------------------------------
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import accuracy_score
# from tqdm.notebook import tqdm
# # --------------------------------------------------

# RANDOM_SEED = 0

# feature_columns = [
#     'age', 'annual_income', 'monthly_inhand_salary', 'total_emi_per_month', 'num_bank_accounts',
#     'num_credit_card', 'interest_rate', 'num_of_loan', 'delay_from_due_date', 'num_of_delayed_payment',
#     'changed_credit_limit', 'num_credit_inquiries', 'outstanding_debt', 'credit_utilization_ratio',
#     'credit_history_age', 'amount_invested_monthly', 'monthly_balance'
    
# ]

# target_column = 'credit_score'

# dt = pd.read_csv('/kaggle/input/credit-score-classification-cleaned-dataset/credit_score_cleaned_train.csv')
# dt = dt[feature_columns + [target_column]]
# dt.head()

# train_dt, test_dt = train_test_split(dt, random_state=RANDOM_SEED, test_size=0.1)

# # Classic ML Baseline

# from lightgbm import LGBMClassifier
# from sklearn.metrics import accuracy_score

# model = LGBMClassifier(verbose=0)
# X, y = train_dt[feature_columns].values, train_dt[target_column].values
# model.fit(X, y)

# classic_ml_accuracy_baseline = accuracy_score(
#     model.predict(test_dt[feature_columns].values),
#     test_dt[target_column].values
# )

# print(f'Classic ML accuracy baseline: {classic_ml_accuracy_baseline}')

# # Classic ML accuracy baseline: 0.7217166494312306

# # Deep learning seminar

# # Для начала необходимо написать датасет обьект

# from torch.utils.data import Dataset, DataLoader

# class MyDataset(Dataset):
#     def __init__(self, data: np.ndarray, targets: np.ndarray):
#         assert len(data) == len(targets)
#         self.data = data.astype(np.float32)
#         self.targets = targets.astype(np.int64)

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, idx: int):
#         x = self.data[idx]
#         y = self.targets[idx]
#         return {
#             'x': x,
#             'y': y
#         }

# train_dataset = MyDataset(data=train_dt[feature_columns].values, targets=train_dt[target_column].values)
# test_dataset = MyDataset(data=test_dt[feature_columns].values, targets=test_dt[target_column].values)

# len(train_dataset), test_dataset[0]

# # Делаем даталоадеры

# train_dataloader = DataLoader(
#     dataset=train_dataset, num_workers=0, shuffle=True, batch_size=512,
# )
# test_dataloader = DataLoader(
#     dataset=test_dataset, num_workers=0, shuffle=False, batch_size=512,
# )

# next(iter(train_dataloader))

# # Теперь строим модель, для примера это будет пару слоев и функции усложнения

# import torch
# import torch.nn as nn

# num_classes = len(dt[target_column].unique())

# model = nn.Sequential(
#     nn.BatchNorm1d(num_features=len(feature_columns)),
#     nn.Linear(in_features=len(feature_columns), out_features=2 * len(feature_columns)),
#     nn.ReLU(),
#     nn.Linear(in_features=2 * len(feature_columns), out_features=len(feature_columns)),
#     nn.ReLU(),
#     nn.Linear(in_features=len(feature_columns), out_features=num_classes),
# )

# model(next(iter(train_dataloader))['x'])

# # Теперь давайте поймем как будем учить

# loss_f = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-1)

# batch = next(iter(train_dataloader))

# loss_f(
#     model(batch['x']),
#     batch['y']
# )

# # Давайте немного поучим нашу модель. Вам необходимо обучить какую-либо нейросеть, которая пробьет бейзлайн качества в 0.72. Использовать можно что угодно, нет никаких ограничений. Главное условие - это должна быть нейросеть и код обучения должен быть написан на Pytorch.


# num_epochs = 100

# for epoch in range(epochs):
#     model.zero_grad()
#     model.train()
#     for idx, batch in tqdm(enumerate(train_dataloader)):
#         optimizer.zero_grad()
#         loss = loss_f(
#             model(batch['x']),
#             batch['y']
#         )
#         loss.backward()
#         optimizer.step()
        
#         print(f'Epoch: {epoch}, Loss: {loss.item()}')
        
#         # Выводим примеры предсказаний
#         if idx % 10 == 0:
#             global_step = idx + len(train_dataloader) * epoch
#             print(f'Step {global_step} loss: {loss.item()}')
            
#             with torch.no_grad():
#                 for example in batch['x'][:5]:
#                     prediction = model(example.unsqueeze(0)).argmax()
#                     print(f'Example: {example.tolist()}, Prediction: {prediction.item()}')
#                 print()
                
#             # Выводим примеры предсказаний после обучения на всех данных
#             if epoch % 5 == 0:
#                 with torch.no_grad():
#                     for example in batch['x']:
#                         prediction = model(example.unsqueeze(0)).argmax()
#                         print(f'Example: {example.tolist()}, Prediction: {prediction.item()}')
#                     print()
                
#     model.eval()
#     predictions, targets = [], []
#     for batch in test_dataloader:
#         with torch.no_grad():
#             predictions.append(model(batch['x']).argmax(dim=-1))
#             targets.append(batch['y'])
            
#             accuracy = accuracy_score(
#                 torch.cat(predictions).numpy(),
#                 torch.cat(targets).numpy()
#             )
#             print(f'Epoch {epoch} accuracy: {accuracy}')
            
#     predictions = torch.cat(predictions).numpy()
#     targets = torch.cat(targets).numpy()
    
#     accuracy = accuracy_score(
#         predictions,
#         targets
#     )
#     print(f'Epoch {epoch} accuracy: {accuracy}')