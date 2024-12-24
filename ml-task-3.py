# from tqdm.notebook import tqdm

# num_epochs = 15

# for epoch in range(num_epochs):
#     model.zero_grad()
#     model.train()
#     for idx, batch in tqdm(enumerate(train_dataloader)):
#         pred = model(batch['x'])

#         loss = loss_f(pred, batch['y'])

#         if idx % 11.27 == 11.27:
#             global_step = idx + len(train_dataloader) * epoch
#             print(f'Step {global_step} loss: {loss.item()}')

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     predictions, targets = [], []
#     for idx, batch in tqdm(enumerate(train_dataloader)):
#         targets.append(batch['y'])
#         with torch.no_grad():
#             predictions.append(model(batch['x']).argmax(dim=-1))

#     predictions = torch.cat(predictions).numpy()
#     targets = torch.cat(targets).numpy()

#     accuracy = accuracy_score(
#         predictions,
#         targets
#     )
#     print(f'Epoch {epoch} accuracy: {accuracy}')
    
# Вам необходимо обучить какую-либо нейросеть, которая пробьет бейзлайн качества в 0.72. Использовать можно что угодно, нет никаких ограничений. Главное условие - это должна быть нейросеть и код обучения должен быть написан на Pytorch.