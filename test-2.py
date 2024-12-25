
epochs = 10

for epoch in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        loss = loss_f(
            model(batch['x']),
            batch['y']
        )
        loss.backward()
        optimizer.step()
        
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        
        # Выводим примеры предсказаний
        if epoch % 5 == 0:
            with torch.no_grad():
                for example in batch['x'][:5]:
                    prediction = model(example.unsqueeze(0)).argmax()
                    print(f'Example: {example.tolist()}, Prediction: {prediction.item()}')
                print()
                
        # Сохраняем модель
        torch.save(model.state_dict(), 'model.pth')
        
        # Выводим примеры предсказаний после сохранения
        if epoch % 5 == 0:
            with torch.no_grad():
                for example in batch['x'][:5]:
                    prediction = model(example.unsqueeze(0)).argmax()
                    print(f'Example: {example.tolist()}, Prediction: {prediction.item()}')
                print()
        
        # Выводим примеры предсказаний после загрузки
        if epoch % 5 == 0:
            model.load_state_dict(torch.load('model.pth'))
            with torch.no_grad():
                for example in batch['x'][:5]:
                    prediction = model(example.unsqueeze(0)).argmax()
                    print(f'Example: {example.tolist()}, Prediction: {prediction.item()}')
                print()
        
        # Выводим примеры предсказаний после обучения на новых данных
        if epoch % 5 == 0:
            new_examples = torch.tensor([[0.2, 0.1, 0.3], [-0.3, 0.4, 0.2], [-0.1, 0.5, 0.4]])
            with torch.no_grad():
                predictions = model(new_examples).argmax(dim=1)
                for example, prediction in zip(new_examples.tolist(), predictions.tolist()):
                    print(f'Example: {example}, Prediction: {prediction}')
                print()
        
    # Выводим примеры предсказаний после обучения на всех данных
    if epoch % 5 == 0:
        with torch.no_grad():
            for example in batch['x']:
                prediction = model(example.unsqueeze(0)).argmax()
                print(f'Example: {example.tolist()}, Prediction: {prediction.item()}')
            print()