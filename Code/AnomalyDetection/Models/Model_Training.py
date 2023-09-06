def Model_Training(num_epochs, dataloader, model,optimizer,criterion):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            batch = tuple(t.to('cuda') for t in batch)  # Move batch to GPU
            input_ids_batch, attention_mask_batch, y_batch = batch

            optimizer.zero_grad()
            outputs = model(input_ids_batch, attention_mask_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training complete!")

    return model


def LSTM_Model_Training(num_epochs, train_loader, model, optimizer, criterion, device):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training complete!")
    
    return model

