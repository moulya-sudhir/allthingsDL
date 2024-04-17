import torch
from tqdm.auto import tqdm

def train_step(model, loader, loss_fn, optimizer, device):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc += (y_pred == y).sum().item() / len(y_pred)
    train_loss /= len(loader)
    train_acc /= len(loader)

    return train_loss, train_acc

def test_step(model, loader, loss_fn, device):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            y_logits = model(X)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            test_acc += (y_pred == y).sum().item() / len(y_pred)
        test_loss /= len(loader)
        test_acc /= len(loader)

    return test_loss, test_acc

def train(model, train_loader, test_loader, loss_fn, optimizer, device, epochs = 5):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device=device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device=device)
        print(f'Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}')
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    results = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'test_loss': test_losses,
        'test_acc': test_accs
    }
    return results