import torch
import torch.nn as nn


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)


def train_model(model, train_loader, val_loader,
                num_epochs=150, patience=10,
                lr=1e-3, weight_decay=1e-3,
                model_name="model",
                device=None):
    """
    Train a model and save the best weights based on validation accuracy.
    train_loader/val_loader should return: (X, y, filename)
    """
    if device is None:
        device = get_device()
    
    # Loss Function, gets output and y_batch(real labels)
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer with L2 regularization (weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Move model to CPU/GPU
    model = model.to(device)

    best_val_acc = 0.0
    epochs_no_improve = 0
    best_path = f"best_{model_name}.pt"

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

     # Main training loop over epochs 
    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        model.train() # enable dropout/batchnorm in training mode
        running_loss, correct, total = 0.0, 0, 0
        # Iterate over mini-batches from train_loader
        for batch in train_loader:
            X_batch, y_batch = batch[0], batch[1]   # ignore filename if exists
            # Move batch to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            # Reset gradients from previous step
            optimizer.zero_grad()
            # Forward pass: MFCC->layers->model prediction
            outputs = model(X_batch)
            # Compute loss for this batch
            loss = criterion(outputs, y_batch)
            # Backward pass: compute gradients
            loss.backward()
            # Update weights
            optimizer.step()

            # Loss and accuracy statistics
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1) # class with max score
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

        # Average train loss and accuracy over all training samples
        train_loss = running_loss / total
        train_acc  = correct / total

        # --- Validation ---
        model.eval() # evaluation mode (no dropout, batchnorm in eval mode)

        val_running_loss, val_correct, val_total = 0.0, 0, 0

        # Disable gradient computation during validation
        with torch.no_grad():
            for batch in val_loader:
                X_batch, y_batch = batch[0], batch[1]   # ignore filename if exists
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                # Forward
                outputs = model(X_batch)
                # Compute validation loss
                loss = criterion(outputs, y_batch)
                # Compute validation accuracy
                val_running_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (preds == y_batch).sum().item()

        # Average validation loss & accuracy        
        val_loss = val_running_loss / val_total
        val_acc  = val_correct / val_total

        # Store history for plotting
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            print("  → New best model saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    history = {
        "train_loss": train_loss_hist,
        "val_loss":   val_loss_hist,
        "train_acc":  train_acc_hist,
        "val_acc":    val_acc_hist,
        "best_path":  best_path,
        "best_val_acc": best_val_acc
    }
    return history

def load_model(model_class, num_classes, best_path, device=None):
    """Create model, load state_dict, return in eval mode."""
    if device is None:
        device = get_device()
    model = model_class(num_classes).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    return model


def evaluate_loader(model, loader, device=None):
    """Return (loss, acc) on a DataLoader. loader returns (X, y, filename)."""
    if device is None:
        device = get_device()
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total, correct, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for X_batch, y_batch, _fn_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            total += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

    return running_loss / total, correct / total
