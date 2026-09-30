import datetime
import os

import torch
import torch.utils.data


# define training loop
def training_loop(
    n_epochs, optimizer, model, loss_fn, train_loader, validation_loader, device, model_name,
    out_dir=".", checkpoint_every=50,
):
    os.makedirs(os.path.join(out_dir, "models", "backup"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "trainingProtocol"), exist_ok=True)
    training_protocol = (
        "Start Training for "
        + str(n_epochs)
        + " Epochs ("
        + str(datetime.datetime.now(datetime.UTC))
        + ")\n"
    )
    print(training_protocol)
    for epoch in range(0, n_epochs):
        loss_train = 0.0
        model.train()
        for input, label in train_loader:
            input_gpu = input.to(device=device)
            label_gpu = label.to(device=device)
            output = model(input_gpu)
            loss = loss_fn(output, label_gpu)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        loss_val = 0.0
        model.eval()
        with torch.no_grad():
            for val_input, val_label in validation_loader:
                val_input = val_input.to(device)
                val_label = val_label.to(device)

                predictions = model(val_input )
                loss_val += loss_fn(predictions,val_label).item()


        epoch_info = str(
            datetime.datetime.now(datetime.UTC).time()
        ) + " Epoch: %d, train_loss: %f, val_loss: %f" % (epoch + 1, loss_train / len(train_loader), loss_val/ len(validation_loader))
        training_protocol += epoch_info + "\n"
        print(epoch_info)
        if (epoch+1) % checkpoint_every == 0:
            ckpt_path = os.path.join(out_dir, "models", "backup", model_name + "_backup_"+str(epoch+1)+".pt")
            torch.save(model.state_dict(), ckpt_path)
            print("Model saved at epoch ", epoch+1, "->", ckpt_path)

    final_path = os.path.join(out_dir, "models", model_name + ".pt")
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print("Final model saved ->", final_path)

    with open(os.path.join(out_dir, "trainingProtocol", "TrainingProtocol_" + model_name + ".txt"), "a") as file:
        file.write(training_protocol)
