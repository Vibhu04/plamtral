import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time, datetime
from tqdm import tqdm
import transformers
import random
import sys
from tl_lib.utils import *



def train(
    model_obj = None,
    train_loader = None,
    val_loader = None,
    actual_batch_size = 15,
    epochs = 20,
    base_lr = 0.001,
    weight_decay = 0.001,
    scheduler = None,
    warmup_steps = 5000,
    model_save_name = None,
    save_model_freq = 300,
    val_freq = 100,
    dropout = 0.1,
    write_logs = False,
    logs_folder = 'runs',
    delete_logs = True,
    load_existing_model = False,
    verbose = False
    ):
  

    if model_obj is None:
        raise Exception("Model object was not provided.")
    if train_loader is None or val_loader is None:
        raise Exception("train_loader or val_loader were not provided.")
    if model_save_name is None:
        raise Exception("Path to directory for saving model was not provided.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    base_model = model_obj.base_model
    model_size = model_obj.model_size

    if hasattr(model_obj, 'model'):
        model = model_obj.model
    else:
        model = model_obj

    if write_logs:
        writer = SummaryWriter(logs_folder)

    if delete_logs:
        # Delete all the logs
        for root, dirs, files in os.walk(logs_folder):
            for file in files:
                os.remove(os.path.join(root, file))


    if load_existing_model:
        state_dict = torch.load(model_save_name)
        model.load_state_dict(state_dict)


    model = model.to(device)

    optimizer = get_optimizer(model, model_obj, base_lr, weight_decay)
    scheduler = select_scheduler(scheduler, model_obj.technique, optimizer, len(train_loader), epochs, actual_batch_size, warmup_steps)

    model.train()
    model_obj.model = model

    seq_count = 0
    start = time.time()


    for epoch in range(epochs):

        print(f"EPOCH {epoch} started" + '=' * 30)

        for train_counter, train_batch in enumerate(train_loader, 0):

            tokens = train_batch.to(device)
            tokens = process_tokens(tokens, device, model_obj.technique)
            outputs = model(tokens, labels=tokens)
            loss = outputs[0]
            loss.backward()
            seq_count += 1

            print('[%d, %5d] train loss: %.5f' % (epoch + 1, seq_count, loss.detach().data))
            if write_logs:
                writer.add_scalar("train_loss", float(loss.detach().data), seq_count)

            # Resorting to this approach as processing input sequences in 
            # parallel would often cause 'CUDA out of memory'
            if seq_count % actual_batch_size == 0:
              optimizer.step()
              scheduler.step()
              optimizer.zero_grad()
              model.zero_grad()
              model = batch_routine(model, model_obj)

            if seq_count % save_model_freq == 0:
                torch.save(model.state_dict(), model_save_name)

            if seq_count % val_freq == 0:
                validate(model, model_obj, val_loader, device, seq_count // val_freq, writer)
                print("Time elapsed:", str(datetime.timedelta(seconds=time.time() - start)))
                model.train()

            if hasattr(model_obj, 'model'):
                model_obj.model = model

            model = modify(model, model_obj, seq_count)
            

            


def validate(model, model_obj, val_loader, device, batch_count, writer):

    model.eval()
    counter = 0
    with torch.no_grad():
        running_loss = 0

        for val_counter, val_batch in enumerate(tqdm(val_loader), 0):

            tokens = val_batch.to(device)
            tokens = process_tokens(tokens, device, model_obj.technique)
            outputs = model(tokens, labels=tokens)
            loss = outputs[0]
            counter += 1
            running_loss += loss

        print('[%d       ] validation loss: %.5f' % (batch_count,
                                                     running_loss / len(val_loader)))
        writer.add_scalar('val loss', running_loss / len(val_loader),
                          batch_count)


