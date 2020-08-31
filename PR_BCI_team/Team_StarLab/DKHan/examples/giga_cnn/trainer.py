import torch
import numpy as np
import torch.nn.functional as F


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler,epochidx, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    # for epoch in range(0, start_epoch):
    #     scheduler.step()

    scheduler.step()

    # Train stage
    train_loss, tri_loss, clf_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

    message = 'Epoch: {}/{}. Train set: loss: {:.4f},tloss: {:.4f},closs: {:.4f}'\
        .format(epochidx + 1, n_epochs, train_loss,tri_loss,clf_loss)


    for metric in metrics:
        message += '\t{}: {}'.format(metric.name(), metric.value())

    val_loss, val_tri_loss , val_clf_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
    val_loss /= len(val_loader)
    val_tri_loss /= len(val_loader)
    val_clf_loss /= len(val_loader)

    message += '\nEpoch: {}/{}. Valid set: loss: {:.4f},tloss: {:.4f},closs: {:.4f}'\
        .format(epochidx + 1, n_epochs,val_loss, val_tri_loss, val_clf_loss)
    for metric in metrics:
        message += '\t{}: {}'.format(metric.name(), metric.value())

    print(message)

    return train_loss, val_loss


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss_triplet = 0
    total_loss_clf = 0
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            target = tuple(t.cuda() for t in target)

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)


        cls_target = (target[0]%2).clone().long()

        if len(outputs) == 1:
            loss = F.nll_loss(cls_output,cls_target)
            triplet_loss = 0
            clf_loss = 0

        elif len(outputs) == 5: #분류기따로 포함
            cls_output = outputs[3]
            clf_loss = F.nll_loss(cls_output, cls_target)
            loss_inputs = outputs[0:3]
            if target is not None:
                loss_inputs += target
            loss_inputs += (clf_loss,)
            loss_outputs = loss_fn(*loss_inputs)
            loss1 = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            loss2 = F.nll_loss(outputs[4], cls_target)
            loss = 0.5*loss1+0.5*loss2
        elif len(outputs) == 4: #분류로스 포함
            cls_output = outputs[3]
            clf_loss = F.nll_loss(cls_output,cls_target)
            loss_inputs = outputs[0:3]
            if target is not None:
                loss_inputs += target
            loss_inputs += (clf_loss,)
            triplet_loss,gamma = loss_fn(*loss_inputs)

            loss = (1 - gamma) * triplet_loss + gamma *clf_loss

            # loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        else: #quintuplets
            cls_output = outputs[5]
            clf_loss = F.nll_loss(cls_output, cls_target)
            loss_inputs = outputs[0:5]

            loss_inputs += (clf_loss,)
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs


        losses.append(loss.item())

        total_loss_triplet += triplet_loss.item()
        total_loss_clf += clf_loss.item()
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss_triplet /= (batch_idx + 1)
    total_loss_clf /= (batch_idx + 1)
    total_loss /= (batch_idx + 1)

    return total_loss, total_loss_triplet, total_loss_clf, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        total_loss_triplet = 0
        total_loss_clf = 0

        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                target = tuple(t.cuda() for t in target)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            cls_target = (target[0] % 2).clone().long()
            if len(outputs) == 1:
                loss = F.nll_loss(cls_output, cls_target)
                triplet_loss = 0
                clf_loss = 0
            elif len(outputs) == 5:  # 분류기따로 포함
                cls_output = outputs[3]
                clf_loss = F.nll_loss(cls_output, cls_target)
                loss_inputs = outputs[0:3]
                if target is not None:
                    loss_inputs += target
                loss_inputs += (clf_loss,)
                loss_outputs = loss_fn(*loss_inputs)
                loss1 = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                loss2 = F.nll_loss(outputs[4], cls_target)
                loss = 0.5 * loss1 + 0.5 * loss2
            elif len(outputs) == 4:  # 분류로스 포함
                cls_output = outputs[3]
                clf_loss = F.nll_loss(cls_output, cls_target)
                loss_inputs = outputs[0:3]
                if target is not None:
                    loss_inputs += target
                loss_inputs += (clf_loss,)
                triplet_loss, gamma = loss_fn(*loss_inputs)

                loss = (1 - gamma) * triplet_loss + gamma * clf_loss

            else:  # quintuplets
                cls_output = outputs[5]
                clf_loss = F.nll_loss(cls_output, cls_target)
                loss_inputs = outputs[0:5]

                loss_inputs += (clf_loss,)
                loss_outputs = loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            val_loss += loss.item()
            total_loss_triplet += triplet_loss.item()
            total_loss_clf += clf_loss.item()


            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, total_loss_triplet, total_loss_clf, metrics
