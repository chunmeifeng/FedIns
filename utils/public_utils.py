import torch


def communication(server_model, models, client_weights, client_num):
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            if 'ssf_s' in key or 'Key' in key or 'fc_weight_pool' in key or 'fc_bias_pool' in key:
                temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])



def para_frozen(server_model, frozen_mode):
    if frozen_mode == -1:
        # frozen all
        for name, param in server_model.named_parameters():
            param.requires_grad = False
    elif frozen_mode == 1:
        # ssf_pool
        for name, param in server_model.named_parameters():
            if 'ssf_s' in name or 'Key' in name or 'fc_weight_pool' in name or 'fc_bias_pool' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    


def pool_train(queryModel, model, data_loader, optimizer, loss_fun, device, topk, training_loss_weight):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        # iterate over each instance in the batch
        for i in range(data.size(0)):
            optimizer.zero_grad()
            x = data[i:i+1].to(device)
            t = target[i:i+1].to(device)
            query = (queryModel.forward_features(x))[:, 0]
            output, reduced_sim = model(x, query, topk)
            loss = (1-training_loss_weight)*loss_fun(output, t) + training_loss_weight*(1-reduced_sim)
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            total += t.size(0)
            pred = output.data.max(1)[1]
            correct += pred.eq(t.view(-1)).sum().item()
    return loss_all/total, correct/total



def pool_test(queryModel, model, data_loader, loss_fun, device, topk, training_loss_weight):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            # instance-level evaluation
            for i in range(data.size(0)):
                x = data[i:i+1].to(device)
                t = target[i:i+1].to(device)
                query = (queryModel.forward_features(x))[:, 0]
                output, reduced_sim = model(x, query, topk)
                loss = (1-training_loss_weight)*loss_fun(output, t) + training_loss_weight*(1-reduced_sim)
                loss_all += loss.item()
                total += t.size(0)
                pred = output.data.max(1)[1]
                correct += pred.eq(t.view(-1)).sum().item()
    return loss_all/total, correct/total
