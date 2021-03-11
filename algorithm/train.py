def train(model, optimizer, criterion, dataset, device, params, test_loader, transition_matrix=None, verbose=False):
    print("Training by 10-fold and repeat with 10 epoches.....")
    if verbose:
        print('==== Start Training ====')
    # Train Flag
    model.train()
    
    # Recordings
    recordings = Recordings(params.log_progress_every, params.evaluate_model_every)

    # Number of classes
    nbr_classes = len(set(dataset.tensors[1].numpy()))

    # Type casting for inverse transition matrix
    if transition_matrix is not None:
        transition_matrix = transition_matrix.float()
    
    for epoch in range(params.epochs):

        # For shuffling, regenerate the dataloaders
        t_sampler, v_sampler = get_training_validation_samplers(dataset)
        t_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=t_sampler)
        v_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=v_sampler)
        
        for iteration, (inputs, labels) in enumerate(t_loader):
            # Prepare to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Reset grad
            optimizer.zero_grad()

            # Outputs
            output_labels = model(inputs.float())

            # Loss
            if transition_matrix is not None:
                unique_labels = [c * torch.ones(labels.size()[0]).long().to(device) for c in range(nbr_classes)]
                losses = [criterion(output_labels, l) for l in unique_labels]
                losses = torch.stack(losses)
                corrected_losses = losses.transpose(0,1)@transition_matrix
                loss = corrected_losses.gather(1, labels.view(-1,1)).mean()
            else:
                loss = criterion(output_labels, labels).mean()

            # Optimize
            loss.backward()
            optimizer.step()
            
            # =======
            # Log Progress
            if ((iteration + 1) % params.log_progress_every == 0):
                recordings.progress.append(loss.data.item())
                if verbose:
                    recordings.log_progress(epoch, params.epochs, iteration, len(t_loader))
        
        # Evaluate
        if ((epoch + 1) % params.evaluate_model_every == 0):
            evaluation_loss, evaluation_accuracy = evaluate(model, test_loader, device, validation=False)
            recordings.evaluation.loss.append(evaluation_loss); recordings.evaluation.accuracy.append(evaluation_accuracy)

            if verbose:
                recordings.log_evaluation(epoch, params.epochs)

        # Save temporary model
        if ((epoch + 1) % params.save_model_every == 0):
            torch.save(model.state_dict(), params.model_filename)
    
    # Save final model
    torch.save(model.state_dict(), params.model_filename)

    # Save recordings
    serialize_object(recordings, params.recordings_filename)

    if verbose:
        print('==== End Training ====')
    
    return model, recordings
        



