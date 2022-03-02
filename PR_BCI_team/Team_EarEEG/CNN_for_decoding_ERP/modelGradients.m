function [gradients, loss] = modelGradients(dlnet, dlX, Y)
    % Forward data through the dlnetwork object.
    dlY = forward(dlnet,dlX);

    % Compute loss.
    loss = crossentropy(dlY,Y);
    
    % Compute gradients.
    gradients = dlgradient(loss,dlnet.Learnables);
end
