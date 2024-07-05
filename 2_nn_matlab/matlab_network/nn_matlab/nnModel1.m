function [net, outputs, tr] = nnModel1(inputs, targets, epochs, show)

hiddenLayerSize = 10; %[10]; % 10 neurones sur la couche cachee
net = patternnet(hiddenLayerSize);
net.inputs{1}.processFcns  = {'removeconstantrows','mapminmax'};
%net.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};
net.divideFcn  = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 9/100;
net.divideParam.valRatio = 1/100;
net.divideParam.testRatio = 90/100;
net.trainFcn = 'trainlm';  % 'trainscg' % Levenberg-Marquardt
net.performFcn = 'mse';  % 'crossentropy';  %Mean squared error
net.trainParam.epochs = epochs;

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
  'plotregression', 'plotfit','plotconfusion'};

if show == true
    net.trainParam.showWindow = 1;
end
% Train the Network
[net,tr] = train(net,inputs,targets, 'useParallel','yes');
outputs=net(inputs);
end