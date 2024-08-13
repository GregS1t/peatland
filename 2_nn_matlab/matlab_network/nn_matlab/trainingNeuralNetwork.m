%% Training Neural Network

clear
clc

%% Define Model

prefix = 'n50w80';

disp(['Tile used for training : ' prefix])

usingModel = 'nnModel1';
processShow = true;

%% PATH

dateTimeStr = datestr(datetime,'ddmmmyyyy_HHMMSS');

% output
fileNet = ['/home/gsainton/CALER/PEATMAP/1_NN_training/training_data/trained_' prefix '_Network.mat'];
archivedData = ['archives/' usingModel '_' dateTimeStr '.mat'];

% input
fileAllTrainData = ['/home/gsainton/CALER/PEATMAP/1_NN_training/training_data/trainingData_' prefix '.mat']

%% Training

load(fileAllTrainData, 'input', 'target');

if strcmp(usingModel, 'nnModel1')
    [net, outputs, tr] = nnModel1(input', target', 50, processShow);

else
    error('Model does not exist')

end

%% Save Trained Network

save(fileNet, 'net', 'tr');
fprintf('SAVED %s\n', fileNet); 

copyfile(fileNet, archivedData);
fprintf('SAVED %s\n', archivedData);

