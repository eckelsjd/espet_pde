%% Train dlnetwork
% Joshua Eckels
% 3/2/22
clear all;
close all;
clc;
file = fullfile('../data/base/train/train_base.mat');
load(file);
%% Load data
data_dir = '../data/feasible/';
file = fullfile(data_dir,'train','train_dffnet_max.mat');
load(file);

% Pre-process
[xn, xs] = mapminmax(xdata, -1, 1);
[yn, ys] = mapminmax(ydata, -1, 1);

% Split train/val/test sets
Ndata = size(xn,2);
Ntrain = floor(0.8*Ndata);
Nval = floor(0.1*Ndata);
Ntest = Ndata - Ntrain - Nval;

idx = randperm(Ndata);
idxTrain = idx(1:Ntrain);
idxVal = idx(Ntrain+1:Ntrain+Nval);
idxTest = idx(Ntrain+Nval+1:end);

xTrain = xn(:,idxTrain); yTrain = yn(:,idxTrain);
xVal = xn(:,idxVal);     yVal = yn(:,idxVal);
xTest = xn(:,idxTest);   yTest = yn(:,idxTest);

%% Set up network
layers = [
    featureInputLayer(4,"Name","featureinput")
%     fullyConnectedLayer(18,"Name","fc_5")
%     tanhLayer("Name","tanh_5")
%     fullyConnectedLayer(14,"Name","fc_6")
%     tanhLayer("Name","tanh_6")
    fullyConnectedLayer(10,"Name","fc_7")
    tanhLayer("Name","tanh_7")
    fullyConnectedLayer(5,"Name","fc_8")
    tanhLayer("Name", "tanh_8")
    fullyConnectedLayer(1,"Name","fc_9")
    regressionLayer("Name","regressionoutput")];

miniBatchSize = 1024;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate', 0.001, ... % default
    'MaxEpochs', 20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{xVal', yVal'}, ...
    'ValidationFrequency', 50, ... % default
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment', 'parallel', ...
    'OutputFcn', @(x)makeLogVertAx(x));

% LearnRateSchedule = piecewise
% LearnRateDropFactor = 0.1
% LearnRateDropPeriod = 10 % epochs
% L2Regularization = 0.0001
% GradientDecayFactor = 0.9
% SquaredGradientDecayFactor = 0.999
% Epsilon = 10^-8
% GradientThreshold = Inf

%% Things to check
% Validity of simulation solutions on large design space
% 2. Normalization over large design space
% 3. mini batch size and learning rate tuning
% 1. Conv 1D for spatial s dimension? or reduce number of s points

%% Train network
% Features: Ndata x Nfeatures
% Responses: Ndata x Nresponses
[net, info] = trainNetwork(xTrain', yTrain', layers, options);

% Save
save(fullfile(data_dir,'models','model_dlnet.mat'),'net','info','options', ...
     'idxTrain', 'idxVal', 'idxTest', 'xs', 'ys');

%% Functions
function stop = makeLogVertAx(state)
    stop = false; % The function has to return a value.

    % Only do this once, following the 1st iteration
    if state.Iteration == 1
      % Get handles to "Training Progress" figures:
      hF  = findall(groot,'type','figure','Tag','NNET_CNN_TRAININGPLOT_UIFIGURE');
      % Assume the latest figure (first result) is the one we want, and get its axes:
      hAx = findall(hF(1),'type','Axes');

      for i = 1:length(hAx)
          if contains(hAx(i).Tag, 'REGRESSION')
              set(hAx(i), 'YScale','log')
          end
      end
    end
end