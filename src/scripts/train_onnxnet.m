%% Train SeriesNetwork on feasible/base
% Joshua Eckels
% 3/2/22
clear all;
close all;
clc;

% Load training data
data_dir = '../../data/';
dataset1 = 'base'; 
dataset2 = 'feasible';
prefix = 'train';
file = fullfile(data_dir, dataset2, prefix, sprintf('%s_dffnet_max.mat', prefix));
feasible_data = load(file);
file = fullfile(data_dir, dataset1, prefix,sprintf('%s_dffnet_max.mat', prefix));
base_data = load(file);

% Combine feasible and base design spaces
xdata = [feasible_data.xdata, base_data.xdata];
ydata = [feasible_data.ydata, base_data.ydata];

% Remove Emax outliers for better training
[ydata,outlierIndices] = rmoutliers(ydata,"percentiles",[0 99.98]);
xdata = xdata(:, ~outlierIndices);

% Non-dimensionalize
hvec = xdata(4,:);
V0 = 1000;
x = [xdata(1,:)./hvec; log10(xdata(2,:)./hvec); xdata(3,:); xdata(5,:)./hvec];
y = log10(ydata ./ (V0./hvec));

% Normalize
[xn, xs] = mapminmax(x, -1, 1);
[yn, ys] = mapminmax(y, -1, 1);

% Split train/val sets
Ndata = size(xn,2);
Ntrain = floor(0.8*Ndata);
Nval = Ndata - Ntrain;

idx = randperm(Ndata);
idxTrain = idx(1:Ntrain);
idxVal = idx(Ntrain+1:end);

xTrain = xn(:,idxTrain); yTrain = yn(:,idxTrain);
xVal = xn(:,idxVal);     yVal = yn(:,idxVal);

%% Set up network
[layers, num_params] = construct_ffnet([4, 9, 5, 1]);

miniBatchSize = 2048;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate', 0.001, ... % default
    'MaxEpochs', 2000, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{xVal', yVal'}, ...
    'ValidationFrequency', 200, ... % default 50
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputFcn', @(x)makeLogVertAx(x));

%% Other training options
% LearnRateSchedule = piecewise
% LearnRateDropFactor = 0.1
% LearnRateDropPeriod = 10 % epochs
% L2Regularization = 0.0001
% GradientDecayFactor = 0.9
% SquaredGradientDecayFactor = 0.999
% Epsilon = 10^-8
% GradientThreshold = Inf

%% Train network
% Features: Ndata x Nfeatures
% Responses: Ndata x Nresponses
[net, info] = trainNetwork(xTrain', yTrain', layers, options);

% Save network mat object and training info
save(fullfile(data_dir, dataset1, 'models','model_onnxnet.mat'),'net', ...
    'info','options', 'xs', 'ys', 'miniBatchSize', 'V0');

% Export to onnx and save normalization data
exportONNXNetwork(net, fullfile(data_dir, dataset1, 'models', 'esi_surrogate.onnx'))
save(fullfile(data_dir, dataset1, 'models', 'norm_data.mat'), 'xs', 'ys', 'V0')

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

% Construct simple regression feedforward FC network
function [layers, num_params] = construct_ffnet(nodes)
    % If you're reading this, you might wonder why not use Matlab's built in
    % feedforwardnet. Well I did. And it doesn't export to .onnx, so I
    % build it manually here.
    Ninputs = nodes(1);
    Noutputs = nodes(end);

    num_params = 0;
    layers = [];
    layers = [layers featureInputLayer(Ninputs,"Name","featureinput")];

    % Hidden layers
    for ii = 2:length(nodes)-1
        num_hidden_nodes = nodes(ii);

        fc_name = sprintf('fc_%i',ii);
        tanh_name = sprintf('tanh_%i',ii);
        layers = [layers
            fullyConnectedLayer(num_hidden_nodes,"Name",fc_name)
            tanhLayer("Name",tanh_name)];

        num_params = num_params + num_hidden_nodes * (nodes(ii-1) + 1);

    end
    
    num_params = num_params + (Noutputs) * (nodes(ii) + 1);
    fc_name = sprintf('fc_%i',length(nodes));
    layers = [layers
        fullyConnectedLayer(Noutputs, "Name", fc_name)
        regressionLayer("Name","regressionoutput")];
end