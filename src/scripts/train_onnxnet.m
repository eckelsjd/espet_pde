%% Train SeriesNetwork on feasible/base
% Joshua Eckels
% 3/2/22
clear all;
close all;
clc;

% Load training data
data_dir = '../../data/';
dataset = 'geometry';
prefix = 'train';
file = fullfile(data_dir, dataset, prefix, sprintf('%s_dffnet_max.mat', prefix));
geometry_data = load(file);
xdata = geometry_data.xdata;
ydata = geometry_data.ydata;

% Remove Emax outliers for better training
% [ydata,outlierIndices] = rmoutliers(ydata,"percentiles",[0 99.9]);
% xdata = xdata(:, ~outlierIndices);
% [ydata,outlierIndices] = rmoutliers(ydata,"percentiles",[0 99.9]);
% xdata = xdata(:, ~outlierIndices);

% Non-dimensionalize
hvec = xdata(4,:);
V0 = 1000;
xoffset = 1000e-6;
x = [(xdata(1,:)+xoffset)./hvec; xdata(2,:)./hvec; xdata(3,:); xdata(5,:)./hvec];
y = ydata ./ (V0./hvec);

% Convert to uniform distribution by CDF transform
[x1, lambdax1] = exp_cdf(x(1,:));
[x2, lambdax2] = exp_cdf(x(2,:));
% % Alpha is already pretty uniform by itself
[x4, lambdax4] = exp_cdf(x(4,:));
[y, lambda_y] = exp_cdf(y);
x = [x1; x2; x(3,:); x4];
lambda_x = [lambdax1, lambdax2, 0, lambdax4];

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
    'MaxEpochs', 1000, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{xVal', yVal'}, ...
    'ValidationFrequency', 100, ... % default 50
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ExecutionEnvironment', 'parallel', ...
    'OutputFcn', @(x)makeLogVertAx(x), ...
    'OutputNetwork', 'best-validation-loss');

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
save(fullfile(data_dir, dataset, 'models','model_onnxnet.mat'),'net', ...
    'info','options', 'xs', 'ys', 'lambda_x', 'lambda_y', 'xoffset', 'miniBatchSize', 'V0');

% Export to onnx and save normalization data
exportONNXNetwork(net, fullfile(data_dir, dataset, 'models', 'esi_surrogate.onnx'));
save(fullfile(data_dir, dataset, 'models', 'norm_data.mat'), 'xs', 'ys', 'V0', 'lambda_x', 'lambda_y', 'xoffset');

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


% Fit an exponential distribution to the column vector x and return the CDF
% of the samples in x (CDF(X) ~ U(0,1))
function [u, lambda] = exp_cdf(x)
    pd = fitdist(x', 'exponential');
    lambda = 1/pd.mu;
    u = 1 - exp(-lambda*x);
end