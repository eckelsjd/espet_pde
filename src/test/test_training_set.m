%% Test training data sparsity
clear all;
close all;
clc;
addpath('../emitter')
addpath('../postproc')
V = 1000;
te = 76e-6;

% Load training data
data_dir = '../../data/';
dataset = 'geometry';
prefix = 'train';
file = fullfile(data_dir, dataset, prefix, sprintf('%s_dffnet_max.mat', prefix));
geometry_data = load(file);
xdata = geometry_data.xdata;
ydata = geometry_data.ydata;
N = length(xdata);

file = fullfile('../../data', 'geometry', 'models', 'model_onnxnet.mat');
load(file); % Gives net object and normalization settings

emax = zeros(1,N);
for i=1:N
    d = xdata(1,i);
    rc = xdata(2,i);
    alpha = xdata(3,i);
    h = xdata(4,i);
    ra = xdata(5,i);

    % Preprocess
    exp_cdf = @(x, lambda) 1 - exp(-lambda*x);
    x = [(d+xoffset)/h; rc/h; alpha; ra/h];
    x = [exp_cdf(x(1), lambda_x(1)); exp_cdf(x(2), lambda_x(2)); x(3); exp_cdf(x(4), lambda_x(4)) ];
    xn = mapminmax('apply',x, xs);
    
    % Predict
    exp_cdf_inv = @(u, lambda) (-1/lambda)*log(abs(1-u));
    yn = predict(net,xn');
    ypred = mapminmax('reverse', yn', ys);
    ypred = exp_cdf_inv(ypred, lambda_y);
    ypred = ypred .* (V/h);
    
    % Save result
    emax(i) = ypred;
end

rel_error = 100*(abs(emax - ydata)./ydata);

% Plot results
figure()
histogram(rel_error,'FaceColor','red')
label_str = sprintf('Relative error ($\\%%$)');
xlabel(label_str,'Interpreter','latex');
set(gcf,'color','white')