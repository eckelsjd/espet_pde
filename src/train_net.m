%% Train feedforward neural network
% Joshua Eckels
% 12/11/21
clear all;
close all;
clc;
addpath('.\emitter\');
data_dir = '..\data\afet\';
files = dir(fullfile(data_dir,'train'));

% Loop over all training data
xdata = zeros(5,1);
ydata = zeros(2,1);
for ii = 1:length(files)
    fprintf('%i\n',ii);
    if files(ii).isdir
        continue
    end
    file = fullfile(files(ii).folder,files(ii).name);
    load(file);

    x = train_data(1:5,:); % Input
    y = train_data(6:7,:); % Output
    xdata = [xdata, x];
    ydata = [ydata, y];
end
xdata = xdata(:,2:end);
ydata = ydata(:,2:end);

% Feedforward neural network
net = feedforwardnet(10);
net = train(net,xdata,ydata);

figure()
ypred = net(x);
subplot(1,2,1);
plot(x(5,:),y(1,:),'-k');
hold on;
plot(x(5,:),ypred(1,:),'--r');
subplot(1,2,2);
plot(x(5,:),y(2,:),'-k');
hold on;
plot(x(5,:),ypred(2,:),'--r');