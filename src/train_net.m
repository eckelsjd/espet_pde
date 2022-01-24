%% Train feedforward neural network
% Joshua Eckels
% 12/11/21
%% Load data
clear all;
close all;
clc;
addpath('./emitter/');
data_dir = '../data/feasible/';
file = fullfile(data_dir,'train','train_dffnet.mat');
load(file);

%% Split data set
trainRatio = 0.80;
valRatio = 0.10;
testRatio = 0.10;

% Total Mini-batch size of 10240
trainBS = 2^15; % 8192
valBS = valRatio*(trainBS/trainRatio);
testBS = testRatio*(trainBS/trainRatio);
[trainInd, valInd, testInd] = dividerand(length(xdata),trainRatio,valRatio,testRatio);

xtrain_ds = arrayDatastore(xdata(:,trainInd),'ReadSize',trainBS,'IterationDimension',2);
ytrain_ds = arrayDatastore(ydata(:,trainInd),'ReadSize',trainBS,'IterationDimension',2);
train_ds = combine(xtrain_ds,ytrain_ds);
train_mbq = minibatchqueue(train_ds,'MiniBatchSize',trainBS, 'OutputEnvironment','cpu',...
    'OutputAsDlarray',false);

xval_ds = arrayDatastore(xdata(:,valInd),'ReadSize',valBS,'IterationDimension',2);
yval_ds = arrayDatastore(ydata(:,valInd),'ReadSize',valBS,'IterationDimension',2);
val_ds = combine(xval_ds,yval_ds);
val_mbq = minibatchqueue(val_ds,'MiniBatchSize',valBS, 'OutputEnvironment','cpu',...
    'OutputAsDlarray',false);

xtest_ds = arrayDatastore(xdata(:,testInd),'ReadSize',testBS,'IterationDimension',2);
ytest_ds = arrayDatastore(ydata(:,testInd),'ReadSize',testBS,'IterationDimension',2);
test_ds = combine(xtest_ds,ytest_ds);
test_mbq = minibatchqueue(test_ds,'MiniBatchSize',testBS, 'OutputEnvironment','cpu',...
    'OutputAsDlarray',false);

%% Train feedforward neural network
% Set epochs
epochs = 1;
iter_per_batch = 1000;
net = feedforwardnet([34, 30, 26, 22, 18, 14, 10, 6]);
% net = feedforwardnet([32, 28, 24, 20]);
net.trainFcn = 'trainscg';
net.trainParam.epochs = iter_per_batch;
net.trainParam.max_fail = 1000;
net.divideFcn = 'divideind';

% Plot performance
figure()
perfLine = animatedline('Color',[0.85 0.325 0.098]);
vperfLine = animatedline('Color',[0 0.4 1]);
set(gca, 'YScale', 'log')
ylim([0 inf]);
xlabel("Iteration");
ylabel("Performance (MSE)");
grid on

% Save performance metrics
perf = zeros(3,1);

iteration = 0;
start = tic;
for epoch = 1:epochs
    curr_batch = 0;
    shuffle(train_mbq);
    shuffle(val_mbq);
    shuffle(test_mbq);

    while hasdata(train_mbq)
        curr_batch = curr_batch + 1;
        batch_start = tic;
        [x,y] = next(train_mbq);
        [xval,yval] = next(val_mbq);
        [xtest,ytest] = next(test_mbq);

        train_len = length(x);
        val_len = length(xval);
        test_len = length(xtest);

        % Group train,val,test data together
        xtrain = [x,xval,xtest];
        ytrain = [y,yval,ytest];
        net.divideParam.trainInd = 1:train_len;
        net.divideParam.valInd = train_len+1:train_len+val_len;
        net.divideParam.testInd = train_len+val_len+1:train_len+val_len+test_len;

        [net,tr] = train(net,xtrain,ytrain,'useParallel','yes','useGPU','yes');

        % Plot
        D = duration(0,0,toc(start),'Format','hh:mm:ss');
        addpoints(perfLine,iteration + tr.epoch(2:end),tr.perf(2:end))
        addpoints(vperfLine,iteration + tr.epoch(2:end),tr.vperf(2:end))
        title("Epoch: " + epoch + ", Elapsed: " + string(D))
        drawnow
        iteration = iteration + tr.epoch(end);

        % Update
        perf = [perf, [tr.perf(2:end);tr.vperf(2:end);tr.tperf(2:end)]];
        fprintf('Epoch: %d Batch: %d Runtime: %7.4f\n', epoch, curr_batch, toc(batch_start));

    end
    num_batches = curr_batch;
end
perf = perf(:,2:end);
xtest = xdata(:,testInd);
ytest = ydata(:,testInd);
save(fullfile(data_dir,'models','model_dffnet.mat'),'net','perf','xtest','ytest','trainBS',...
    'iter_per_batch','epochs','num_batches','trainRatio','valRatio','testRatio');
% exportONNXNetwork(net,fullfile(data_dir,'models','model_dffnet.onnx'));