%% Test Matlab's SeriesNetwork and export to onnx
% Joshua Eckels
% 3/2/22
clear all;
close all;
clc;
addpath('../emitter')
addpath('../postproc')

% Load network and training data
data_dir = '../../data/';
dataset1 = 'base'; 
dataset2 = 'feasible';
prefix = 'train';
file = fullfile(data_dir, dataset2, prefix, sprintf('%s_dffnet_max.mat', prefix));
feasible_data = load(file);
file = fullfile(data_dir, dataset1, prefix,sprintf('%s_dffnet_max.mat', prefix));
base_data = load(file);
xtrain = [feasible_data.xdata, base_data.xdata];
ytrain = [feasible_data.ydata, base_data.ydata];

file = fullfile(data_dir, dataset1, 'models',"model_onnxnet.mat");
load(file); % Gives net object and xs, ys  normalization settings

% Load test data
file = fullfile(data_dir, dataset2, 'test', 'test_dffnet_max.mat');
test_fd = load(file);
file = fullfile(data_dir,dataset1, 'test','test_dffnet_max.mat');
test_bd = load(file);

%% Show results on feasible and base test sets
xtest = [test_bd.xdata]; % [test_fd.xdata]
ytest = [test_bd.ydata]; % [test_fd.ydata]

% Preprocess
extractor_thickness = 76*10^(-6);
V0 = 1000;
hvec = xtest(4,:);
x = [xtest(1,:)./hvec; log10(xtest(2,:)./hvec); xtest(3,:); xtest(5,:)./hvec];
xn = mapminmax('apply',x, xs);

% Predict
yn = predict(net,xn');
ypred = mapminmax('reverse', yn', ys);
ypred = 10.^ypred;
ypred = ypred .* (V0./hvec);

% Analytical solution
Emax_ms = zeros(1,length(xtest));
x = 0; % test only at the tip of the emitter (x=0)
for i = 1:length(xtest)
    d_ms = xtest(1,i);
    rc_ms = xtest(2,i);
    [ytip, Ex_tip, Ey_tip] = EPOST.ms_solution(rc_ms, d_ms, V0, x);
    Emax_ms(i) = sqrt(Ex_tip^2 + Ey_tip^2);
end

% Error metrics for surrogate
rel_error = 100*(abs(ypred - ytest)./ytest);
rmse = sqrt(mean((ypred - ytest).^2));

% Error metrics for Martinez-Sanchez
rel_error_ms = 100*(abs(Emax_ms - ytest)./ytest);
rmse_ms = sqrt(mean((Emax_ms - ytest).^2));

% Plot results
figure()
histogram(rel_error,'Normalization','pdf','FaceColor','red')
label_str = sprintf('Relative error ($\\%%$)');
xlabel(label_str,'Interpreter','latex');
set(gcf,'color','white')

figure()
histogram(rel_error_ms,'Normalization','pdf','NumBins', 100);
hold on;
histogram(rel_error,'Normalization','pdf','FaceColor','red','FaceAlpha',0.7, 'NumBins',6);
set(gca,'YScale','log')
label_str = sprintf('Relative error ($\\%%$)');
xlabel(label_str,'Interpreter','latex');
leg = legend('Martinez-Sanchez', 'Surrogate');
set(leg,'Interpreter','latex');
set(gcf,'color','white')

% Show worst results
[~, idx] = max(rel_error);
params = xtest(:,idx);
figure();
subplot(5,1,1);
histogram(xtrain(1,:)*1e6,'Normalization','pdf');
hold on;
xline(params(1)*1e6,'-r','LineWidth',2);
xlabel('d [$\mu$m]','Interpreter','latex');
subplot(5,1,2);
histogram(xtrain(2,:)*1e6,'Normalization','pdf');
hold on;
xline(params(2)*1e6,'-r','LineWidth',2);
xlabel('$r_c$ [$\mu$m]','Interpreter','latex');
subplot(5,1,3);
histogram(xtrain(3,:)*(180/pi),'Normalization','pdf');
hold on;
xline(params(3)*(180/pi),'-r','LineWidth',2);
xlabel('$\alpha$ [deg]','Interpreter','latex');
subplot(5,1,4);
histogram(xtrain(4,:)*1e6,'Normalization','pdf');
hold on;
xline(params(4)*1e6,'-r','LineWidth',2);
xlabel('$h$ [$\mu$m]','Interpreter','latex');
subplot(5,1,5);
histogram(xtrain(5,:)*1e6,'Normalization','pdf');
hold on;
xline(params(5)*1e6,'-r','LineWidth',2);
xlabel('$r_a$ [$\mu$m]','Interpreter','latex');
set(gcf,'Position',[200 800 400 700])
set(gcf,'color','white')

emitter = Emitter(params(1),params(2),params(3),params(4),params(5),extractor_thickness,V0);
EPOST.solplot(emitter);