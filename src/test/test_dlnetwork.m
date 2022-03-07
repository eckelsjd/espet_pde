%% Test dlnetwork
% Joshua Eckels
% 3/2/22
clear all;
close all;
clc;

%% Load network and data
addpath('../emitter/');
addpath('..');
data_dir = '../../data/base/';
file = fullfile(data_dir,'train','train_dffnet_max.mat');
load(file);
file = fullfile(data_dir,'models',"model_dlnet.mat");
load(file);


%% Show results on test set (Emax network)
xtest = xdata(:, idxTest);
ytest = ydata(idxTest);

% Get mean height of the training samples
f = importdata('../../data/base/samples/samples.txt');
hvec = f.data(:,4);
h = mean(hvec);

% Show sampling distribution
figure(4);
subplot(5,1,1);
histogram(f.data(:,1)*1e6,'Normalization','probability');
xlabel('d [$\mu$m]','Interpreter','latex');
subplot(5,1,2);
histogram(f.data(:,2)*1e6,'Normalization','probability');
xlabel('$r_c$ [$\mu$m]','Interpreter','latex');
subplot(5,1,3);
histogram(f.data(:,3),'Normalization','probability');
xlabel('$\alpha$ [deg]','Interpreter','latex');
subplot(5,1,4);
histogram(f.data(:,4)*1e6,'Normalization','probability');
xlabel('$h$ [$\mu$m]','Interpreter','latex');
subplot(5,1,5);
histogram(f.data(:,5)*1e6,'Normalization','probability');
xlabel('$r_a$ [$\mu$m]','Interpreter','latex');
set(gcf,'Position',[200 800 400 700])

figure(5)
set(gcf,'Position',[700 800 600 500])
extractor_thickness = 76*10^(-6);
V0 = 1000;

for i = 1:length(xtest)
    
    params = xtest(:,i); % use the mean height h
    d = params(1)*h;
    rc = params(2)*h;
    alpha = params(3);
    ra = params(4)*h;
    
    emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0);
    [~,~,s,Ex,Ey] = EPOST.emitter_solution(emitter);
    t_mag = sqrt(Ex.^2 + Ey.^2);
    t_mag_max = max(t_mag);

    % Non-dimensionalize network inputs
    xpred = [d/h; rc/h; alpha; ra/h];

    % Normalize with xs settings from training
    xt_n = mapminmax('apply', xpred, xs);

    % Predict on normalized inputs xt_n (Ndata x Nfeatures)
    yt_n = predict(net, xt_n');
    
    % Unnormalize with ys settings from training
    ypred = mapminmax('reverse', yt_n', ys);

    % Re-dimensionalize
    Emax_pred = ypred * (V0/h);
    pd = 100*((t_mag_max - Emax_pred)/t_mag_max);
    pd_txt = sprintf('Diff = $%.2f \\%%$', pd);

    fprintf('Case = %d d = %.3E rc = %.3E alpha = %.2f ra = %.3E\n',i, d, rc, alpha*(180/pi), ra);
    figure(5)
    subplot(1,2,1);
    plot(s/s(end),t_mag,'-k');
    hold on;
    yline(Emax_pred,'-g','LineWidth',2);
    hold off;
    xlabel('Distance along emitter [\%]','Interpreter','latex');
    ylabel('$|E|$','Interpreter','latex')
    leg = legend('Simulation', 'Surrogate (max)');
    set(leg, 'Interpreter','latex')
    SW = [min(xlim) min(ylim)] + [diff(xlim) diff(ylim)]*0.05;
    fig_text = text(SW(1), SW(2), pd_text,'Interpreter','latex');

    figure(4)
    subplot(5,1,1);
    hold on;
    ah = xline(d*1e6,'-r','LineWidth',2);
    hold off;
    subplot(5,1,2);
    hold on;
    bh = xline(rc*1e6,'-r','LineWidth',2);
    hold off;
    subplot(5,1,3);
    hold on;
    ch = xline(alpha*(180/pi),'-r','LineWidth',2);
    hold off;
    subplot(5,1,4);
    hold on;
    dh = xline(h*1e6,'-r','LineWidth',2);
    hold off;
    subplot(5,1,5);
    hold on;
    eh = xline(ra*1e6,'-r','LineWidth',2);
    hold off;

    waitforbuttonpress
    delete(ah);
    delete(bh);
    delete(ch);
    delete(dh);
    delete(eh);
    delete(fig_text);
end

%% Show results on test set (full s network)
xTest = xdata(:,idxTest);
yTest = ydata(:,idxTest);

% Grab only unique test geometries (ignore s coordinate)
[xunique, ind, ~] = uniquetol(xTest(1:4,:)',1e-6,'ByRows',true);
xtest = xunique';
% ytest = yTest(:,ind);
% yt_n = mapminmax('apply', ytest, ys);

% Get mean height of the training samples
f = importdata('../../data/feasible/samples/samples.txt');
hvec = f.data(:,4);
h = mean(hvec);

% Prediction s grid
N = 200;
sd = linspace(0,1,N);

% Show sampling distribution
figure(4);
subplot(5,1,1);
histogram(f.data(:,1)*1e6,'Normalization','probability');
xlabel('d [$\mu$m]','Interpreter','latex');
subplot(5,1,2);
histogram(f.data(:,2)*1e6,'Normalization','probability');
xlabel('$r_c$ [$\mu$m]','Interpreter','latex');
subplot(5,1,3);
histogram(f.data(:,3),'Normalization','probability');
xlabel('$\alpha$ [deg]','Interpreter','latex');
subplot(5,1,4);
histogram(f.data(:,4)*1e6,'Normalization','probability');
xlabel('$h$ [$\mu$m]','Interpreter','latex');
subplot(5,1,5);
histogram(f.data(:,5)*1e6,'Normalization','probability');
xlabel('$r_a$ [$\mu$m]','Interpreter','latex');
set(gcf,'Position',[200 800 400 700])

figure(5)
set(gcf,'Position',[700 800 600 500])
extractor_thickness = 76*10^(-6);
V0 = 1000;

% Down-select
xtest = xtest(:,1:200:end);

for i = 1:length(xtest)
    
    params = xtest(:,i); % use the mean height h
    d = params(1)*h;
    rc = params(2)*h;
    alpha = params(3);
    ra = params(4)*h;
    
    emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0);
    [~,~,s,Ex,Ey] = EPOST.emitter_solution(emitter);
    t_mag = sqrt(Ex.^2 + Ey.^2);

    % Non-dimensionalize network inputs
    xpred = [repmat([d/h; rc/h; alpha; ra/h],1,N); sd];

    % Normalize with xs settings from training
    xt_n = mapminmax('apply', xpred, xs);

    % Predict on normalized inputs xt_n (Ndata x Nfeatures)
    yt_n = predict(net, xt_n');
    
    % Unnormalize with ys settings from training
    ypred = mapminmax('reverse', yt_n', ys);

    % Re-dimensionalize
    Ex_pred = ypred(1,:)*(V0/h);
    Ey_pred = ypred(2,:)*(V0/h);
    pred_mag = sqrt(Ex_pred.^2 + Ey_pred.^2);

    fprintf('Case = %d d = %.3E rc = %.3E alpha = %.2f ra = %.3E\n',i, d, rc, alpha*(180/pi), ra);
    figure(5)
    subplot(1,2,1);
    plot(s/s(end),Ex,'-k');
    hold on;
    plot(sd,Ex_pred,'--r');
    hold off;
    xlabel('Distance along emitter [\%]','Interpreter','latex');
    ylabel('$E_x$','Interpreter','latex')
    leg = legend('Simulation', 'Surrogate');
    set(leg, 'Interpreter','latex')
    subplot(1,2,2);
    plot(s/s(end),Ey,'-k');
    hold on;
    plot(sd,Ey_pred,'--r');
    xlabel('Distance along emitter [\%]','Interpreter','latex');
    ylabel('$E_y$','Interpreter','latex')
    leg = legend('Simulation', 'Surrogate');
    set(leg, 'Interpreter','latex')
    hold off;

    figure(4)
    subplot(5,1,1);
    hold on;
    ah = xline(d*1e6,'-r','LineWidth',2);
    hold off;
    subplot(5,1,2);
    hold on;
    bh = xline(rc*1e6,'-r','LineWidth',2);
    hold off;
    subplot(5,1,3);
    hold on;
    ch = xline(alpha*(180/pi),'-r','LineWidth',2);
    hold off;
    subplot(5,1,4);
    hold on;
    dh = xline(h*1e6,'-r','LineWidth',2);
    hold off;
    subplot(5,1,5);
    hold on;
    eh = xline(ra*1e6,'-r','LineWidth',2);
    hold off;

    waitforbuttonpress
    delete(ah);
    delete(bh);
    delete(ch);
    delete(dh);
    delete(eh);
end