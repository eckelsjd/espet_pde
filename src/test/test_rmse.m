%% Compare RMSE of surrogate and analytical predictions
% Joshua Eckels
% 1/23/22
clear all;
close all;
clc;

addpath('../emitter')
addpath('..')
load('../../data/feasible/models/model_dffnet.mat');
f = importdata('../../data/feasible/test/samples/samples.txt');
extractor_thickness = 76*10^(-6);
V0 = 1000;

% Interpolation grid
N = 200;
sd = linspace(0,1,N);

% Get geometry test samples
% [d, rc, alpha, h, ra]
samples = f.data;
[nsamples, nparams] = size(samples);

% Allocate space for rmse calculations (save Ex and Ey mse separately)
rmse_ms = zeros(2,nsamples);        % RMSE(martinez-sanchez,simulation)
rmse_surr = zeros(2,nsamples);      % RMSE(surrogate,simulation)
rmse_ms_max = zeros(1,nsamples);    % Just RMSE at peak electric field
rmse_surr_max = zeros(1,nsamples);

% Loop and compare mse
for ii = 1:nsamples
    params = samples(ii,:);
    d = params(1); rc = params(2); alpha = params(3); h = params(4); ra = params(5);

    % Simulation (truth solution)
    filename = EPOST.get_filename(d,rc,alpha,h,ra,'esimtest');
    load(fullfile('../../data/feasible/test/sims',filename)); % gets emitter sim data
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);
    Ex_sim = interp1(s/s(end),Ex,sd,'pchip');
    Ey_sim = interp1(s/s(end),Ey,sd,'pchip');
    Emag_sim = sqrt(Ex_sim.^2 + Ey_sim.^2);

    % Analytical (MS solution)
    [y, Ex, Ey] = EPOST.ms_solution(rc, d, V0, x);
    ds = [0, sqrt(diff(x).^2 + diff(y).^2)];
    s = cumsum(ds);
    Ex_ms = interp1(s/s(end),Ex,sd,'pchip');
    Ey_ms = interp1(s/s(end),Ey,sd,'pchip');
    Emag_ms = sqrt(Ex_ms.^2 + Ey_ms.^2);

    % Surrogate (approximate solution)
    xpred = [repmat([d/h; rc/h; alpha*(pi/180); ra/h],1,N); sd];
    ypred = net(xpred);
    Ex_surr = ypred(1,:)*(V0/h);
    Ey_surr = ypred(2,:)*(V0/h);
    Emag_surr = sqrt(Ex_surr.^2 + Ey_surr.^2);

    % Save rmse results
    rmse_ms(1,ii) = sqrt(mean((Ex_sim - Ex_ms).^2));
    rmse_ms(2,ii) = sqrt(mean((Ey_sim - Ey_ms).^2));
    rmse_surr(1,ii) = sqrt(mean((Ex_sim - Ex_surr).^2));
    rmse_surr(2,ii) = sqrt(mean((Ey_sim - Ey_surr).^2));

    rmse_ms_max(ii) = sqrt(mean((max(Emag_sim) - max(Emag_ms)).^2));
    rmse_surr_max(ii) = sqrt(mean((max(Emag_sim) - max(Emag_surr)).^2));
end

%% Load results directly
clear all;
close all;
clc;
load('rmse_results.mat');

% Plot histogram of errors for Ex, Ey, and Emag predictions
figure()
subplot(1,3,1);
histogram(rmse_ms(1,:),'Normalization','probability');
hold on;
histogram(rmse_surr(1,:),'Normalization','probability','FaceColor',[1 0 0]);
xlabel('RMSE($E_x$) [V/m]','Interpreter','latex');
legend('Martinez-Sanchez','Surrogate');
% legend('Surrogate')
subplot(1,3,2);
histogram(rmse_ms(2,:),'Normalization','probability');
hold on;
histogram(rmse_surr(2,:),'Normalization','probability','FaceColor',[1 0 0]);
xlabel('RMSE($E_y$) [V/m]','Interpreter','latex');
% legend('Surrogate')
legend('Martinez-Sanchez','Surrogate');
subplot(1,3,3);
histogram(rmse_ms_max,'Normalization','probability');
hold on;
histogram(rmse_surr_max,'Normalization','probability','FaceColor',[1 0 0]);
xlabel('RMSE($|\vec{E}_{tip}|$) [V/m]','Interpreter','latex');
legend('Martinez-Sanchez','Surrogate');
% legend('Surrogate')