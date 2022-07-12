%% Test training data sparsity
clear all;
close all;
clc;
addpath('../emitter')
addpath('../postproc')

ra = linspace(1e-5, 3e-3, 10);
d = 360e-6;
rc = 1.6e-5;
alpha = 30*(pi/180);
h = 350e-6;
V = 100e3;
te = 76e-6;
mesh_size = 0.8e-6;  % [m]

% Load training data
data_dir = '../../data/';
dataset = 'geometry';
prefix = 'train';
file = fullfile(data_dir, dataset, prefix, sprintf('%s_dffnet_max.mat', prefix));
geometry_data = load(file);
xdata = geometry_data.xdata;
ydata = geometry_data.ydata;
Nsamples = length(xdata);

p = [d, rc, alpha, h, 2000e-6]';
[xn, ~] = mapminmax([xdata, p], 0, 1);

params = repmat(xn(:,end), 1, Nsamples);
dist = vecnorm(xn(:,1:end-1) - params);
[min_dist, min_idx] = min(dist);

xp = xdata(:, min_idx);
nearest_dist = abs(p - xp);
nearest_rel_dist = 100*(p-xp)./xp;

N = 30;
d_samples = rand(1,N)*(2*nearest_dist(1)) + (xp(1) - nearest_dist(1));
rc_samples = rand(1,N)*(2*nearest_dist(2)) + (xp(2) - nearest_dist(2));
alpha_samples = rand(1,N)*(2*nearest_dist(3)) + (xp(3) - nearest_dist(3));
h_samples = rand(1,N)*(2*nearest_dist(4)) + (xp(4) - nearest_dist(4));
ra_samples = rand(1,N)*(2*nearest_dist(5)) + (xp(5) - nearest_dist(5));

file = fullfile('../../data', 'geometry', 'models', 'model_onnxnet.mat');
load(file); % Gives net object and normalization settings

emax = zeros(1,N);
emax_sim = zeros(1,N);
for i=1:N
    d = d_samples(i);
    rc = rc_samples(i);
    alpha = alpha_samples(i);
    h = h_samples(i);
    ra = ra_samples(i);
    e = Emitter(d, rc, alpha, h, ra, te, V);
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(e);
    Emag = sqrt(Ex.^2 + Ey.^2);
    emax_sim(i) = max(Emag);

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

rel_error = 100*(abs(emax - emax_sim)./emax_sim);

% Plot results
figure()
histogram(rel_error,'NumBins',10,'FaceColor','red')
label_str = sprintf('Relative error ($\\%%$)');
xlabel(label_str,'Interpreter','latex');
set(gcf,'color','white')

%% Test hyperboloid
% h = Hyperboloid(d, rc, alpha, h, 0, te, V, mesh_size);
% [x,y,s,Ex,Ey] = EPOST.emitter_solution(h);
% [y, Ex_ms, Ey_ms] = EPOST.ms_solution(rc, d, V, x);
% Emag = sqrt(Ex.^2 + Ey.^2);
% Emag_ms = sqrt(Ex_ms.^2 + Ey_ms.^2);
% 
% figure()
% hold on;
% plot(x, Emag, '--r');
% plot(x, Emag_ms, '-k');
% leg = legend('Simulated hyperboloid', 'Martinez-Sanchez');
% set(leg, 'Interpreter','latex');
% xlabel('X coordinate [m]', 'Interpreter' ,'latex');
% ylabel('$|E|$ on emitter surface [V/m]', 'Interpreter', 'latex');
% set(gcf,'color','white');