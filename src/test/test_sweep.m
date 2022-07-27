%% Test  1d parameter sweep
clear all;
close all;
clc;
file = fullfile('../../data', 'geometry', 'models', 'model_onnxnet.mat');
load(file); % Gives net object and normalization settings
addpath('../emitter')
addpath('../postproc')

% Base design
N = 20;
% ra = 300e-6*ones(1,N);
% d = 500e-6*ones(1,N);
% rc = 30e-6*ones(1,N);
% alpha = 30*(pi/180)*ones(1,N);
% h = 300e-6*ones(1,N);
ra = 300e-6*ones(1,N);
d = 360e-6*ones(1,N);
rc = 16e-6*ones(1,N);
alpha = 30*(pi/180)*ones(1,N);
h = 350e-6*ones(1,N);
V = 100e3;
te = 76e-6;

% Swept parameter
ra = linspace(10e-6, 3000e-6, N);
% d = linspace(-280e-6, 3000e-6, N);
% rc = linspace(1e-6, 100e-6, N);
% alpha = linspace(10, 70, N)*(pi/180);
% h = linspace(50e-6, 1000e-6, N);
sweep = ra*1e3;

emax = zeros(1, N);
emax_sim = zeros(1, N);
for i=1:N
    % Simulation
    e = Emitter(d(i), rc(i), alpha(i), h(i), ra(i), te, V);
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(e);
    Emag = sqrt(Ex.^2 + Ey.^2);
    emax_sim(i) = max(Emag);

    % Preprocess
    exp_cdf = @(x, lambda) 1 - exp(-lambda*x);
    x = [(d(i)+xoffset)/h(i); rc(i)/h(i); alpha(i); ra(i)/h(i)];
    x = [exp_cdf(x(1), lambda_x(1)); exp_cdf(x(2), lambda_x(2)); x(3); exp_cdf(x(4), lambda_x(4)) ];
    xn = mapminmax('apply',x, xs);
    
    % Predict
    exp_cdf_inv = @(u, lambda) (-1/lambda)*log(abs(1-u));
    yn = predict(net,xn');
    ypred = mapminmax('reverse', yn', ys);
    ypred = exp_cdf_inv(ypred, lambda_y);
    ypred = ypred .* (V/h(i));
    
    % Surrogate
    emax(i) = ypred;
end

% Martinez Sanchez
N_ms = 1000;
d_ms = linspace(min(d), max(d), N_ms);
rc_ms = linspace(min(rc), max(rc), N_ms);
sweep_ms = linspace(min(sweep), max(sweep), N_ms);
emax_ms = zeros(1, N_ms);
idx = find(d_ms > 0);
for i = 1:N_ms
    if d_ms(i) < 0
        % pass
    else
        [y, Ex, Ey] = EPOST.ms_solution(rc_ms(i), d_ms(i), V, 0);
        emax_ms(i) = sqrt(Ex.^2 + Ey.^2);
    end
end

figure()
plot(sweep, emax, '--or');
hold on;
plot(sweep, emax_sim, '--ok');
plot(sweep_ms(idx), emax_ms(idx), '-k');
xlabel('Radius of aperture $R_a$ [$mm$]','Interpreter','latex');
% xlabel('Tip-to-extractor distance $d$ [$\mu m$]','Interpreter','latex');
% xlabel('Radius of curvature $R_c$ [$\mu m$]','Interpreter','latex');
% xlabel('Cone half-angle $\alpha$ [deg]','Interpreter','latex');
% xlabel('Emitter height $h$ [$\mu m$]','Interpreter','latex');
ylabel('Max E-field magnitude [V/m]', 'Interpreter','latex');
leg = legend('Surrogate', 'Simulation', 'Martinez-Sanchez');
set(leg, 'Interpreter','latex');
set(gcf, 'color','white');

%% Test 2d parameter sweep with surrogate only
clear all;
close all;
clc;
file = fullfile('../../data', 'geometry', 'models', 'model_onnxnet.mat');
load(file); % Gives net object and normalization settings
addpath('../postproc');
N = 500;
grid_sz = [N, N];
ra = 300e-6*ones(grid_sz);
d = 500e-6*ones(grid_sz);
rc = 30e-6*ones(grid_sz);
alpha = 30*(pi/180)*ones(grid_sz);
h = 300e-6*ones(grid_sz);
V = 1000;
te = 76e-6;

% Sweep params
sweep1 = linspace(10e-6, 3000e-6, N);
sweep2 = linspace(-280e-6, 3000e-6, N);
[g1, g2] = meshgrid(sweep1, sweep2);
ra = g1;
d = g2;

exp_cdf = @(x, lambda) 1 - exp(-lambda*x);
hvec = reshape(h, 1, []);
x1 = (reshape(d, 1, []) + xoffset)./hvec;
x2 = reshape(rc, 1, [])./hvec;
x3 = reshape(alpha, 1, []);
x4 = reshape(ra, 1, [])./hvec;
x = [exp_cdf(x1, lambda_x(1)); exp_cdf(x2, lambda_x(2)); x3; exp_cdf(x4, lambda_x(4))];
xn = mapminmax('apply',x, xs);

% Predict
exp_cdf_inv = @(u, lambda) (-1/lambda)*log(abs(1-u));
yn = predict(net,xn');
ypred = mapminmax('reverse', yn', ys);
ypred = exp_cdf_inv(ypred, lambda_y);
ypred = ypred .* (V./hvec);

% Reshape to grid and plot
y_g = reshape(ypred, grid_sz);

nc = 100;
contourf(g1*1e6, g2*1e6, y_g, nc, 'LineColor','none');
xlabel('Radius of aperture $R_a$ [$\mu m$]','Interpreter','latex');
% xlabel('Tip-to-extractor distance $d$ [$\mu m$]','Interpreter','latex');
% xlabel('Radius of curvature $R_c$ [$\mu m$]','Interpreter','latex');
% xlabel('Cone half-angle $\alpha$ [deg]','Interpreter','latex');
% xlabel('Emitter height $h$ [$\mu m$]','Interpreter','latex');
% ylabel('Radius of aperture $R_a$ [$\mu m$]','Interpreter','latex');
ylabel('Tip-to-extractor distance $d$ [$\mu m$]','Interpreter','latex');
% ylabel('Radius of curvature $R_c$ [$\mu m$]','Interpreter','latex');
% ylabel('Cone half-angle $\alpha$ [deg]','Interpreter','latex');
% ylabel('Emitter height $h$ [$\mu m$]','Interpreter','latex');
set(gcf,'color','white');
rgb_cmap = EPOST.interp_cmap(EPOST.cmap, nc);
colormap(flipud(rgb_cmap));
cb = colorbar();
cb.LineWidth = 0.8;
cb.FontSize = 11;
set(get(cb,'label'),'interpreter','latex');
set(get(cb,'label'),'string','Electric field magnitude [$\frac{V}{m}$]');
set(get(cb,'label'),'FontSize',11);
    
