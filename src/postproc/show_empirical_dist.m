%% Show empirical PDF fits
% Joshua Eckels
% 6/6/22
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

% Load model
file = fullfile(data_dir, dataset, 'models',"model_onnxnet.mat");
load(file); % Gives net object and normalization settings

% Non-dimensionalize
hvec = xdata(4,:);
V0 = 1000;
xoffset = 1000e-6;
x = [(xdata(1,:)+xoffset)./hvec; xdata(2,:)./hvec; xdata(3,:); xdata(5,:)./hvec];
y = ydata ./ (V0./hvec);

% Redistribute
exp_cdf = @(x, lambda) 1 - exp(-lambda*x);
xd = [exp_cdf(x(1,:), lambda_x(1)); exp_cdf(x(2,:), lambda_x(2)); x(3,:); exp_cdf(x(4,:), lambda_x(4))];
yd = exp_cdf(y, lambda_y);

% Plot results
% Radius of curvature
figure()
hold on;
histogram(x(2,:),'Normalization','pdf','FaceColor', 'blue');
x_d = linspace(min(x(2,:)), max(x(2,:)), 1000);
pdf = lambda_x(2)*exp(-lambda_x(2)*x_d);
plot(x_d, pdf, 'Color', [230 25 75]/255, 'LineStyle', '-', 'LineWidth', 2);
xlabel('Scaled radius of curvature $\frac{R_c}{h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
set(gcf,'color','white') 
leg = legend('1-D marginal', 'PDF fit');
set(leg, 'Interpreter','latex');
xlim([min(x(2,:)) inf])

figure()
subplot(1,2,1);
histogram(x(2,:),'Normalization','pdf','FaceColor', 'blue');
xlabel('Scaled radius of curvature $\frac{R_c}{h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(x(2,:)) inf])
subplot(1,2,2);
histogram(xd(2,:),'Normalization','pdf','FaceColor', 'red');
xlabel('Transformed radius of curvature $g\left(\frac{R_c}{h}\right)$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(xd(2,:)) inf])
set(gcf,'color','white') 

% Tip-to extractor distance
idx = 1;
figure()
hold on;
histogram(x(idx,:),'Normalization','pdf','FaceColor', 'blue');
x_d = linspace(min(x(idx,:)), max(x(idx,:)), 1000);
pdf = lambda_x(idx)*exp(-lambda_x(idx)*x_d);
plot(x_d, pdf, 'Color', [230 25 75]/255, 'LineStyle', '-', 'LineWidth', 2);
xlabel('Scaled tip-to-extractor distance $\frac{d + d_0}{h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
set(gcf,'color','white') 
leg = legend('1-D marginal', 'PDF fit');
set(leg, 'Interpreter','latex');
xlim([min(x(idx,:)) inf])

figure()
subplot(1,2,1);
histogram(x(idx,:),'Normalization','pdf','FaceColor', 'blue');
xlabel('Scaled tip-to-extractor distance $\frac{d + d_0}{h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(x(idx,:)) inf])
subplot(1,2,2);
histogram(xd(idx,:),'Normalization','pdf','FaceColor', 'red');
xlabel('Transformed tip-to-extractor distance $g\left(\frac{d + d_0}{h}\right)$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(xd(idx,:)) inf])
set(gcf,'color','white') 

% Aperture radius
idx = 4;
figure()
hold on;
histogram(x(idx,:),'Normalization','pdf','FaceColor', 'blue');
x_d = linspace(min(x(idx,:)), max(x(idx,:)), 1000);
pdf = lambda_x(idx)*exp(-lambda_x(idx)*x_d);
plot(x_d, pdf, 'Color', [230 25 75]/255, 'LineStyle', '-', 'LineWidth', 2);
xlabel('Scaled radius of aperture $\frac{R_a}{h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
set(gcf,'color','white') 
leg = legend('1-D marginal', 'PDF fit');
set(leg, 'Interpreter','latex');
xlim([min(x(idx,:)) inf])

figure()
subplot(1,2,1);
histogram(x(idx,:),'Normalization','pdf','FaceColor', 'blue');
xlabel('Scaled radius of aperture $\frac{R_a}{h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(x(idx,:)) inf])
subplot(1,2,2);
histogram(xd(idx,:),'Normalization','pdf','FaceColor', 'red');
xlabel('Transformed radius of aperture $g\left(\frac{R_a}{h}\right)$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(xd(idx,:)) inf])
set(gcf,'color','white') 

% Electric field magnitude
figure()
hold on;
histogram(y,'Normalization','pdf','FaceColor', 'blue');
y_d = linspace(min(y), max(y), 1000);
pdf = lambda_y*exp(-lambda_y*y_d);
plot(y_d, pdf, 'Color', [230 25 75]/255, 'LineStyle', '-', 'LineWidth', 2);
xlabel('Scaled electric field magnitude $\frac{E_{\mathrm{max}}}{V_0/h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
set(gcf,'color','white') 
leg = legend('1-D marginal', 'PDF fit');
set(leg, 'Interpreter','latex');
xlim([min(y) 50])

figure()
subplot(1,2,1);
histogram(y,'Normalization','pdf','FaceColor', 'blue');
xlabel('Scaled electric field magnitude $\frac{E_{\mathrm{max}}}{V_0/h}$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(y) 50])
subplot(1,2,2);
histogram(yd,'Normalization','pdf','FaceColor', 'red');
xlabel('Transformed electric field magnitude $g\left(\frac{E_{\mathrm{max}}}{V_0/h}\right)$', 'Interpreter','latex');
ylabel('PDF','Interpreter', 'latex');
xlim([min(yd) inf])
set(gcf,'color','white') 