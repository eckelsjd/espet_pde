% Joshua Eckels
% 12/8/21
clear all;
close all;
clc;
addpath('../emitter');
addpath('../postproc');
% sampler_input_fname = '..\data\afet.json'; 
% fid = fopen(fname); 
% raw = fread(fid,inf); 
% str = char(raw'); 
% fclose(fid); 
% val = jsondecode(str);

%% AFET-2 design
% d = 0;
% rc = 10*10^(-6);
% alpha = 15*(pi/180);
% h = 300*10^(-6); 
% ra = 254*10^(-6);
d = 3e-6;
rc = 1.5e-5;
alpha = 2.678e-1;
h = 3.018e-4;
ra = 2.486e-4;
extractor_thickness = 76*10^(-6);
V0 = 1000;

emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness, V0);
[e_x,e_y,e_s,Ex,Ey] = EPOST.emitter_solution(emitter);
Emag = sqrt(Ex.^2 + Ey.^2);

figure()
plot(e_x, Emag,'-k');
xlabel('X [m]','Interpreter','latex');
ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');

skip = 8;
EPOST.solplot(emitter);
EPOST.vectorplot(emitter,skip);

% Surrogate
% Load network and training data
data_dir = '../../data/';
dataset = 'geometry';
file = fullfile(data_dir, dataset, 'models',"model_onnxnet.mat");
load(file); % Gives net object and normalization settings

% Preprocess
x = [(d+xoffset)/h; rc/h; alpha; ra/h];
x = [exp_cdf(x(1), lambda_x(1)); exp_cdf(x(2), lambda_x(2)); x(3); exp_cdf(x(4), lambda_x(4)) ];
xn = mapminmax('apply',x, xs);

% Predict
yn = predict(net,xn');
ypred = mapminmax('reverse', yn', ys);
ypred = exp_cdf_inv(ypred, lambda_y);
ypred = ypred * (V0/h);

function u = exp_cdf(x, lambda)
    u = 1 - exp(-lambda*x);
end

function y = exp_cdf_inv(u, lambda)
    y = (-1/lambda)*log(abs(1-u));
end
