%% Load data
clear all;
close all;
clc;
addpath('../emitter')
addpath('..')
% load('../../data/feasible/models/model_dffnet.mat');
% load('../../data/feasible/model_tr.mat');
f = importdata('../../data/feasible/samples/samples.txt');
fb = importdata('../../data/base/samples/samples.txt');

% nbins = 10;
h = figure();
subplot(5,1,1);
histogram(f.data(:,1)*1000,'Normalization','probability');
xlabel('d [mm]','Interpreter','latex');
subplot(5,1,2);
histogram(f.data(:,2)*1000,'Normalization','probability');
xlabel('$r_c$ [mm]','Interpreter','latex');
subplot(5,1,3);
histogram(f.data(:,3),'Normalization','probability');
xlabel('$\alpha$ [deg]','Interpreter','latex');
subplot(5,1,4);
histogram(f.data(:,4)*1000,'Normalization','probability');
xlabel('$h$ [mm]','Interpreter','latex');
subplot(5,1,5);
histogram(f.data(:,1)*1000,'Normalization','probability');
xlabel('$r_a$ [mm]','Interpreter','latex');

set(gcf,'Position',[200 800 400 700]);
