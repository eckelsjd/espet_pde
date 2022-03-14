%% View design space
% 3/9/22
% Joshua Eckels
clear all;
close all;
clc;

fb = importdata('../data/base/samples/samples.txt');
ff = importdata('../data/feasible/samples/samples.txt');
load('xbad.mat') % outliers

figure();

subplot(5,1,1);
histogram(ff.data(:,1)*1e6,'Normalization','count', 'FaceAlpha',0.4);
hold on;
histogram(fb.data(:,1)*1e6,'Normalization','count', 'FaceColor', 'red', 'FaceAlpha',0.8);
xline(xbad(1,:)*1e6,'-r')
xlabel('d [$\mu$m]','Interpreter','latex');

subplot(5,1,2);
histogram(ff.data(:,2)*1e6,'Normalization','count', 'FaceAlpha',0.4);
hold on;
histogram(fb.data(:,2)*1e6,'Normalization','count', 'FaceColor', 'red', 'FaceAlpha',0.8);
xline(xbad(2,:)*1e6,'-r')
xlabel('$r_c$ [$\mu$m]','Interpreter','latex');

subplot(5,1,3);
histogram(ff.data(:,3),'Normalization','count', 'FaceAlpha',0.4);
hold on;
histogram(fb.data(:,3),'Normalization','count', 'FaceColor', 'red', 'FaceAlpha',0.8);
xline(xbad(3,:)*(180/pi),'-r')
xlabel('$\alpha$ [deg]','Interpreter','latex');

subplot(5,1,4);
histogram(ff.data(:,4)*1e6,'Normalization','count', 'FaceAlpha',0.4);
hold on;
histogram(fb.data(:,4)*1e6,'Normalization','count', 'FaceColor', 'red', 'FaceAlpha',0.8);
xline(xbad(4,:)*1e6,'-r')
xlabel('$h$ [$\mu$m]','Interpreter','latex');

subplot(5,1,5);
histogram(ff.data(:,5)*1e6,'Normalization','count', 'FaceAlpha',0.4);
hold on;
histogram(fb.data(:,5)*1e6,'Normalization','count', 'FaceColor', 'red', 'FaceAlpha',0.8);
xline(xbad(5,:)*1e6,'-r')
xlabel('$r_a$ [$\mu$m]','Interpreter','latex');

set(gcf,'Position',[200 800 400 700])
