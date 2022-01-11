clear all;
close all;
clc;

% Load data
addpath('..\emitter')
addpath('..')
load('../../data/feasible/model.mat');
f = importdata('../../data/feasible/samples/samples.txt');
nsamples = 1035;
params = mean(f.data(1:nsamples,:));
d = params(1);
rc = params(2);
alphad = params(3)*(pi/180);
h = params(4);
ra = params(5);
extractor_thickness = 76*10^(-6);
V0 = 1000;

% Forward UQ
N = 5000;

% Sample geometry parameters from U(+-5%)
di = unifrnd(0.95*d,1.05*d,1,N);
rci = unifrnd(0.95*rc,1.05*rc,1,N);
alphai = unifrnd(0.95*alphad,1.05*alphad,1,N);
hi = unifrnd(0.95*h,1.05*h,1,N);
rai = unifrnd(0.95*ra,1.05*ra,1,N);

% Non-dimensionalize
lchar = hi;
in = zeros(5,N);
in(1,:) = di./lchar;
in(2,:) = rci./lchar;
in(3,:) = alphai;
in(4,:) = rai./lchar;
in(5,:) = 0; % evaluate at s=0 (the tip)

% Evaluate surrogate model on samples
y = net(in);
Ex = y(1,:)*(V0/h);
Ey = y(2,:)*(V0/h);

% Evaluate surrogate model on target
ym = net([d/h; rc/h; alphad; ra/h; 0]);

% Plot results
figure()
hold on;
scatter(Ex,Ey,10,'ob','MarkerFaceAlpha',0.6,'MarkerFaceColor','b');
plot(ym(1)*(V0/h),ym(2)*(V0/h),'.r','MarkerSize',15);
xlabel('$E_x$ [V/m]','Interpreter','latex');
ylabel('$E_y$ [V/m]','Interpreter','latex');
leg = legend('Samples','Target');
set(leg,'Interpreter','latex');
set(gcf,'color','white');
alpha(0.1);