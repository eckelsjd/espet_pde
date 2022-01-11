clear all;
close all;
clc;

addpath('..\emitter')
addpath('..')
load('../../data/feasible/model.mat');
load('../../data/feasible/model_tr.mat');
f = importdata('../../data/feasible/samples/samples.txt');
nsamples = 1035;
params = min(f.data(1:nsamples,:));
d = params(1);
rc = params(2);
alpha = params(3)*(pi/180);
h = params(4);
ra = params(5);
extractor_thickness = 76*10^(-6);
V0 = 1000;

% Simulation prediction
N = 20;
dtheta = ((pi/2)-alpha)/N;
ds = rc*dtheta;
emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0,ds);
[x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);
t_mag = sqrt(Ex.^2 + Ey.^2);

% Network prediction
N = 200;
sd = linspace(0,1,N);
lchar = h;
in = zeros(5,N);
in(1,:) = d/lchar;
in(2,:) = rc/lchar;
in(3,:) = alpha;
in(4,:) = ra/lchar;
in(5,:) = sd;
ypred = net(in);
pred_mag = sqrt(ypred(1,:).^2 + ypred(2,:).^2)*(V0/lchar);

figure()
subplot(1,2,1);
plot(s/s(end),Ex,'-k');
hold on;
plot(sd,ypred(1,:)*(V0/lchar),'--r');
xlabel('Distance along emitter [\%]','Interpreter','latex');
ylabel('$E_x$','Interpreter','latex')
subplot(1,2,2);
plot(s/s(end),Ey,'-k');
hold on;
plot(sd,ypred(2,:)*(V0/lchar),'--r');
xlabel('Distance along emitter [\%]','Interpreter','latex');
ylabel('$E_y$','Interpreter','latex')

figure()
plot(s/s(end), t_mag,'-k');
hold on;
plot(sd,pred_mag,'--r');
xlabel('Distance along emitter [\%]','Interpreter','latex');
ylabel('Electric field magnitude $\left[ \frac{V}{m} \right]$','Interpreter','latex');
legend('Simulation','Surrogate');

% Parameter study on d
ds = linspace(min(f.data(1:nsamples,1)),max(f.data(1:nsamples,1)),N);
[xg,yg] = meshgrid(sd,ds);
pts = [xg(:) yg(:)]';
in = zeros(5,size(pts,2));
in(1,:) = pts(2,:)/lchar;
in(2,:) = rc/lchar;
in(3,:) = alpha;
in(4,:) = ra/lchar;
in(5,:) = pts(1,:);
ypred = net(in);
pred_mag = sqrt(ypred(1,:).^2 + ypred(2,:).^2)*(V0/lchar);
pred_grid = reshape(pred_mag,size(xg));
figure()
h = surf(xg,yg,pred_grid);
set(h,'edgecolor','none');
colormap hot
xlabel('Distance along emitter (s)');
ylabel('Tip-to-extractor distance (d)');
zlabel('Electric field magnitude (V/m)');

%% Plot training results
figure()
semilogy(tr.epoch,tr.perf,'-r','LineWidth',3);
hold on;
semilogy(tr.epoch,tr.vperf,'-b','LineWidth',1.5);
semilogy(tr.epoch,tr.tperf,'-g','LineWidth',1);
leg = legend('Training','Validation','Test');
set(leg,'Interpreter','latex');
xlabel('Epoch','interpreter','latex');
ylabel('Mean squared-error (MSE)','interpreter','latex');
set(gcf,'color','white');
xlim([0 500]);

%% Old
% N1 = 100;
% N2 = 40;
% x1 = linspace(0,10,N1);
% x2 = linspace(0,10,N2);
% y1 = x1.^2;
% y2 = x2.^2 + 5*randn(1,N2);
% net = feedforwardnet(10);
% net = train(net,x2,y2);

% y1p = net(x2);
% figure()
% plot(x1,y1,'-k');
% hold on;
% plot(x2,y1p,'--r');
% 
% net2 = train(net,x1,y1);
% y2p = net(x2);
% plot(x2,y2p,'--b');

% N = 50;
% x1 = linspace(-5,5,N);
% x2 = linspace(-2,2,N/2);
% [x1g,x2g] = meshgrid(x1,x2);
% data = [x1g(:), x2g(:)]';
% % y1 = x.^2 + 5*randn(1,N);
% y1g = x1g.^2 + x2g.^2;
% y2g = x1g + x2g;
% yt = [y1g(:), y2g(:)]';
% 
% net = feedforwardnet(10);
% net = train(net,data,yt);
% ypred = net(data);
% 
% y1p = reshape(ypred(1,:),size(y1g));
% y2p = reshape(ypred(2,:),size(y2g));
% 
% % yp = reshape(ypred,size(yg));
% % 
% figure()
% subplot(1,2,1);
% surf(x1g,x2g,y1g,'FaceAlpha',0.5);
% subplot(1,2,2);
% surf(x1g,x2g,y1p);
% 
% figure()
% subplot(1,2,1);
% surf(x1g,x2g,y2g,'FaceAlpha',0.5);
% subplot(1,2,2);
% surf(x1g,x2g,y2p);
% 
% % figure()
% % hold on;
% % plot(x,yt,'-k');
% % plot(x,y,'--r');
% 
