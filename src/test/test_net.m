%% Load data
clear all;
close all;
clc;

addpath('../emitter')
addpath('..')
load('../../data/feasible/models/model_dffnet.mat');
% load('../../data/feasible/model_tr.mat');
f = importdata('../../data/feasible/samples/samples.txt');

%% Test on mean design
params = mean(f.data,1);
d = params(1);
rc = params(2);
alpha = params(3)*(pi/180);
h = params(4);
ra = params(5);
extractor_thickness = 76*10^(-6);
V0 = 1000;

d = 3e-6; rc = 1.5e-5; alpha = 2.678e-1; h = 3.018e-4; ra = 2.486e-4;

% Simulation prediction
emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0);
[x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);
t_mag = sqrt(Ex.^2 + Ey.^2);

% Network prediction
N = 200;
sd = linspace(0,1,N);
xpred = [repmat([d/h; rc/h; alpha; ra/h],1,N); sd];
ypred = net(xpred);
Ex_pred = ypred(1,:)*(V0/h);
Ey_pred = ypred(2,:)*(V0/h);
test_mag = sqrt(Ex_pred.^2 + Ey_pred.^2);

figure(1)
subplot(1,2,1);
plot(s/s(end),Ex,'-k');
hold on;
plot(sd,Ex_pred,'--r');
xlabel('Distance along emitter [\%]','Interpreter','latex');
ylabel('$E_x$','Interpreter','latex')
legend('Truth','Surrogate')
subplot(1,2,2);
plot(s/s(end),Ey,'-k');
hold on;
plot(sd,Ey_pred,'--r');
xlabel('Distance along emitter [\%]','Interpreter','latex');
ylabel('$E_y$','Interpreter','latex')
legend('Truth','Surrogate')

figure(2)
plot(s/s(end), t_mag,'-k');
hold on;
plot(sd,test_mag,'--r');
xlabel('Distance along emitter [\%]','Interpreter','latex');
ylabel('Electric field magnitude $\left[ \frac{V}{m} \right]$','Interpreter','latex');
legend('Simulation','Surrogate');

%% Plot training results
figure(3)
semilogy(perf(1,:),'-r','LineWidth',3);
hold on;
semilogy(perf(2,:),'-b','LineWidth',1.5);
semilogy(perf(3,:),'-g','LineWidth',1);
leg = legend('Training','Validation','Test');
set(leg,'Interpreter','latex');
xlabel('Iteration','interpreter','latex');
ylabel('Mean squared-error (MSE)','interpreter','latex');
set(gcf,'color','white');

%% Loop over and view test set
% View sampling distribution
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

% Get unique geometry test samples (by column, only rows 1-4)
[xunique, ind, ~] = uniquetol(xtest(1:4,:)',1e-6,'ByRows',true);
xtest = xunique';
ytest = ytest(:,ind);

% Down-select
xtest = xtest(:,1:200:end);
ytest = ytest(:,1:200:end);


%%% COMMENT/UNCOMMENT to Use separate test files instead
fd = importdata('../../data/feasible/test/samples/samples.txt');
hvec = fd.data(:,4);
xtest = [fd.data(:,1:2), fd.data(:,3)*(pi/180), fd.data(:,5)]';

figure(5)
set(gcf,'Position',[700 800 600 500])

for i = 1:length(xtest)
    
    params = xtest(:,i); % use the mean height h
%     d = params(1)*h;
%     rc = params(2)*h;
%     alpha = params(3);
%     ra = params(4)*h;
    d = params(1);
    rc = params(2);
    alpha = params(3);
    ra = params(4);
    h = hvec(i);
    
    emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0);
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);
    t_mag = sqrt(Ex.^2 + Ey.^2);

    xpred = [repmat([d/h; rc/h; alpha; ra/h],1,N); sd];
    ypred = net(xpred);
    Ex_pred = ypred(1,:)*(V0/h);
    Ey_pred = ypred(2,:)*(V0/h);
    test_mag = sqrt(Ex_pred.^2 + Ey_pred.^2);

    fprintf('Case = %d d = %.3E rc = %.3E alpha = %.2f ra = %.3E\n',i, d, rc, alpha*(180/pi), ra);
    figure(5)
    subplot(1,2,1);
    plot(s/s(end),Ex,'-k');
    hold on;
    plot(sd,Ex_pred,'--r');
    hold off;
    xlabel('Distance along emitter [\%]','Interpreter','latex');
    ylabel('$E_x$','Interpreter','latex')
    subplot(1,2,2);
    plot(s/s(end),Ey,'-k');
    hold on;
    plot(sd,Ey_pred,'--r');
    xlabel('Distance along emitter [\%]','Interpreter','latex');
    ylabel('$E_y$','Interpreter','latex')
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

%% Test scaling (same dimensionless inputs, different geometry)
des1 = [100e-6, 25e-6, 0.8, 250e-6, 125e-6]; % [d,rc,alpha (rad), h, ra];
des2 = [200e-6, 50e-6, 0.8, 500e-6, 250e-6];
des3 = [129e-6, 32.25e-6, 0.8, 322.5e-6, 161.25e-6];

emitter1 = Emitter(des1(1),des1(2),des1(3),des1(4),des1(5),extractor_thickness,V0);
[~,~,s1,Ex1,Ey1] = EPOST.emitter_solution(emitter1);
emitter2 = Emitter(des2(1),des2(2),des2(3),des2(4),des2(5),extractor_thickness,V0);
[~,~,s2,Ex2,Ey2] = EPOST.emitter_solution(emitter2);
emitter3 = Emitter(des3(1),des3(2),des3(3),des3(4),des3(5),extractor_thickness,V0);
[~,~,s3,Ex3,Ey3] = EPOST.emitter_solution(emitter3);

xparam = [0.4; 0.1; 0.8; 0.5];
xpred = [repmat(xparam,1,N); sd];
ypred = net(xpred);
Ex_pred = ypred(1,:);
Ey_pred = ypred(2,:);

figure()
subplot(3,2,1)
plot(s1/s1(end),Ex1,'-k');
hold on;
plot(sd,Ex_pred*(V0/des1(4)),'--r');
xlabel('s [\%]','Interpreter','latex');
ylabel('Design 1: $E_x$','Interpreter','latex')
subplot(3,2,2);
plot(s1/s1(end),Ey1,'-k');
hold on;
plot(sd,Ey_pred*(V0/des1(4)),'--r');
xlabel('s [\%]','Interpreter','latex');
ylabel('$E_y$','Interpreter','latex')

subplot(3,2,3)
plot(s2/s2(end),Ex2,'-k');
hold on;
plot(sd,Ex_pred*(V0/des2(4)),'--r');
xlabel('s [\%]','Interpreter','latex');
ylabel('Design 2: $E_x$','Interpreter','latex')
subplot(3,2,4);
plot(s2/s2(end),Ey2,'-k');
hold on;
plot(sd,Ey_pred*(V0/des2(4)),'--r');
xlabel('s [\%]','Interpreter','latex');
ylabel('$E_y$','Interpreter','latex')

subplot(3,2,5)
plot(s3/s3(end),Ex3,'-k');
hold on;
plot(sd,Ex_pred*(V0/des3(4)),'--r');
xlabel('s [\%]','Interpreter','latex');
ylabel('Design 3: $E_x$','Interpreter','latex')
subplot(3,2,6);
plot(s3/s3(end),Ey3,'-k');
hold on;
plot(sd,Ey_pred*(V0/des3(4)),'--r');
xlabel('s [\%]','Interpreter','latex');
ylabel('$E_y$','Interpreter','latex')

%% Parameter study on d
% ds = linspace(min(f.data(1:nsamples,1)),max(f.data(1:nsamples,1)),N);
% [xg,yg] = meshgrid(sd,ds);
% pts = [xg(:) yg(:)]';
% in = zeros(5,size(pts,2));
% in(1,:) = pts(2,:)/lchar;
% in(2,:) = rc/lchar;
% in(3,:) = alpha;
% in(4,:) = ra/lchar;
% in(5,:) = pts(1,:);
% ytest = net(in);
% test_mag = sqrt(ytest(1,:).^2 + ytest(2,:).^2)*(V0/lchar);
% pred_grid = reshape(test_mag,size(xg));
% figure()
% h = surf(xg,yg,pred_grid);
% set(h,'edgecolor','none');
% colormap hot
% xlabel('Distance along emitter (s)');
% ylabel('Tip-to-extractor distance (d)');
% zlabel('Electric field magnitude (V/m)');

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
