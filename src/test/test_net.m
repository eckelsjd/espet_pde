clear all;
close all;
clc;

N1 = 100;
N2 = 40;
x1 = linspace(0,10,N1);
x2 = linspace(0,10,N2);
y1 = x1.^2;
y2 = x2.^2 + 5*randn(1,N2);
net = feedforwardnet(10);
net = train(net,x2,y2);

y1p = net(x2);
figure()
plot(x1,y1,'-k');
hold on;
plot(x2,y1p,'--r');

net2 = train(net,x1,y1);
y2p = net(x2);
plot(x2,y2p,'--b');

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
