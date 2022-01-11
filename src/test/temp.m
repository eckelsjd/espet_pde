clear all;
close all;
clc;

N = 1000;
mu = [0.5 0.5];
cov = [0.6 0.3; 0.3 0.5];

R = mvnrnd(mu,cov, N);

figure()
plot(R(:,1),R(:,2),'.k','MarkerSize',8);
xlabel('$E_x$ [V/m]','Interpreter','latex');
ylabel('$E_y$ [V/m]', 'Interpreter','latex');