% Joshua Eckels
% 12/8/21
clear all;
close all;
clc;
addpath('..\emitter')
addpath('..\postproc')

%% AFET-2 design
d = 5*10^(-6);
rc = 10*10^(-6);
alpha = 15*(pi/180);
h = 100*10^(-6); 
ra = 254*10^(-6);
extractor_thickness = 76*10^(-6);
V0 = 1000;

%% Feasible design
% d = 50E-06;
% rc = 10E-06;
% alpha = 30*(pi/180);
% h = 200E-06;
% ra = 80E-06;
% extractor_thickness = 10*10^(-6);
% V0 = 1000;

% Estimate mesh size
% N = 20;
% dtheta = ((pi/2)-alpha)/N;
% ds = rc*dtheta;

%% Test mesh convergence
% Reference solution
% mesh_ref = 5*10^(-3);
% hyper_ref = Hyperboloid(d,rc,alpha,h,ra,extractor_thickness,mesh_ref);
emitter_ref = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0);
% load afet_ref_emitter.mat
% [xref, yref, sref, Exref, Eyref] = EPOST.emitter_solution(emitter_ref);
% emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0,2e-06);
% [ex, ey, es, eEx,eEy] = EPOST.emitter_solution(emitter);
% e_sol = sqrt(eEx.^2 + eEy.^2);
xpoints = 0;
[ypoints, Ex, Ey] = EPOST.ms_solution(rc, d, V0, xpoints);
ref_Emag = sqrt(Ex.^2 + Ey.^2);
% ref_sol = sqrt(Exref.^2 + Eyref.^2);

% Simulation
% ref_sol = smooth(ref_sol);
% ms = [5e-7, 8e-7, 1e-6, 2e-6, 3e-6, 7e-6, 1e-5, 5e-5];
ms = [1e-7, 3e-7, 5e-7, 8e-7, 2e-6, 3e-6, 5e-6, 1e-5];
% ms = [5e-7];
plotColors = jet(length(ms));
% rmse = zeros(size(ms));
etip = zeros(size(ms));
figure()
set(gcf,'color','white');
for ii = 1:length(ms)
    e = Hyperboloid(d,rc,alpha,h,ra,extractor_thickness,V0, ms(ii));
    [hx,hy,hs,hEx,hEy] = EPOST.emitter_solution(e);
    h_Emag = sqrt(hEx.^2 + hEy.^2);
    etip(ii) = h_Emag(1);
%     ref_interp = interp1(xref, ref_sol, hx);
%     rmse(ii) = sqrt(mean((h_Emag - ref_interp).^2));
%     plot(hx, h_Emag,'Color', plotColors(ii,:));
%     plot(hx, h_Emag,'--r');
%     hold on
end
% plot(r, ref_Emag, '-k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% legend('Simulation','Analytical')
% EPOST.solplot(emitter_ref);
% EPOST.solplot(e);

% figure()
% semilogx(ms, rmse, '-ok');
% xlabel('Mesh size [m]','Interpreter','latex')
% ylabel('RMSE with reference simulation', 'Interpreter','latex');

figure()
semilogx(ms, etip,'ok');
hold on;
yline(ref_Emag,'-r');
yline(ref_Emag*0.99,'--r');
yline(ref_Emag*1.01,'--r');
yline(ref_Emag*0.95,'--b');
yline(ref_Emag*1.05,'--b');
xlabel('Mesh size [m]','Interpreter','latex')
ylabel('$|\vec{E}|_{tip}$ [V/m]','Interpreter','latex');
leg = legend('Simulation', 'Analytical','$1\%$ bound','','$5\%$ bound');
set(leg,'Interpreter','latex');
ylim([etip(end)*0.95, ref_Emag*1.2])
xlim([ms(1)*0.97, ms(end)*1.03]);

% 1% Converged mesh
% mesh_size = 5e-7;
% hyper = Hyperboloid(d,rc,alpha,h,ra,extractor_thickness,mesh_size);
% [hx,hy,hs,hEx,hEy] = EPOST.emitter_solution(hyper);
% h_Emag = sqrt(hEx.^2 + hEy.^2);
% figure()
% plot(hx, h_Emag,'--r');
% hold on;
% plot(xref, ref_Emag, '-k');
% plot(xref, ref_sol, '--k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% EPOST.solplot(hyper);

%% Test cone height convergence
% mesh_size = 5e-7;
% hss = linspace(50e-6,300e-6,5);
% plotColors = jet(length(hss));
% etip = zeros(size(hss));
% figure()
% set(gcf,'color','white');
% for ii = 1:length(hss)
%     hyper = Hyperboloid(d,rc,alpha,hss(ii),ra,extractor_thickness,mesh_size);
%     [hx,hy,hs,hEx,hEy] = EPOST.emitter_solution(hyper);
%     h_Emag = sqrt(hEx.^2 + hEy.^2);
%     plot(hx, h_Emag,'Color', plotColors(ii,:));
%     etip(ii) = h_Emag(1);
%     hold on
% end
% plot(xref, ref_Emag, '-k');
% plot(xref, ref_sol, '--k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% 
% figure()
% plot(hss, etip,'ok');
% hold on;
% yline(ref_Emag(1),'-r');
% yline(ref_Emag(1)*0.99,'--r');
% yline(ref_Emag(1)*1.01,'--r');
% yline(ref_Emag(1)*0.95,'--b');
% yline(ref_Emag(1)*1.05,'--b');
% xlabel('Cone height [m]','Interpreter','latex')
% ylabel('$E_{tip}$ [V/m]','Interpreter','latex');
% legend('Simulation', 'Analytical','1% bound');

%% Test right boundary condition convergence
% mesh_size = 5e-7;
% right_sf = linspace(1.1,2,10); % right bc location scale factor
% plotColors = jet(length(right_sf));
% etip = zeros(size(right_sf));
% mse = zeros(size(right_sf));
% figure()
% set(gcf,'color','white');
% for ii = 1:length(right_sf)
%     hyper = Hyperboloid(d,rc,alpha,h,ra,extractor_thickness,mesh_size,right_sf(ii));
%     [hx,hy,hs,hEx,hEy] = EPOST.emitter_solution(hyper);
%     h_Emag = sqrt(hEx.^2 + hEy.^2);
%     ref_eq = interp1(xref,ref_sol,hx);
%     mse(ii) = mean((h_Emag - ref_eq).^2);
%     plot(hx, h_Emag,'Color', plotColors(ii,:));
%     etip(ii) = h_Emag(1);
%     hold on
% end
% plot(xref, ref_Emag, '-k');
% plot(xref, ref_sol, '--k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% 
% figure()
% plot(right_sf, etip,'ok');
% hold on;
% yline(ref_Emag(1),'-r');
% yline(ref_Emag(1)*0.99,'--r');
% yline(ref_Emag(1)*1.01,'--r');
% yline(ref_Emag(1)*0.95,'--b');
% yline(ref_Emag(1)*1.05,'--b');
% xlabel('Right BC scale factor','Interpreter','latex')
% ylabel('$E_{tip}$ [V/m]','Interpreter','latex');
% legend('Simulation', 'Analytical','1% bound');
% 
% figure()
% plot(right_sf, sqrt(mse), '-ok');
% xlabel('Right BC scale factor','Interpreter','latex')
% ylabel('RMSE with reference simulation', 'Interpreter','latex');

%% Summary
% mesh_size = 5e-7;
% right_sf = 1.5;
% hyper = Hyperboloid(d,rc,alpha,h,ra,extractor_thickness,mesh_size,right_sf);
% [hx,hy,hs,hEx,hEy] = EPOST.emitter_solution(hyper);
% h_Emag = sqrt(hEx.^2 + hEy.^2);
% plot(hx, h_Emag,'--r','LineWidth',2);
% hold on;
% plot(xref, ref_sol, '--k');
% plot(xref, ref_Emag, '-k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% legend('Converged sim', 'Reference sim', 'Analytical');
% EPOST.solplot(hyper);
