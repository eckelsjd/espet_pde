% Joshua Eckels
% 11/16/21
clear all;
close all;
clc;

addpath('../emitter');
addpath('..');

% Input geometry range
d_range = [1e-6 5e-3];      % emitter-to-extractor distance [m]
rc_range = [1e-5 1e-4];     % Radius of curvature (tip) [m]
alpha_range = [15 60];      % Emitter half angle [deg]
h_range = [1e-6 1e-2];      % Emitter height [m]
ra_range = [1e-5 1e-2];     % Radius of aperture [m]

d = 100e-6; 
rc = 50e-6; 
alpha = 35*(pi/180);
h = 500e-6; % 0.0015;  
ra = 300e-6;
mesh_size = 2e-6; % 0.00023 [m]
extractor_thickness = 76*10^(-6);
V0 = 1000; % [V]

% Simulation solution
emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness,V0,mesh_size);
[ex,ey,es,Ex,Ey] = EPOST.emitter_solution(emitter);
e_Emag = sqrt(Ex.^2 + Ey.^2);

% Simulated hyperboloid solution
% hyper = Hyperboloid(d,rc,alpha,h,ra,extractor_thickness,V0, mesh_size,1.5);
% [hx,hy,hs,hEx,hEy] = EPOST.emitter_solution(hyper);
% h_Emag = sqrt(hEx.^2 + hEy.^2);

% Martinez-Sanchez analytical solution
% [E_r, E_z] = EPOST.ms_solution(hyper);
H0 = 1/rc;
eta0 = sqrt( (H0*d)/(H0*d + 1) );
a = d * sqrt( (H0*d + 1)/(H0*d) );
r_coord = ex;
xi_coord = sqrt(1 + (r_coord/(a*sqrt(1-eta0^2))).^2);
e_premult = V0/(a*atanh(eta0));
r_premult = (eta0./(xi_coord.^2 - eta0^2)) .* sqrt((xi_coord.^2 - 1)/(1-eta0^2));
z_premult = xi_coord ./ (xi_coord.^2 - eta0^2);
E_r = e_premult*r_premult;
E_z = e_premult*(-z_premult);
z_coord = a*xi_coord*eta0;
ms_Emag = sqrt(E_r.^2 + E_z.^2);

figure()
plot(ex, e_Emag, '-r');
hold on
plot(r_coord, ms_Emag, '-k');
% plot(hx, h_Emag, '--b');
xlabel('x [m]','Interpreter','latex');
ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% leg = legend('Cone','M-S','M-S sim');
leg = legend('Cone with aperture simulation','Martinez-Sanchez analytical');
set(leg,'Interpreter','latex');

% figure()
% hold on
% plot(ex, ey, '--r');
% plot(hx, hy, '-b');
% axis equal;
% [xmin,xmax,ymin,ymax] = EPOST.get_geo_limits(emitter);
% xlim([xmin xmax]);
% ylim([ymin,ymax]);
% leg = legend('Cone','M-S');
% set(leg,'Interpreter','latex');
% set(leg,'Location','southeast');
% skip = 10;
% quiver(r_coord(1:skip:end,1),z_coord(1:skip:end,1),E_r(1:skip:end,1),E_z(1:skip:end,1),'b','LineWidth',1.2);
% quiver(emitter_x(1:skip:end,1),emitter_y(1:skip:end,1),E_x(1:skip:end,1),E_y(1:skip:end,1),'r','LineWidth',1.2);

EPOST.solplot(emitter);
% EPOST.solplot(hyper);