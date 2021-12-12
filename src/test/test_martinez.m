% Joshua Eckels
% 11/16/21
clear all;
close all;
clc;

% Input geometry range
d_range = [1e-6 5e-3];      % emitter-to-extractor distance [m]
rc_range = [1e-5 1e-4];     % Radius of curvature (tip) [m]
alpha_range = [15 60];      % Emitter half angle [deg]
h_range = [1e-6 1e-2];      % Emitter height [m]
ra_range = [1e-5 1e-2];     % Radius of aperture [m]

d = 0.0005; 
rc = 0.00015; 
alpha = 35*(pi/180);
h = 0.0015; % 0.0015;  
ra = 0.000001;
mesh_size = 0.00005; % 0.00023 [m]
extractor_thickness = 0.001; % [m]

% Simulation solution
emitter = EmitterMS(d,rc,alpha,h,ra,extractor_thickness,mesh_size);
[ex,ey,es,Ex,Ey] = EPOST.emitter_solution(emitter);
e_Emag = sqrt(Ex.^2 + Ey.^2);

% Simulated hyperboloid solution
hyper = Hyperboloid(d,rc,alpha,h,ra,extractor_thickness,mesh_size);
[hx,hy,hs,hEx,hEy] = EPOST.emitter_solution(hyper);
h_Emag = sqrt(hEx.^2 + hEy.^2);

% Martinez-Sanchez analytical solution
[E_r, E_z] = EPOST.ms_solution(hyper);
ms_Emag = sqrt(E_r.^2 + E_z.^2);

figure()
plot(ex, e_Emag, '--r');
hold on
plot(hx, ms_Emag, '-b');
plot(hx, h_Emag, '--b');
xlabel('X [m]','Interpreter','latex');
ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
leg = legend('Cone','M-S','M-S sim');
set(leg,'Interpreter','latex');

figure()
hold on
plot(ex, ey, '--r');
plot(hx, hy, '-b');
axis equal;
[xmin,xmax,ymin,ymax] = EPOST.get_geo_limits(emitter);
xlim([xmin xmax]);
ylim([ymin,ymax]);
leg = legend('Cone','M-S');
set(leg,'Interpreter','latex');
set(leg,'Location','southeast');
% skip = 10;
% quiver(r_coord(1:skip:end,1),z_coord(1:skip:end,1),E_r(1:skip:end,1),E_z(1:skip:end,1),'b','LineWidth',1.2);
% quiver(emitter_x(1:skip:end,1),emitter_y(1:skip:end,1),E_x(1:skip:end,1),E_y(1:skip:end,1),'r','LineWidth',1.2);

EPOST.solplot(emitter);
EPOST.solplot(hyper);