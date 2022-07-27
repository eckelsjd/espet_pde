% Joshua Eckels
% 11/16/21
clear all;
close all;
clc;

addpath('../emitter');
addpath('../postproc');

% Input geometry range
d_range = [1e-6 5e-3];      % emitter-to-extractor distance [m]
rc_range = [1e-5 1e-4];     % Radius of curvature (tip) [m]
alpha_range = [15 60];      % Emitter half angle [deg]
h_range = [1e-6 1e-2];      % Emitter height [m]
ra_range = [1e-5 1e-2];     % Radius of aperture [m]

% d = 100e-6; 
% rc = 50e-6; 
% alpha = 35*(pi/180);
% h = 500e-6; % 0.0015;  
% ra = 300e-6;
% V0 = 1000; % [V]
ra = 5e-6;
d = 360e-6;
rc = 16e-6;
N = 2;
alpha = [14 30]*(pi/180);
h = 350e-6;
V0 = 100e3;
extractor_thickness = 76*10^(-6);

ms_emag = zeros(1,N);
sim_emag = zeros(1,N);
figure(1)
for i = 1:N
    % Simulation solution
    emitter = Emitter(d,rc,alpha(i),h,ra,extractor_thickness,V0);
    [ex,ey,es,Ex,Ey] = EPOST.emitter_solution(emitter);
    e_Emag = sqrt(Ex.^2 + Ey.^2);
    sim_emag(i) = max(e_Emag);
    
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
    ms_emag(i) = max(ms_Emag);

    figure(1)
    subplot(1, 2, i)
    xlabel('r')
    ylabel('z')
    plot(ex, ey, '-k')
    hold on;
    plot(r_coord, z_coord, '-r');
    legend('Emitter cone', 'MS hyperboloid')
end

%% plot stuff
figure()
plot(alpha*180/pi, sim_emag, '-ko');
hold on;
plot(alpha*180/pi, ms_emag, '-ro');
xlabel('Cone half-angle $\alpha$ [deg]','Interpreter','latex');
ylabel('Max E-field [V/m]', 'Interpreter' ,'latex');
set(gcf,'color','white');

% figure()
% plot(ex, e_Emag, '-r');
% hold on
% plot(r_coord, ms_Emag, '-k');
% % plot(hx, h_Emag, '--b');
% xlabel('x [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% % leg = legend('Cone','M-S','M-S sim');
% leg = legend('Cone simulation','Martinez-Sanchez analytical');
% set(leg,'Interpreter','latex');

% figure()
% hold on
% plot(ex, ey, '--r');
% % plot(hx, hy, '-b');
% plot(r_coord, z_coord, '-b');
% axis equal;
% [xmin,xmax,ymin,ymax] = EPOST.get_geo_limits(emitter);
% xlim([xmin xmax]);
% ylim([ymin,ymax]);
% leg = legend('Cone','MS');
% set(leg,'Interpreter','latex');
% set(leg,'Location','southeast');
% skip = 10;
% quiver(r_coord(1:skip:end),z_coord(1:skip:end),E_r(1:skip:end),E_z(1:skip:end),'b','LineWidth',1.2);
% quiver(ex(1:skip:end),ey(1:skip:end),Ex(1:skip:end),Ey(1:skip:end),'r','LineWidth',1.2);

% EPOST.solplot(emitter);
% EPOST.solplot(hyper);