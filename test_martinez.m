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
h = 0.0015;  
ra = 0.000001; % mean(ra_range);

mesh_size = 0.00005; % 0.00023 [m]
refine_factor = 3;

% radii = linspace(1e-6, 5e-3, 15);
% plotColors = jet(length(radii));
% figure()
% set(gcf,'color','white');
% for ii = 1:length(radii)
%     emitter = EmitterSim(d,rc,alpha,h,radii(ii));
%     emitter.emitterMesh(mesh_size, refine_factor);
%     emitter.emitterSolve()
%     emitter_x = emitter.emitterSolution(:,1);
%     E_x = emitter.emitterSolution(:,4);
%     E_y = emitter.emitterSolution(:,5);
%     emitter_Emag = sqrt(E_x.^2 + E_y.^2);
%     plot(emitter_x, emitter_Emag,'Color', plotColors(ii,:));
%     hold on
% end

% Simulation solution
emitter = EmitterSim(d,rc,alpha,h,ra,2);
emitter.emitterMesh(mesh_size, refine_factor);
emitter.emitterSolve()
emitter_x = emitter.emitterSolution(:,1);
emitter_y = emitter.emitterSolution(:,2);
E_x = emitter.emitterSolution(:,4);
E_y = emitter.emitterSolution(:,5);
emitter_Emag = sqrt(E_x.^2 + E_y.^2);

% Martinez-Sanchez solution
V0 = emitter.emagmodel.BoundaryConditions.BCAssignments(1).Voltage;
H0 = 1/rc;
eta0 = sqrt( (H0*d)/(H0*d + 1) );
a = d * sqrt( (H0*d + 1)/(H0*d) );
r_coord = emitter_x;
xi_coord = sqrt(1 + (r_coord/(a*sqrt(1-eta0^2))).^2);
z_coord = a*eta0*xi_coord;
e_premult = V0/(a*atanh(eta0));
r_premult = (eta0./(xi_coord.^2 - eta0^2)) .* sqrt((xi_coord.^2 - 1)/(1-eta0^2));
z_premult = xi_coord ./ (xi_coord.^2 - eta0^2);
E_r = e_premult*r_premult;
E_z = e_premult*(-z_premult);
Emag = sqrt(E_r.^2 + E_z.^2);

% n = [E_x E_y];
% n_mag = sqrt(sum(n.^2,2));
% un = zeros(size(n));
% un(:,1) = n(:,1)./n_mag;
% un(:,2) = n(:,2)./n_mag;
% % step = emitter.emagmodel.Mesh.MinElementSize/3;
% step = 0.03*10^(-3);
% offset_x = emitter_x + step*un(:,1);
% offset_y = emitter_y + step*un(:,2);
% 
% q_points = [offset_x, offset_y];
% nodes = emitter.emagmodel.Mesh.Nodes';
% emitter_nodes = findNodes(emitter.emagmodel.Mesh, 'region', 'Edge', [1 2]);
% nodes(emitter_nodes,:) = NaN;
% k = dsearchn(nodes, q_points);
% 
% E_x = emitter.results.ElectricField.Ex(k,1);
% E_y = emitter.results.ElectricField.Ey(k,1);
% offset_Emag = sqrt(E_x.^2 + E_y.^2);

figure()
% subplot(1,2,1);
plot(r_coord, Emag, '-b');
hold on
plot(emitter_x, emitter_Emag, '-r');
xlabel('X [m]','Interpreter','latex');
ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
leg = legend('Martinez-Sanchez','Numerical');
set(leg,'Interpreter','latex');
% subplot(1,2,2);
% pd = 100*(abs(emitter_Emag-Emag)./Emag);
% plot(r_coord, pd,'-k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('Percent Difference','Interpreter','latex');


emitter.emitterPlot();

figure()
hold on
plot(r_coord, z_coord, '-b');
plot(emitter_x, emitter_y, '-r');
axis equal;
[xmin,xmax,ymin,ymax] = emitter.get_geo_limits();
xlim([xmin xmax]);
ylim([ymin,ymax]);
leg = legend('Martinez-Sanchez','Numerical');
set(leg,'Interpreter','latex');
set(leg,'Location','southeast');
skip = 10;
quiver(r_coord(1:skip:end,1),z_coord(1:skip:end,1),E_r(1:skip:end,1),E_z(1:skip:end,1),'b','LineWidth',1.2);
quiver(emitter_x(1:skip:end,1),emitter_y(1:skip:end,1),E_x(1:skip:end,1),E_y(1:skip:end,1),'r','LineWidth',1.2);