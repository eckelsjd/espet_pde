clear all;
close all;
clc;

% d = 5e-3; % mean(d_range); % 3e-3
% rc = 3e-3;  % mean(rc_range);
% alpha = 43*(pi/180); % mean(alpha_range) * pi/180;
% h = 0.01432;  % mean(h_range); corresponds to rb=1e-2
% ra = 2e-3; % mean(ra_range);

d = 0.002;% 3e-3; % mean(d_range);
rc = 0.002;%2e-3;  % mean(rc_range);
alpha = 30*(pi/180); % mean(alpha_range) * pi/180;
h = 0.015;% 0.01432;  % mean(h_range); corresponds to rb=1e-2
ra = 0.004; % mean(ra_range);

mesh_size = 0.0003; % 0.00023 [m]
refine_factor = 4;

%% Test 2D axisymmetric
emitter = EmitterSim(d,rc,alpha,h,ra,2);
emitter.emitterMesh(mesh_size, refine_factor);
emitter.emitterSolve();
emitter.emitterPlot();

%% Test far field
% emitter = EmitterSim(d,rc,alpha,h,ra,0);
% emitter.emitterMesh(mesh_size, refine_factor);
% emitter.emitterSolve();
% emitter.emitterPlot();
% cfield = emitter.emitterSolution;
% emitter = EmitterSim(d,rc,alpha,h,ra,1);
% emitter.emitterMesh(mesh_size, refine_factor);
% emitter.emitterSolve();
% emitter.emitterPlot();
% ffield = emitter.emitterSolution;

% figure()
% subplot(1,2,1);
% c_mag = sqrt(cfield(:,4).^2 + cfield(:,5).^2);
% f_mag = sqrt(ffield(:,4).^2 + ffield(:,5).^2);
% plot(cfield(:,1),c_mag,'-b');
% hold on
% plot(ffield(:,1),f_mag,'-r');
% xlabel('X [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');
% legend('Close field','Far field');
% subplot(1,2,2);
% pd = 100*(abs(f_mag-c_mag)./c_mag);
% plot(cfield(:,1), pd,'-k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('Percent Difference','Interpreter','latex');

%% Test offset
% emitter = EmitterSim(d,rc,alpha,h,ra);
% emitter.emitterMesh(mesh_size, refine_factor);
% emitter.emitterSolve()
% n = emitter.emitterSolution(:,4:5);
% n_mag = sqrt(sum(n.^2,2));
% un = zeros(size(n));
% un(:,1) = n(:,1)./n_mag;
% un(:,2) = n(:,2)./n_mag;
% % step = emitter.emagmodel.Mesh.MinElementSize/3;
% step = 0.03*10^(-3);
% start_x = emitter.emitterSolution(:,1);
% start_y = emitter.emitterSolution(:,2);
% offset_x = start_x + step*un(:,1);
% offset_y = start_y + step*un(:,2);
% figure()
% Emag = sqrt(emitter.results.ElectricField.Ex.^2 + ...
%                 emitter.results.ElectricField.Ey.^2);
% pdeplot(emitter.emagmodel,'XYData',Emag');
% rgb_cmap = emitter.interp_cmap(emitter.cmap, 100);
% colormap(flipud(rgb_cmap));
% cb = colorbar();
% hold on;
% plot(start_x,start_y,'-r','LineWidth',1.5);
% plot(offset_x, offset_y, '-b','LineWidth',1.5);
% q_points = [offset_x, offset_y];
% nodes = emitter.emagmodel.Mesh.Nodes';
% emitter_nodes = findNodes(emitter.emagmodel.Mesh, 'region', 'Edge', [1 2]);
% nodes(emitter_nodes,:) = NaN;
% k = dsearchn(nodes, q_points);
% plot(nodes(k,1),nodes(k,2),'ok');
% plot(nodes(:,1),nodes(:,2),'.g');
% emitter.emitterPlot()