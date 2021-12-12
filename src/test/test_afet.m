% Joshua Eckels
% 12/8/21
clear all;
close all;
clc;
addpath('..\emitter');
sampler_input_fname = '..\data\afet.json'; 
fid = fopen(fname); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
val = jsondecode(str);

%% AFET-2 design
d = 0;
rc = 10*10^(-6);
alpha = 15*(pi/180);
h = 300*10^(-6); 
ra = 254*10^(-6);
extractor_thickness = 76*10^(-6);
V0 = 1000;

% mesh_size = 5e-6;
% emitter = Emitter(d,rc,alpha,h,ra,extractor_thickness, V0, mesh_size);
% [e_x,e_y,e_s,Ex,Ey] = EPOST.emitter_solution(emitter);
% e_Emag = sqrt(Ex.^2 + Ey.^2);
% 
% figure()
% plot(e_x, e_Emag,'-k');
% xlabel('X [m]','Interpreter','latex');
% ylabel('$|\vec{E}|$ [V/m]', 'Interpreter','latex');

% skip = 8;
% EPOST.solplot(emitter);
% EPOST.vectorplot(emitter,skip);
