%% run_sims.m
% 12/10/21
clear all;
close all;
clc;
dataset = 'geometry'; % base/feasible/geometry dataset
addpath('../emitter/');
addpath('../postproc/')
data_dir = '../../data';
samples_file = fullfile(data_dir, dataset,'samples','samples.txt');
json_input = fullfile(data_dir, dataset, 'sampler_input.json');
prefix = 'esim';

%% Data import
% Get simulation parameters (extractor thickness, bias voltage)
fd = fopen(json_input); 
raw = fread(fd,inf); 
json_str = char(raw'); 
fclose(fd); 
json_obj = jsondecode(json_str);
te = json_obj.sim_params.ExtractorThickness;
V0 = json_obj.sim_params.BiasVoltage;

% Get geometry samples
% [d, rc, alpha, h, ra]
datastruct = importdata(samples_file);
samples = datastruct.data;
[nsamples, nparams] = size(samples);

% Store bad simulation parameters
% bad_mesh = [];
% bad_edges = [];

%% Run simulations
% tic
parfor ii = 1:nsamples
    if mod(ii, 100) == 0
        % toc
        fprintf('Iteration: %i\n',ii)
        % tic
    end
    curr_geo = samples(ii,:);
    d = curr_geo(1); rc = curr_geo(2); alpha = curr_geo(3)*(pi/180);
    h = curr_geo(4); ra = curr_geo(5);

    % Skip if we already have this data
    fname = EPOST.get_filename(d, rc, curr_geo(3), h, ra, prefix);
    fulldir = fullfile(data_dir, dataset, 'sims',fname);
    if isfile(fulldir)
        continue
    end

    try
        emitter = Emitter(d,rc,alpha,h,ra,te,V0);
        parsave(fulldir, emitter);
    catch ME
        fprintf('Caught an ME error\n')
        % Save parameters that caused bad simulations
%         switch ME.identifier
%             case 'EmitterSim:badMesh'
%                 bad_mesh = [bad_mesh; [d, rc, curr_geo(3), h, ra]];
%             case 'pde:pdeModel:BadGeomIntersectingEdges'
%                 bad_edges = [bad_edges; [d, rc, curr_geo(3), h, ra]];
%             otherwise
%                 rethrow(ME)
%         end
    end
end

% Save the failed simulation parameters
% bs_file = fullfile(data_dir, dataset, 'samples', 'bad_mesh_samples.txt');
% fd = fopen(bs_file, 'wt');
% fprintf(fd, 'd rc alpha h ra\n');
% fclose(fd);
% writematrix(bad_mesh, bs_file, 'Delimiter', ' ', 'WriteMode', 'append');
% 
% bs_file = fullfile(data_dir, dataset, 'samples', 'bad_edges_samples.txt');
% fd = fopen(bs_file, 'wt');
% fprintf(fd, 'd rc alpha h ra\n');
% fclose(fd);
% writematrix(bad_edges, bs_file, 'Delimiter', ' ', 'WriteMode', 'append');

function parsave(fname, emitter)
    save(fname, 'emitter');
end
