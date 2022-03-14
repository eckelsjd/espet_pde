%% run_sims.m
% 12/10/21
clear all;
close all;
clc;
dataset = 'base'; % base or feasible dataset
addpath('../emitter/');
data_dir = '../../data';
samples_file = fullfile(data_dir, dataset,'samples','samples.txt');
json_input = fullfile(data_dir, dataset, 'sampler_input.json');
prefix = 'esim';

%% Data import
% Get simulation parameters (extractor thickness, bias voltage, mesh size)
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

%% Run simulations
for ii = 1:nsamples
    if mod(ii, 100) == 0
        fprintf('Iteration: %i\n',ii)
    end
    curr_geo = samples(ii,:);
    d = curr_geo(1); rc = curr_geo(2); alpha = curr_geo(3)*(pi/180);
    h = curr_geo(4); ra = curr_geo(5);

    emitter = Emitter(d,rc,alpha,h,ra,te,V0);

%     EPOST.solplot(emitter)
%     waitforbuttonpress
%     close all;

    fname = EPOST.get_filename(d, rc, curr_geo(3), h, ra, prefix);
    fulldir = fullfile(data_dir, dataset, 'sims',fname);
    save(fulldir,'emitter');
end
