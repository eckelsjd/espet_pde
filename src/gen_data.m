%% Generate training data
% Joshua Eckels
% 12/11/21
clear all;
close all;
clc;
addpath('.\emitter\');
data_dir = '..\data\feasible\';
files = dir(fullfile(data_dir,'sims'));
prefix = 'train';

% Loop over all simulation files
for ii = 1:length(files)
    if files(ii).isdir
        continue
    end
    file = fullfile(files(ii).folder,files(ii).name);
    load(file); % loads the emitter simulation
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);

    % Allocate space for training data
    % Rows 1-5: (Input) d,rc,alpha,ra,s
    % Row 6-7: (Output) Ex, Ey
    train_data = zeros(7,length(x));

    % Non-dimensionalize
    lchar = emitter.h;  % use the emitter height as characteristic length
    train_data(1,:) = emitter.d/lchar;
    train_data(2,:) = emitter.rc/lchar;
    train_data(3,:) = emitter.alpha;
    train_data(4,:) = emitter.ra/lchar;
    train_data(5,:) = s/s(end);
    train_data(6,:) = Ex/(emitter.V0/lchar);
    train_data(7,:) = Ey/(emitter.V0/lchar);

    % Save 
    fname = EPOST.get_filename(emitter.d, emitter.rc, emitter.alpha*(180/pi),...
        emitter.h, emitter.ra, prefix);
    fname = fullfile(data_dir, prefix, fname);
    save(fname,'train_data');
end