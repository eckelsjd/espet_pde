%% Generate training/test data
% Joshua Eckels
% 12/11/21
clear all;
close all;
clc;
prefix = 'train';  % test or train
dataset = 'geometry'; % base or feasible dataset
addpath('../emitter/');
addpath('../postproc/');
data_dir = '../../data';
files = dir(fullfile(data_dir, dataset, 'sims'));

% Curve discretization of training data
% A = 1.8; b=1; c=-1;
% [~, yg] = FGSR.compute_grid(A,b,c,0,1);

%% Save in most general-use format (train_base_cell.mat) (cell array)
% Allocate space
idxs = [];
for ii = 1:length(files)
    if files(ii).name(1) == '.'
        idxs = [idxs, ii];
    end
end
files(idxs) = [];
Nsamples = length(files);
xtrain = cell(1,Nsamples);
ytrain = cell(1, Nsamples);
V0 = cell(1, Nsamples);
fprintf("starting samples: %d\n",Nsamples);

% Loop over all simulation files
bad_idxs = zeros(1, Nsamples);
parfor ii = 1:Nsamples
     % load the emitter simulation
    fprintf("Iteration: %d\n",ii);
    file = fullfile(files(ii).folder,files(ii).name);
    try
        fd = load(file);
    catch ME
        fprintf("Caught an err in file: %s\n", file)
        bad_idxs(ii) = 1;
        continue
    end
    emitter = fd.emitter;
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);
    
    xtrain{ii} = [emitter.d; emitter.rc; emitter.alpha; emitter.h; emitter.ra];
    ytrain{ii} = [s; Ex; Ey];
    V0{ii} = emitter.V0;
end

% Get rid of corrupted files
idxs = find(bad_idxs==1);
xtrain(idxs) = [];
ytrain(idxs) = [];
fprintf("Remaining samples: %d\n",length(xtrain));
% Save
cell_file = sprintf('%s_%s_cell.mat', prefix, dataset);
save(fullfile(data_dir, dataset, prefix, cell_file),'xtrain','ytrain', 'V0');

%% Save in dimensionless feedforward net format (train_dffnet.mat)
% Loop over all training data
% file = fullfile(data_dir, dataset, prefix, cell_file);
% load(file); % gives xtrain, ytrain, and V0
% Nsamples = length(xtrain);
% xdata = zeros(5,1);
% ydata = zeros(2,1);

% for ii = 1:Nsamples
%     % Get number of discrete points for each sample
%     ycurr = ytrain(:,:,ii);
%     ycurr = ycurr(~isnan(ycurr)); % ycurr = [s, Ex, Ey]
%     Ncurr = length(ycurr)/3;
%     ycurr = reshape(ycurr,[Ncurr,3]);
%     s = ycurr(:,1)/ycurr(end,1);
%     Ex = ycurr(:,2); Ey = ycurr(:,3);
% 
%     % Nondimensionalize
%     h = xtrain(4,1,ii);
%     x = [xtrain(1:3,:,ii) ./ [h;h;1]; xtrain(5,:,ii) / h];
%     x = [repmat(x,1,Ncurr); s'];
%     y = [Ex'; Ey'] / (V0/h);
% 
%     % Concatenate training data into big array
%     xdata = [xdata, x];
%     ydata = [ydata, y];
% end
% xdata = xdata(:,2:end);
% ydata = ydata(:,2:end);
% save(fullfile(data_dir, dataset, prefix, sprintf('%s_dffnet.mat', prefix)),"xdata","ydata");

%% Save in Emax dimensionless feedforward net format (train_dffnet_max.mat)
% Loop over all training data
file = fullfile(data_dir, dataset, prefix, cell_file);
load(file); % gives xtrain, ytrain, and V0
Nsamples = length(xtrain);
xdata = zeros(5,Nsamples);
ydata = zeros(1,Nsamples);

parfor ii = 1:Nsamples
    % Get number of discrete points for each sample
    fprintf("Iteration %d\n",ii)
    ycurr = ytrain{ii};
    Ex = ycurr(2,:); Ey = ycurr(3,:);
    Emag = sqrt(Ex.^2 + Ey.^2);
    [Emax, idx] = max(Emag);
%     Ex_max = Ex(idx);
%     Ey_max = Ey(idx);

    % Nondimensionalize
%     h = xcurr(4);
%     x = [xcurr(1:3) ./ [h;h;1]; xcurr(5) / h];
%     y = Emax / (V0/h);

    % Save training data to array
    xcurr = xtrain{ii};
    xdata(:,ii) = xcurr;
    ydata(ii) = Emax;
end
save(fullfile(data_dir, dataset, prefix, sprintf('%s_dffnet_max.mat', prefix)),"xdata","ydata","V0");
