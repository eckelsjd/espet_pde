%% Generate training data
% Joshua Eckels
% 12/11/21
clear all;
close all;
clc;
addpath('./emitter/');
data_dir = '../data/base/';
files = dir(fullfile(data_dir,'sims'));
prefix = 'train';

% Curve discretization of training data
% A = 1.8; b=1; c=-1;
% [~, yg] = FGSR.compute_grid(A,b,c,0,1);

%% Save in most general-use format (train_base_cell.mat) (cell array)
% Allocate space
Nsamples = length(files) - 2;
xtrain = cell(1,Nsamples);
ytrain = cell(1, Nsamples);

% Loop over all simulation files
jj = 1;
for ii = 1:length(files)
    if files(ii).isdir
        continue
    end

     % load the emitter simulation
    file = fullfile(files(ii).folder,files(ii).name);
    load(file);
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);
    
    xtrain{jj} = [emitter.d; emitter.rc; emitter.alpha; emitter.h; emitter.ra];
    ytrain{jj} = [s; Ex; Ey];
    jj = jj + 1;
end

% Save
V0 = emitter.V0;
save(fullfile(data_dir, prefix, 'train_base_cell.mat'),'xtrain','ytrain', 'V0');

%% Save in most general-use format (train_base.mat)
% Nmax = 571; % 345
% Find largest s dimension
% for ii = 1:length(files)
%     if files(ii).isdir
%         continue
%     end
%     file = fullfile(files(ii).folder,files(ii).name);
%     load(file); % loads the emitter simulation
%     [~,~,s,~,~] = EPOST.emitter_solution(emitter);
%     Ncurr = length(s);
% 
%     if Ncurr > Nmax
%         Nmax = Ncurr;
%     end
% end

% Allocate space
Nsamples = length(files) - 2;
xtrain = zeros(5,1,Nsamples);
ytrain = NaN(Nmax, 3, Nsamples);

% Loop over all simulation files
jj = 1;
for ii = 1:length(files)
    if files(ii).isdir
        continue
    end

     % load the emitter simulation
    file = fullfile(files(ii).folder,files(ii).name);
    load(file);
    [x,y,s,Ex,Ey] = EPOST.emitter_solution(emitter);
%     Ex_interp = interp1(s,Ex,yg,'pchip');
%     Ey_interp = interp1(s,Ey,yg,'pchip');
    
    Ncurr = length(s);
    xtrain(:,:,jj) = [emitter.d; emitter.rc; emitter.alpha; emitter.h; emitter.ra];
    ytrain(1:Ncurr,:,jj) = [s', Ex', Ey'];
    jj = jj + 1;

    % Allocate space for training data
    % Rows 1-5: (Input) d,rc,alpha,ra,s
    % Row 6-7: (Output) Ex, Ey
    % train_data = zeros(7,length(x));

    % Non-dimensionalize
%     lchar = emitter.h;  % use the emitter height as characteristic length
%     train_data(1,:) = emitter.d/lchar;
%     train_data(2,:) = emitter.rc/lchar;
%     train_data(3,:) = emitter.alpha;
%     train_data(4,:) = emitter.ra/lchar;
%     train_data(5,:) = s/s(end);
%     train_data(6,:) = Ex/(emitter.V0/lchar);
%     train_data(7,:) = Ey/(emitter.V0/lchar);

%     % Save 
%     fname = EPOST.get_filename(emitter.d, emitter.rc, emitter.alpha*(180/pi),...
%         emitter.h, emitter.ra, prefix);
%     fname = fullfile(data_dir, prefix, fname);
% %     save(fname,'train_data');
%     save(fname,'xtrain','ytrain');
end

% Save
V0 = emitter.V0;
save(fullfile(data_dir, prefix, 'train_base.mat'),'xtrain','ytrain', 'V0');

%% Save in feedforward net format (train_ffnet.mat)
% Loop over all training data
file = fullfile(data_dir,'train','train_base.mat');
load(file); % gives xtrain and ytrain
Nsamples = length(xtrain);
xdata = zeros(6,1);
ydata = zeros(2,1);
for ii = 1:Nsamples
    % Get number of discrete points for each sample
    ycurr = ytrain(:,:,ii);
    ycurr = ycurr(~isnan(ycurr));
    Ncurr = length(ycurr)/3;
    ycurr = reshape(ycurr,[Ncurr,3]);
    s = ycurr(:,1)/ycurr(end,1);
    Ex = ycurr(:,2); Ey = ycurr(:,3);

    % Nondimensionalize
    % Concatenate s array into the training input x
    x = [repmat(xtrain(1:5,:,ii),1,Ncurr); s'];
    y = [Ex'; Ey'];

    % Concatenate training data into big array
    xdata = [xdata, x];
    ydata = [ydata, y];
end
xdata = xdata(:,2:end);
ydata = ydata(:,2:end);
save(fullfile(data_dir,'train','train_ffnet.mat'),"xdata","ydata");

%% Save in dimensionless feedforward net format (train_dffnet.mat)
% Loop over all training data
file = fullfile(data_dir,'train','train_base.mat');
load(file); % gives xtrain, ytrain, and V0
Nsamples = length(xtrain);
xdata = zeros(5,1);
ydata = zeros(2,1);

for ii = 1:Nsamples
    % Get number of discrete points for each sample
    ycurr = ytrain(:,:,ii);
    ycurr = ycurr(~isnan(ycurr)); % ycurr = [s, Ex, Ey]
    Ncurr = length(ycurr)/3;
    ycurr = reshape(ycurr,[Ncurr,3]);
    s = ycurr(:,1)/ycurr(end,1);
    Ex = ycurr(:,2); Ey = ycurr(:,3);

    % Nondimensionalize
    h = xtrain(4,1,ii);
    x = [xtrain(1:3,:,ii) ./ [h;h;1]; xtrain(5,:,ii) / h];
    x = [repmat(x,1,Ncurr); s'];
    y = [Ex'; Ey'] / (V0/h);

    % Concatenate training data into big array
    xdata = [xdata, x];
    ydata = [ydata, y];
end
xdata = xdata(:,2:end);
ydata = ydata(:,2:end);
save(fullfile(data_dir,'train','train_dffnet.mat'),"xdata","ydata");

%% Save in dimensionless feedforward net format (train_dffnet_max.mat)
% but for EMAX only
% Loop over all training data
file = fullfile(data_dir,'train','train_base_cell.mat');
load(file); % gives xtrain, ytrain, and V0
Nsamples = length(xtrain);
xdata = zeros(4,Nsamples);
ydata = zeros(1,Nsamples);

for ii = 1:Nsamples
    % Get number of discrete points for each sample
    ycurr = ytrain{ii};
    Ex = ycurr(2,:); Ey = ycurr(3,:);
    Emag = sqrt(Ex.^2 + Ey.^2);
    [Emax, idx] = max(Emag);
%     Ex_max = Ex(idx);
%     Ey_max = Ey(idx);

    % Nondimensionalize
    xcurr = xtrain{ii};
    h = xcurr(4);
    x = [xcurr(1:3) ./ [h;h;1]; xcurr(5) / h];
    y = Emax / (V0/h);

    % Save training data to array
    xdata(:,ii) = x;
    ydata(ii) = y;
end
save(fullfile(data_dir,'train','train_dffnet_max.mat'),"xdata","ydata");