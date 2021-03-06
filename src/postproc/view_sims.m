clear all;
close all;
clc;
addpath('../emitter/');
data_dir = '../../data/geometry';
files = dir(fullfile(data_dir,'sims'));

% Loop over all training data
for ii = 1:length(files)
    if files(ii).isdir
        continue
    end
    if files(ii).name(1) == '.'
        continue
    end

    file = fullfile(files(ii).folder,files(ii).name);

    load(file);
    EPOST.solplot(emitter);
%     EPOST.emitterplot(emitter);
    waitforbuttonpress
    close all;
end