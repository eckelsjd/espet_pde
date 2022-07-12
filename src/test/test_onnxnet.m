%% Test Matlab's SeriesNetwork and export to onnx
% Joshua Eckels
% 3/2/22
clear all;
close all;
clc;
addpath('../emitter')
addpath('../postproc')

% Load network and training data
data_dir = '../../data/';
dataset = 'geometry'; 
prefix = 'train';
file = fullfile(data_dir, dataset, prefix, sprintf('%s_dffnet_max.mat', prefix));
geometry_data = load(file);
xtrain = geometry_data.xdata;
ytrain = geometry_data.ydata;

file = fullfile(data_dir, dataset, 'models',"model_onnxnet.mat");
load(file); % Gives net object and normalization settings

% Load test data
file = fullfile(data_dir, dataset, 'test', 'test_dffnet_max.mat');
test_data = load(file);
xtest = test_data.xdata;
ytest = test_data.ydata;

%% Show results on test set
% Preprocess
extractor_thickness = 76*10^(-6);
V0 = 1000;
hvec = xtest(4,:);
x = [(xtest(1,:)+xoffset)./hvec; xtest(2,:)./hvec; xtest(3,:); xtest(5,:)./hvec];
x = [exp_cdf(x(1,:), lambda_x(1)); exp_cdf(x(2,:), lambda_x(2)); x(3,:); exp_cdf(x(4,:), lambda_x(4)) ];
xn = mapminmax('apply',x, xs);

% Predict
yn = predict(net,xn');
ypred = mapminmax('reverse', yn', ys);
ypred = exp_cdf_inv(ypred, lambda_y);
ypred = ypred .* (V0./hvec);

% Analytical solution (MS)
% neg_d_idx = find(xtest(1,:) <= 0);
% Emax_ms = zeros(1,length(xtest));
% s = 0; % test only at the tip of the emitter
% for i = 1:length(xtest)
%     if any(neg_d_idx==i)
%         continue
%     end
%     d_ms = xtest(1,i);
%     rc_ms = xtest(2,i);
%     [ytip, Ex_tip, Ey_tip] = EPOST.ms_solution(rc_ms, d_ms, V0, s);
%     Emax_ms(i) = sqrt(Ex_tip^2 + Ey_tip^2);
% end

% Remove cases where d < 0
% Emax_ms(neg_d_idx) = [];
% ytest(neg_d_idx) = [];
% ypred(neg_d_idx) = [];
% xtest(:, neg_d_idx) = [];

% Error metrics for Martinez-Sanchez
% rel_error_ms = 100*(abs(Emax_ms - ytest)./ytest);
% rmse_ms = sqrt(mean((Emax_ms - ytest).^2));

% Error metrics for surrogate
rel_error = 100*(abs(ypred - ytest)./ytest);
rmse = sqrt(mean((ypred - ytest).^2));

% Plot results
figure()
hold on;
% histogram(rel_error_ms,'Normalization','pdf','NumBins', 100)
histogram(rel_error,'Normalization','pdf','FaceColor','red')
label_str = sprintf('Relative percent error ($\\%%$)');
xlabel(label_str,'Interpreter','latex');
ylabel('PDF','Interpreter','latex');
set(gcf,'color','white')
% leg = legend('Martinez-Sanchez', 'Surrogate');
% set(leg,'Interpreter','latex');
% set(gca, 'YScale','log');
xlim([0 inf]);

fprintf('Overall surrogate results:\n');
fprintf('Mean: %.2f \n', mean(rel_error));
fprintf('Std dev: %.2f \n', std(rel_error));
fprintf('Min: %.2f \n', min(rel_error));
fprintf('25th percentile: %.2f \n', prctile(rel_error, 25));
fprintf('50th percentile: %.2f \n', prctile(rel_error, 50));
fprintf('75th percentile: %.2f \n', prctile(rel_error, 75));
fprintf('Max: %.2f \n\n', max(rel_error));

% fprintf('Overall MS results:\n');
% fprintf('Mean: %.2f \n', mean(rel_error_ms));
% fprintf('Std dev: %.2f \n', std(rel_error_ms));
% fprintf('Min: %.2f \n', min(rel_error_ms));
% fprintf('25th percentile: %.2f \n', prctile(rel_error_ms, 25));
% fprintf('50th percentile: %.2f \n', prctile(rel_error_ms, 50));
% fprintf('75th percentile: %.2f \n', prctile(rel_error_ms, 75));
% fprintf('Max: %.2f \n', max(rel_error_ms));

% Save results to file
A = [xtest', rel_error'];
bs_file = fullfile(data_dir, dataset, 'test', 'test_results.txt');
fd = fopen(bs_file, 'wt');
fprintf(fd, 'd rc alpha h ra rel_error\n');
fclose(fd);
writematrix(A, bs_file, 'Delimiter', ' ', 'WriteMode', 'append');


%% Show worst results
[~, idx] = max(rel_error);
params = xtest(:,idx);
% figure();
% subplot(5,1,1);
% histogram(xtrain(1,:)*1e6,'Normalization','pdf');
% hold on;
% xline(params(1)*1e6,'-r','LineWidth',2);
% xlabel('d [$\mu$m]','Interpreter','latex');
% subplot(5,1,2);
% histogram(xtrain(2,:)*1e6,'Normalization','pdf');
% hold on;
% xline(params(2)*1e6,'-r','LineWidth',2);
% xlabel('$r_c$ [$\mu$m]','Interpreter','latex');
% subplot(5,1,3);
% histogram(xtrain(3,:)*(180/pi),'Normalization','pdf');
% hold on;
% xline(params(3)*(180/pi),'-r','LineWidth',2);
% xlabel('$\alpha$ [deg]','Interpreter','latex');
% subplot(5,1,4);
% histogram(xtrain(4,:)*1e6,'Normalization','pdf');
% hold on;
% xline(params(4)*1e6,'-r','LineWidth',2);
% xlabel('$h$ [$\mu$m]','Interpreter','latex');
% subplot(5,1,5);
% histogram(xtrain(5,:)*1e6,'Normalization','pdf');
% hold on;
% xline(params(5)*1e6,'-r','LineWidth',2);
% xlabel('$r_a$ [$\mu$m]','Interpreter','latex');
% % set(gcf,'Position',[200 800 400 700])
% set(gcf,'color','white')

% emitter = Emitter(params(1),params(2),params(3),params(4),params(5),extractor_thickness,V0);
% EPOST.solplot(emitter);

% Relative error > 50%
% idx = find(rel_error >= 100);
[B, I] = sort(rel_error);
idx = I(end-19:end);
res = xtest(:, idx);

fprintf('\nWorst performers ( >50 percent error)\n');
n = 4;
fprintf('Emitter height:\n');
fprintf('Mean: %.2f \n', mean(res(n,:))*1e6);
fprintf('Std dev: %.2f \n', std(res(n,:))*1e6);
fprintf('Min: %.2f \n', min(res(n,:))*1e6);
fprintf('25th percentile: %.2f \n', prctile(res(n,:), 25)*1e6);
fprintf('50th percentile: %.2f \n', prctile(res(n,:), 50)*1e6);
fprintf('75th percentile: %.2f \n', prctile(res(n,:), 75)*1e6);
fprintf('Max: %.2f \n\n', max(res(n,:))*1e6);
n = 3;
fprintf('Cone half-angle:\n');
fprintf('Mean: %.2f \n', mean(res(n,:))*180/pi);
fprintf('Std dev: %.2f \n', std(res(n,:))*180/pi);
fprintf('Min: %.2f \n', min(res(n,:))*180/pi);
fprintf('25th percentile: %.2f \n', prctile(res(n,:), 25)*180/pi);
fprintf('50th percentile: %.2f \n', prctile(res(n,:), 50)*180/pi);
fprintf('75th percentile: %.2f \n', prctile(res(n,:), 75)*180/pi);
fprintf('Max: %.2f \n\n', max(res(n,:))*180/pi);
n = 2;
fprintf('Radius of curvature:\n');
fprintf('Mean: %.2f \n', mean(res(n,:))*1e6);
fprintf('Std dev: %.2f \n', std(res(n,:))*1e6);
fprintf('Min: %.2f \n', min(res(n,:))*1e6);
fprintf('25th percentile: %.2f \n', prctile(res(n,:), 25)*1e6);
fprintf('50th percentile: %.2f \n', prctile(res(n,:), 50)*1e6);
fprintf('75th percentile: %.2f \n', prctile(res(n,:), 75)*1e6);
fprintf('Max: %.2f \n\n', max(res(n,:))*1e6);
n = 1;
fprintf('Tip to extractor distance:\n');
fprintf('Mean: %.2f \n', mean(res(n,:))*1e6);
fprintf('Std dev: %.2f \n', std(res(n,:))*1e6);
fprintf('Min: %.2f \n', min(res(n,:))*1e6);
fprintf('25th percentile: %.2f \n', prctile(res(n,:), 25)*1e6);
fprintf('50th percentile: %.2f \n', prctile(res(n,:), 50)*1e6);
fprintf('75th percentile: %.2f \n', prctile(res(n,:), 75)*1e6);
fprintf('Max: %.2f \n\n', max(res(n,:))*1e6);
n = 5;
fprintf('Radius of aperture:\n');
fprintf('Mean: %.2f \n', mean(res(n,:))*1e6);
fprintf('Std dev: %.2f \n', std(res(n,:))*1e6);
fprintf('Min: %.2f \n', min(res(n,:))*1e6);
fprintf('25th percentile: %.2f \n', prctile(res(n,:), 25)*1e6);
fprintf('50th percentile: %.2f \n', prctile(res(n,:), 50)*1e6);
fprintf('75th percentile: %.2f \n', prctile(res(n,:), 75)*1e6);
fprintf('Max: %.2f \n\n', max(res(n,:))*1e6);
fprintf('Percent error:\n');
fprintf('Mean: %.2f \n', mean(rel_error(idx)));
fprintf('Std dev: %.2f \n', std(rel_error(idx)));
fprintf('Min: %.2f \n', min(rel_error(idx)));
fprintf('25th percentile: %.2f \n', prctile(rel_error(idx), 25));
fprintf('50th percentile: %.2f \n', prctile(rel_error(idx), 50));
fprintf('75th percentile: %.2f \n', prctile(rel_error(idx), 75));
fprintf('Max: %.2f \n\n', max(rel_error(idx)));

% Fit an exponential distribution to the column vector x and return the CDF
% of the samples in x (CDF(X) ~ U(0,1))
function u = exp_cdf(x, lambda)
    u = 1 - exp(-lambda*x);
end

function y = exp_cdf_inv(u, lambda)
    y = (-1/lambda)*log(abs(1-u));
end