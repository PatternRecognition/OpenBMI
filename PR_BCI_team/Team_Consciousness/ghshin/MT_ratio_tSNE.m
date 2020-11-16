%% Ratio / t-SNE
clc; clear; close all 

%% Load + ratio
Path_x = ['H:\Conference\Winter conference\GH_Memory\LM_PSD_24H\x_data\'];
Path_y = ['H:\Conference\Winter conference\GH_Memory\LM_PSD_24H\y_data\'];
DirGroup_x = dir(fullfile(Path_x,'*mat'));
DirGroup_y = dir(fullfile(Path_y,'*mat'));
DirGroup_x = {DirGroup_x.name};
DirGroup_y = {DirGroup_y.name};
for n = 1:size(DirGroup_x,2)
    temp_x(n) = load([Path_x, DirGroup_x{n}]);
    temp_y(n) = load([Path_y, DirGroup_y{n}]);
    
    % each subject
%     X = permute(reshape(temp_x(n).x, [360 size(temp_x(n).x,3)]), [2 1]);
%     y = temp_y(n).y;
%     
%     Y = tsne(X,'Distance','chebychev');
%     gscatter(Y(:,1),Y(:,2),y)
%     hold on
%     
    % all subjects
    if n > 1
        temp_x(1).x = cat(3, temp_x(1).x, temp_x(n).x);
        temp_y(1).y = cat(1, temp_y(1).y, temp_y(n).y);
    end
end

X = permute(reshape(temp_x(1).x, [360 size(temp_x(1).x,3)]), [2 1]);
y = temp_y(1).y;
%% t-SNE


% 
% Y = tsne(X);
% gscatter(Y(:,1),Y(:,2),y)

% each frequency band
% X= temp_x(1).x;
% for i = 1:size(X,2)
%     figure()
%     XX = squeeze(X(:,i,:));
%     temp = permute(XX, [2 1]);
%     y = temp_y(1).y;
%     Y = tsne(temp);
%     gscatter(Y(:,1),Y(:,2),y)
% end

%% PCA
[~,~,s] = unique(y);
[~,D_pca] = pca(X);
figure(1)
plot(D_pca(s==1,1),D_pca(s==1,2), 'rx');
hold on
plot(D_pca(s==2,1),D_pca(s==2,2), 'go');
%% ratio
% X = permute(reshape(temp_x(1).x, [360 size(temp_x(1).x,3)]), [2 1]);
% y = temp_y(1).y;
% tabulate(y);

