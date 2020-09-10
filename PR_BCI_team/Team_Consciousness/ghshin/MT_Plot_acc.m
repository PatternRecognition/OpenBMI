clear; close all; clc;

%% Excel data load
WP = xlsread('H:\NeuroImage\WM_acc.xlsx', 3, 'B3:D19');
PM = xlsread('H:\NeuroImage\WM_acc.xlsx', 3, 'J3:L19');
LM = xlsread('H:\NeuroImage\WM_acc.xlsx', 3, 'R3:T19');

col=[253, 174, 97; 
    171, 221, 164;   
    43, 131, 186];
col=col/255;

i = [2, 0, -2];
%% boxplot [WP]
figure()
boxplot(WP, 'Notch', 'on')
set(gca, 'xtick', []);

set(findobj(gca,'type','line'),'linew',3);
h = findobj(gca,'Tag','Box');

for j=1:length(h)
    x = patch(get(h(j+i(j)),'XData'),get(h(j+i(j)),'YData'), col(j,:),'FaceAlpha',.5);
    % MODIFICATION TO SET THE LINE COLOR TO PATCH COLOR:
    h(length(h)-j+1).Color = x.FaceColor; % reordered to match
    hold on;
    
    x=repmat(1:3,length(WP),1);
    scatter(x(:,j),WP(:,j),'filled','MarkerEdgeColor',col(:,j), 'MarkerFaceColor',col(:,j),'jitter','on','jitterAmount',0.15);
end
% set(findobj(gca,'type','line')','linew',2);
% hLegend = legend(findall(gca,'Tag','Box'), {'BN','AN','24H'});

%% boxplot [PM]
figure()
boxplot(PM, 'Notch', 'on')
set(gca, 'xtick', []);

set(findobj(gca,'type','line'),'linew',3);
h = findobj(gca,'Tag','Box');

for j=1:length(h)
    x = patch(get(h(j+i(j)),'XData'),get(h(j+i(j)),'YData'), col(j,:),'FaceAlpha',.5);
    % MODIFICATION TO SET THE LINE COLOR TO PATCH COLOR:
    h(length(h)-j+1).Color = x.FaceColor; % reordered to match
    hold on;
    
    x=repmat(1:3,length(PM),1);
    scatter(x(:,j),PM(:,j),'filled','MarkerEdgeColor',col(:,j), 'MarkerFaceColor',col(:,j),'jitter','on','jitterAmount',0.15);
end
% set(findobj(gca,'type','line')','linew',2);
% hLegend = legend(findall(gca,'Tag','Box'), {'BN','AN','24H'});

%% boxplot [LM]
figure()
boxplot(LM, 'Notch', 'on')
set(gca, 'xtick', []);

set(findobj(gca,'type','line'),'linew',3);
h = findobj(gca,'Tag','Box');

for j=1:length(h)
    x = patch(get(h(j+i(j)),'XData'),get(h(j+i(j)),'YData'), col(j,:),'FaceAlpha',.5);
    % MODIFICATION TO SET THE LINE COLOR TO PATCH COLOR:
    h(length(h)-j+1).Color = x.FaceColor; % reordered to match
    hold on;
    
    x=repmat(1:3,length(LM),1);
    scatter(x(:,j),LM(:,j),'filled','MarkerEdgeColor',col(:,j), 'MarkerFaceColor',col(:,j),'jitter','on','jitterAmount',0.15);
end
% set(findobj(gca,'type','line')','linew',2);
% hLegend = legend(findall(gca,'Tag','Box'), {'BN','AN','24H'});