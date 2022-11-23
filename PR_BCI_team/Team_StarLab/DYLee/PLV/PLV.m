%%
data_pro;

%%
eegData = epo.x;
  srate = 1000; %Hz
  filtSpec.order = 100;
%   filtSpec.range = [30 150]; % gamma band
  filtSpec.range = [70 150]; % high gamma band
%   dataSelectArr = epo.y'
 dataSelectArr = [[true(100, 1); false(100, 1)],...
      [false(100, 1); true(100, 1)]];
  [plv] = pn_eegPLV(eegData, srate, filtSpec, dataSelectArr);
% [plv] = pn_eegPLV(eegData, srate, filtSpec);
%    figure; plot((0:size(eegData, 2)-1)/srate, squeeze(plv(:, 1, 60, :)));
%   xlabel('Time (s)'); ylabel('Plase Locking Value');
  
  
%   A = adjacency(G)
plv1=plv(:,:,:,1);
plv2=plv(:,:,:,2);

%% Averaging PLV
% avgplv1=mean(plv1,1);
% avgplv2=mean(plv2,1);
avgplv1=mean(plv1);
avgplv2=mean(plv2);
avg1=reshape(avgplv1,60,60)
avgplv3=mean(avg1,1);
% avgplv=mean(plv,1);
disp('Averaging, reshape are completed')
% aa=reshape(avgplv3,1,60)
% aa=reshape(avgplv1,60,60)
% aa=aa+aa'  % 0.5이상이면 connectivity 높다

%% ttest
% 
% [h,p,ci,stats] = ttest(aa);
[h,p,ci,stats] = ttest(avgplv);
% 


%% topo
% topoplot_connect(stats.tstat, myEEG.chanlocs)