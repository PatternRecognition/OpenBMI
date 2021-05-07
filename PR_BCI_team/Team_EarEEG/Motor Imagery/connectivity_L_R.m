%% ¸ðµç trials
chan_L = {'FC5','FC3','FC1','C5','C3','C1','CP5','CP3','CP1'};
chan_R = {'FC2','FC4','FC6','C2','C4','C6','CP2','CP4','CP6'};

epo_L = proc_selectChannels(cap_epo{1,1},chan_L);
epo_R = proc_selectChannels(cap_epo{1,1},chan_R);

%% Left channels
eegData_L = permute(epo_L.x,[2,1,3]);
srate = cap_epo{1,1}.fs;
filtSpec.order = 50;
filtSpec.range = [1 100];
dataSelectArr = logical(epo_L.y');

[plv] = pn_eegPLV(eegData_L, srate, filtSpec, dataSelectArr);
plv_R = plv(:,:,:,1);
plv_L = plv(:,:,:,2);
plv_re = plv(:,:,:,3);

% figure; 
% plot((0:size(eegData, 2)-1)/srate, squeeze(plv(:, 17, 19, :)));
% xlabel('Time (s)'); ylabel('Plase Locking Value');

figure; 
subplot(3,1,1)
plot((0:size(eegData_L, 2)-1)/srate, mean(mean(plv_R,3),2));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
ylim([0 0.5])
title('Right')
grid on

subplot(3,1,2)
plot((0:size(eegData_L, 2)-1)/srate,  mean(mean(plv_L,3),2));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
ylim([0 0.5])
title('Left')
grid on

subplot(3,1,3)
plot((0:size(eegData_L, 2)-1)/srate,  mean(mean(plv_re,3),2));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
ylim([0 0.5])
title('Rest')
grid on

%% Right channels
eegData_R = permute(epo_R.x,[2,1,3]);
srate = cap_epo{1,1}.fs;
filtSpec.order = 50;
filtSpec.range = [1 100];
dataSelectArr = logical(epo_R.y');

[plv] = pn_eegPLV(eegData_R, srate, filtSpec, dataSelectArr);
plv_R = plv(:,:,:,1);
plv_L = plv(:,:,:,2);
plv_re = plv(:,:,:,3);

% figure; 
% plot((0:size(eegData, 2)-1)/srate, squeeze(plv(:, 17, 19, :)));
% xlabel('Time (s)'); ylabel('Plase Locking Value');

figure; 
subplot(3,1,1)
plot((0:size(eegData_R, 2)-1)/srate, mean(mean(plv_R,3),2));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
ylim([0 0.5])
title('Right')
grid on

subplot(3,1,2)
plot((0:size(eegData_R, 2)-1)/srate, mean(mean(plv_L,3),2));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
ylim([0 0.5])
title('Left')
grid on

subplot(3,1,3)
plot((0:size(eegData_R, 2)-1)/srate, mean(mean(plv_re,3),2));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
ylim([0 0.5])
title('Rest')
grid on


