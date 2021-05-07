%% ¸ðµç trials
eegData = permute(cap_epo{1,1}.x,[2,1,3]);
srate = cap_epo{1,1}.fs;
filtSpec.order = 50;
filtSpec.range = [1 100];
dataSelectArr = logical(cap_epo{1,1}.y');

[plv] = pn_eegPLV(eegData, srate, filtSpec, dataSelectArr);
plv_R = plv(:,:,:,1);
plv_L = plv(:,:,:,2);
plv_re = plv(:,:,:,3);

% figure; 
% plot((0:size(eegData, 2)-1)/srate, squeeze(plv(:, 17, 19, :)));
% xlabel('Time (s)'); ylabel('Plase Locking Value');

figure; 
subplot(3,1,1)
plot((0:size(eegData, 2)-1)/srate, squeeze(plv_R(:, 17, 19)));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
title('Right')

subplot(3,1,2)
plot((0:size(eegData, 2)-1)/srate, squeeze(plv_L(:, 17, 19)));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
title('Left')

subplot(3,1,3)
plot((0:size(eegData, 2)-1)/srate, squeeze(plv_re(:, 17, 19)));
xlabel('Time (s)'); ylabel('Plase Locking Value');
xlim([0 1])
title('Rest')

