file= 'Klaus_04_04_08/imag_basketfbKlaus';
szenario= 'imag_lettKlaus_udp_1d_szenario_01';

[cnt, mrk, mnt]= loadProcessedEEG(file);
mrk= mrk_selectClasses(mrk, 'target*');

cnt= proc_addClassifierOutput(cnt, szenario);
cnt= proc_addChannelFromLog(cnt, mrk);

fb= proc_selectChannels(cnt, 'out','yData');
epo= makeEpochs(fb, mrk, [0 2000]);
erp= proc_average(epo);

subplot(1,2,1);
plot(erp.t, squeeze(erp.x(:,1,:)));
legend(erp.className, 2);
title('classifier output');

subplot(1,2,2);
plot(erp.t, squeeze(erp.x(:,2,:)));
legend(erp.className, 2);
title('feedback trajectory');
