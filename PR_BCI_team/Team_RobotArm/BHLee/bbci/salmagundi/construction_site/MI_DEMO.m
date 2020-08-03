file= 'Gabriel_00_09_05/selfpaced2sGabriel';
 
[cnt, mrk, mnt]= loadProcessedEEG(file);

cnt.proc = ['fv= proc_selectChannels(epo, ''FC#'', ''C#'', ''CP#''); ' ...
            'fv= proc_filtBruteFFT(fv, [0.8 3], 128, 150); ' ...  
            'fv= proc_jumpingMeans(fv, 5, 3);'];
cnt.classy = 'LDA'; % this is the default
cnt.proclabel = 0 % this means preprocessing is independent of the
                  % labelling  (default)

T= -1000:50:500;
subplot(211);
MI = calc_MI(cnt, mrk, [], T, [-1280 0]-120);
plot(T, MI);
title('evaluation of MI on training set');

subplot(212);
MI = calc_MI(cnt, mrk, [2 5], T, [-1280 0]-120);
plot(T, MI);
title('MI in x-validation');



return

cnt_emg= proc_selectChannels(cnt, 'EMG*');

cnt_emg.proc = 'fv= proc_detectEMG(epo);';
cnt_emg.classy = 'LDA'; % this is the default
cnt_emg.proclabel = 0 % this means preprocessing is independent of the
                  % labelling  (default)

T= -400:10:100;
subplot(211);
MI = calc_MI(cnt_emg, mrk, [], T, [-300 0]);
plot(T, MI);
title('evaluation of MI on training set');

subplot(212);
MI = calc_MI(cnt_emg, mrk, [2 5], T, [-300 0]);
plot(T, MI);
title('MI in x-validation');
