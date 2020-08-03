[cnt,mrk,mnt]= loadProcessedEEG('Klaus_03_11_04/imagKlaus');
epo= makeEpochs(cnt, mrk, [-500 4000]);

epo= proc_laplace(epo);
epo= proc_selectChannels(epo, 'C3,4');

epo_tf= proc_tf_stft(epo, [4 30]);
epo_tf= proc_average(epo_tf);

nChans= length(epo_tf.clab);
nClasses= size(epo_tf.y,1);
for kk= 1:nClasses,
  for cc= 1:nChans,
    subplot(nChans, nClasses, kk + nClasses*(cc-1));
    imagesc(epo_tf.t, epo_tf.f, epo_tf.x(:,:,cc,kk));
    axis xy;
    xlabel('time  [ms]');
    ylabel('frequency  [Hz]');
    title(sprintf('%s: %s', epo_tf.clab{cc}, epo_tf.className{kk}));
  end
end
unifyCLim;
