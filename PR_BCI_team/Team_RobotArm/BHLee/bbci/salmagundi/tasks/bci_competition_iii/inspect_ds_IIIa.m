file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/graz/'];
file_list= {'k3b', 'k6b', 'l1b'};

su= 1;

file= strcat(file_dir, file_list{su});
S= load(file);

cnt= struct('x',S.s, 'fs',S.HDR.SampleRate, ...
            'title',['graz/' file_list{su}]);
cnt.clab= unblank(cellstr(int2str([1:60]'))');
cnt.clab([28 31 34])= {'C3','Cz','C4'};
mrk= struct('pos', S.HDR.TRIG', 'fs',cnt.fs);
mrk.y= zeros(4,length(mrk.pos));
for cc= 1:4,
  mrk.y(cc,:)= (S.HDR.Classlabel==cc)';
end
mrk.className= {'left','right','foot','tongue'};
%mrk.y(:,S.HDR.ArtifactSelection)= 0;
clear S

mnt= setDisplayMontage(cnt.clab, 'graz60');


%% spectra
epo= makeEpochs(cnt, mrk, [3000 7000]);
spec= proc_spectrum(epo, [2 35]);
grid_plot(spec, mnt);

spec_rsq= proc_r_square(spec);
grid_plot(spec_rsq, mnt);



%% evoked potentials
epo= makeEpochs(cnt, mrk, 3000+[0 750]);
epo= proc_baseline(epo, 3000+[0 250]);
grid_plot(epo, mnt);

epo_rsq= proc_r_square(epo);
grid_plot(epo_rsq, mnt);
