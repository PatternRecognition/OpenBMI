file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/graz/'];
file_list= {'O3VR', 'S4b', 'X11b'};

su= 1;

file= strcat(file_dir, file_list{su});
S= load(file);

cnt= struct('x',S.s, 'fs',S.HDR.SampleRate, ...
            'title',['graz/' file_list{su}]);
cnt.clab= {'C3','C4'};
mrk= struct('pos', S.HDR.TRIG', 'fs',cnt.fs);
mrk.y= zeros(2,length(mrk.pos));
for cc= 1:2,
  mrk.y(cc,:)= (S.HDR.Classlabel==cc);
end
mrk.className= {'left','right'};
%mrk.y(:,S.HDR.ArtifactSelection)= 0;
clear S
mnt= setDisplayMontage(cnt.clab, 'C3,C4,legend');


%% discard test trials
iTest= find(all(mrk.y==0));
mrk_test= mrk_selectEvents(mrk, iTest);
mrk= mrk_selectEvents(mrk, setdiff(1:length(mrk.pos), iTest));

%% substitute for NaN values.
%% stupid method. TODO: make it better
for cc= 1:size(cnt.x,2),
  iBad= find(isnan(cnt.x(:,cc)));
  for ii= 1:length(iBad),
    cnt.x(iBad(ii),cc)= cnt.x(iBad(ii)-1,cc);
  end
end



%% spectra
epo= makeEpochs(cnt, mrk, [3500 8000]);
spec= proc_spectrum(epo, [2 45]);
grid_plot(spec, mnt);

spec_rsq= proc_r_square(spec);
grid_plot(spec_rsq, mnt);


%% TODO: ERDs


%% slow potentials
epo= makeEpochs(cnt, mrk, 3000+[0 750]);
epo= proc_baseline(epo, 250, 'beginning');
grid_plot(epo, mnt);

epo_rsq= proc_r_square(epo);
grid_plot(epo_rsq, mnt);


fv= proc_selectIval(epo, [3500 3750]);
fv= proc_jumpingMeans(fv, 4);
xvalidation(fv, 'LDA');
