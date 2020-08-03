file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/martigny/'];

su= 2;
file_list= strcat(file_dir, 'train_subject', int2str(su), '_raw0', ...
                  cellstr(int2str([1:3]')));
for ff= 1:length(file_list),
  S= load(file_list{ff});
  if ff==1,
    cnt= struct('x', S.X, 'clab',{S.nfo.clab}, 'fs',S.nfo.fs, ...
                'title', untex(S.nfo.name(1:end-2)));
    Y= S.Y;
  else
    cnt.x= cat(1, cnt.x, S.X);
    Y= cat(1, Y, S.Y);
  end
  cnt.T(ff)= size(S.X, 1);
end
mnt= setDisplayMontage(S.nfo.clab, 'martigny');
mnt.xpos= S.nfo.xpos;
mnt.ypos= S.nfo.ypos;
clear S

mrk= struct('fs',cnt.fs);
mrk.pos= find(diff([0; Y]))';
break_points= cumsum(cnt.T(1:end-1));
mrk.pos= unique([mrk.pos break_points+1]);
mrk.y= [Y(mrk.pos)'==2; Y(mrk.pos)'==3; Y(mrk.pos)'==7];
mrk.className= {'left', 'right', 'word'};


%% spectra
min_len= min(diff(mrk.pos))/mrk.fs*1000;
epo= makeEpochs(cnt, mrk, [0 min_len]);
spec= proc_spectrum(epo, [1 105]);
grid_plot(spec, mnt);

spec= proc_spectrum(epo, [1 35]);
grid_plot(spec, mnt);

spec_rsq= proc_r_square(spec);
grid_plot(spec_rsq, mnt);


%% evoked potentials
epo= makeEpochs(cnt, mrk, [0 750]);
epo= proc_baseline(epo, [0 250]);
grid_plot(epo, mnt);

epo_rsq= proc_r_square(epo);
grid_plot(epo_rsq, mnt);



model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.01 0.1 0.3 0.5 0.7];
opt_xv= struct('out_trainloss',1, 'outer_ms',1, 'xTrials',[10 10], ...
               'verbosity',2);

%% eye movements in class 'word'
fv= proc_selectChannels(epo, {'Fp1,2','AF3,4'});
fv= proc_spectrum(fv, [1 5]);
fv= proc_selectClasses(fv, 'left','word');
xvalidation(fv, model_RLDA, opt_xv);


fv= proc_laplace(epo, 'diagonal');
fv= proc_selectChannels(fv, 'C3,4','CP5,1,2');
fv= proc_spectrum(fv, [10 16], fv.fs/2);
fv= proc_selectClasses(fv, 'left','word');
xvalidation(fv, model_RLDA, opt_xv);
