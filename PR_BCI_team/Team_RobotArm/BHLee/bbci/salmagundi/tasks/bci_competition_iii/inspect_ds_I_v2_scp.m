file= [EEG_IMPORT_DIR 'bci_competition_iii/tuebingen/Competition_train'];
S= load(file);
epo= struct('x', permute(S.X, [3 2 1]));
epo.y= double([S.Y'==-1; S.Y'==1]);
epo.className= {'left','tongue'};
epo.fs= 1000;
epo.t= 501:3500;
clear mnt S
for ii= 1:8,
  for jj= 1:8,
    idx= ii+(jj-1)*8;
    mnt.clab{idx}= sprintf('ch_%d_%d', ii, jj);
    mnt.x(idx,1)= (ii-4.5)/5;
    mnt.y(idx,1)= (4.5-jj)/5;
    mnt.box(:,idx)= [ii;9-jj];
  end
end
mnt.box(:,end+1)= [9;8];
mnt.box_sz= ones(size(mnt.box));
epo.clab= mnt.clab;

epo= proc_jumpingMedians(epo, 10);  %% brute force downsampling to 100Hz


file= [EEG_IMPORT_DIR 'bci_competition_iii/tuebingen/Competition_test'];
S= load(file);
epo_te= struct('x', permute(S.X, [3 2 1]));
epo_te.y= ones([1 size(epo_te.x,3)]);
epo_te.className= {'test'};
epo_te.fs= 1000;
epo_te.t= 501:3500;
epo_te.clab= mnt.clab;
clear S

epo_te= proc_jumpingMedians(epo_te, 10);  %% brute force downsampling to 100Hz

epo= proc_appendEpochs(epo, epo_te);
mnt= rmfield(mnt, 'box');
clear epo_te

Epo= epo;
epo= proc_baseline(epo, 250, 'beginning');
epo_rsq= proc_r_square(proc_selectClasses(epo, 'not','test'));
H= grid_plot(epo, mnt);
grid_addBars(epo_rsq, 'box','on', 'h_scale',H.scale);


%% calculate channel scores (slow potetials)
fv= proc_selectIval(epo, [1250 3500]);
fv= proc_jumpingMedians(fv, 25);
for cc= 1:length(fv.clab), fprintf('%s> ', fv.clab{cc}),
  ff= proc_selectChannels(fv, cc);
  ff= proc_selectClasses(ff, 'not','test');
  ssp1(cc)= xvalidation(ff, 'FD', 'xTrials',[3 5]);
end
[so,si]= sort(ssp1);
nSel= max(find(so<0.4));
for cc= 1:nSel,
  fprintf('%s> %.1f%%\n', fv.clab{si(cc)}, 100*so(cc));
end
%[dmy, idx, ssp2]= proc_fs_classifierWeights(fv, 100);

clab= {'ch_7_1','ch_6_2','ch_2_3','ch_6_3','ch_6_5','ch_7_6','ch_6_7'};

clab= fv.clab(si(1:nSel));
fv= proc_selectChannels(fv, clab);
fv2= proc_selectClasses(fv, 'not','test');

model_RLDA= struct('classy', 'RLDA');
model_RLDA.param= [0 0.01 0.1 0.3 0.5 0.7];


model_RDA= struct('classy', 'RDA', ...
                  'msDepth',2, 'inflvar',2);
model_RDA.param(1)= struct('index',2, ...
                           'value',[0 0.05 0.25 0.5 0.75 0.9 1]);
model_RDA.param(2)= struct('index',3, ...
                           'value',[0 0.001 0.01 0.1 0.3 0.5 0.7]);

opt_xv= struct('out_trainloss',1, 'outer_ms',1, 'xTrials',[2 5], ...
               'verbosity',2, 'msTrials',[2 5 -1]);
%xvalidation(fv2, model_RDA, opt_xv);
classy= select_model(fv, model_RDA, [3 5]);

classy= 'FD';
ci3= find(fv.y(3,:));

C= trainClassifier(fv2, classy);

out= applyClassifier(fv, classy, C, ci3);
sum(out<0)
out= out - median(out);
clf; hist(out, 20);
save([BCI_DIR 'tasks/bci_competition_iii/output_temporal'], 'out');


epo_te= proc_selectClasses(epo, 'test');
epo_te.y= [out<0; out>0];
epo_te.className= fv.className(1:2);

figure(1)
grid_plot(epo, mnt);
for ii= 1:length(clab); 
  ax= grid_getSubplots(clab{ii}); 
  set(ax, 'color',[1 0 0]); 
end
figure(2)
grid_plot(proc_baseline(proc_selectClasses(Epo,'not','test'),250), mnt)



clab= {'ch_5_1','ch_5_4','ch_6_4','ch_5_5','ch_5_6','ch_6_6','ch_8_6'};
fv= proc_selectIval(epo, [1250 3500]);
%fv= proc_selectIval(epo, [1000 3500]);
fv= proc_jumpingMeans(fv, 25);
fv= proc_selectChannels(fv, clab);
fv2= proc_selectClasses(fv, 'not','test');
C= trainClassifier(fv2, classy);

out= applyClassifier(fv, classy, C, ci3);
clf; hist(out, 15);
sum(out<0)
out= out-median(out);

epo_te= proc_selectClasses(Epo, 'test');
epo_te.y= [out<0; out>0];
epo_te.className= fv.className(1:2);

figure(1)
grid_plot(proc_baseline(epo_te, 250), mnt);
for ii= 1:length(clab); 
  ax= grid_getSubplots(clab{ii}); 
  set(ax, 'color',[1 0 0]); 
end
figure(2)
grid_plot(proc_baseline(proc_selectClasses(Epo,'not','test'),250), mnt)
