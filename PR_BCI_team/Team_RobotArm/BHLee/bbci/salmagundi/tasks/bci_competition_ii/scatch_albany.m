sub_dir= 'bci_competition_ii/';
subject= 'AA';
%subject= 'BB';
%subject= 'CC';
xTrials= [5 10];

file= sprintf('%salbany_%s_train', sub_dir, subject);
[cnt, mrk, mnt]= loadProcessedEEG(file);
band= [10 14]; 
[b,a]= getButterFixedOrder(band, cnt.fs, 6);
cnt= proc_filt(cnt, b, a);

mrk= mrk_selectClasses(mrk, {'top','bottom'});
epo= makeEpochs(cnt, mrk, [-1000 4500]);

erd= proc_rectifyChannels(epo);
erd= proc_filtByFFT(erd, [0 3], erd.fs);

fv= proc_selectIval(erd, [1500 3000]);
fv.silent= proc_selectIval(erd, [-1000 0]);
fv.proc= ['fv= proc_spatialprojection(epo, 2, epo.silent); ' ...
          'fv= proc_jumpingMeans(fv, 24);'];
doXvalidationPlus(fv, 'LDA', xTrials);

fv.proc= ['fv= proc_spatialprojection(epo, 2, epo, ''smooth''); ' ...
          'fv= proc_jumpingMeans(fv, 24);'];
doXvalidationPlus(fv, 'LDA', xTrials);





fv_dscr= proc_selectIval(fv, [1500 3000]);
[fv_w, W]= proc_spatialprojection(fv_dscr, 4, fv_dscr, 'smooth');

fv_w= proc_jumpingMeans(fv_w, 24);
doXvalidationPlus(fv_w, 'LDA', xTrials);

model= struct('classy','RLDA', 'msDepth',2, 'inflvar',2);
model.param= [0 0.01 0.1 0.5 0.75];
classy= selectModel(fv, model, [3 10 round(9/10*sum(any(fv.y)))]);
doXvalidation(fv_w, classy, xTrials);


fv_dscr= proc_selectIval(fv, [1500 3000]);
fv_silent= proc_selectIval(fv, [-1000 0]);
[fv_w, W]= proc_spatialprojection(fv_dscr, 2, ...
                                  fv_silent,'id',1, ...
                                  fv_dscr,'smooth',1);

fv_w= proc_jumpingMeans(fv_w, 24);
doXvalidationPlus(fv_w, 'LDA', xTrials);

