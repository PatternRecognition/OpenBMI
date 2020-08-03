setup_augcog;
%augcog=struct('file', augcog.file);
fig_dir= 'augcog_misc/';
high_first= [0 0 0 1 1];
ival_audi_low= [[380 450]; [380 450]; [450 550]; [380 420]; [380 450]];
ival_audi_high= [[400 480]; [380 450]; [500 600]; [400 440]; [400 480]];
nn= 4;
task= 'audio';
%task= 'visuell';

mrk_cmt= readMarkerComments(augcog(nn).file, 100);

ii= strmatch(task, mrk_cmt.str);
if high_first(nn),
  si=[1 2 3 4];
else
  si=[3 4 1 2];
end
hs= mrk_cmt.pos(ii(si(1)))*1000/mrk_cmt.fs;
he= mrk_cmt.pos(ii(si(2)))*1000/mrk_cmt.fs;
ls= mrk_cmt.pos(ii(si(3)))*1000/mrk_cmt.fs;
le= mrk_cmt.pos(ii(si(4)))*1000/mrk_cmt.fs;

cnt1= readGenericEEG(augcog(nn).file, [], 100, hs, he-hs);
cnt2= readGenericEEG(augcog(nn).file, [], 100, ls, le-ls);
classDef= {'S  1','D  1';'standard','deviant'};
mrk1= read_markers(augcog(nn).file, 100, classDef, [hs he]);
mrk1.className= {'high s','high d'};
classDef= {'I  1','S  1';'standard','deviant'};
mrk2= read_markers(augcog(nn).file, 100, classDef, [ls le]);
mrk2.className= {'low s','low d'};
[cnt, Mrk]= proc_appendCnt(cnt1, cnt2, mrk1, mrk2);

mnt= projectElectrodePositions(cnt.clab);
mnt.y= 1.2*mnt.y;
grd= sprintf('F3,FC1,Fz,FC2,F4\nC3,CP1,Cz,CP2,C4\nlegend,O1,Pz,O2,P4');
%grd= sprintf('F7,Fp1,Eog,Fp2,F8\nF3,FC1,Fz,FC2,F4\nC3,CP1,Cz,CP2,C4\nP3,O1,Pz,O2,P4,TP9,T7,legend,T8,TP10');
%grd= sprintf('F7,Fp1,Eog,Fp2,F8\nF3,FC1,Fz,FC2,F4\nC3,CP1,Cz,CP2,C4\nP3,O1,legend,O2,P4');
mnt= setDisplayMontage(mnt, grd);
scalp_opt= struct('shading','flat', 'resolution',20, 'contour',-4);


mrk= mrk_selectClasses(Mrk, 'low*');
epo= makeEpochs(cnt, mrk, [-200 700]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt);
saveFigure([fig_dir epo.title '_p300_audi_low'], [12 7]);

ival= ival_audi_low(nn,:);
plotClassTopographies(epo, mnt, ival, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalp_audi_low'], [12 5]);


mrk= mrk_selectClasses(Mrk, 'high*');
epo= makeEpochs(cnt, mrk, [-200 700]);
epo= proc_baseline(epo, [-200 0]);

grid_plot(epo, mnt);
saveFigure([fig_dir epo.title '_p300_audi_high'], [12 7]);

ival= ival_audi_high(nn,:);
plotClassTopographies(epo, mnt, ival, scalp_opt);
saveFigure([fig_dir epo.title '_p300_scalp_audi_high'], [12 5]);




return




% if you want
crit.maxmin=100;
iArte= find_artifacts(epo, {'F3,z,4','C3,z,4','P3,z,4'}, crit);
fprintf('%d artifact trials removed (max-min>%d uV)\n', ...
        length(iArte), crit.maxmin);
epo= proc_selectEpochs(epo, setdiff(1:size(epo.x,3),iArte));

