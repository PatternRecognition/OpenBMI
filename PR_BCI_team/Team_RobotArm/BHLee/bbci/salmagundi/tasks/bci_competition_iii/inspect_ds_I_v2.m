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

clab= {'ch_4_2','ch_5_3','ch_5_4','ch_5_5','ch_6_4','ch_6_5','ch_6_6','ch_7_4','ch_7_5','ch_8_5'}
epo= proc_jumpingMeans(epo, 10);  %% brute force downsampling to 100Hz
epo= proc_selectChannels(epo, clab);


file= [EEG_IMPORT_DIR 'bci_competition_iii/tuebingen/Competition_test'];
S= load(file);
epo_te= struct('x', permute(S.X, [3 2 1]));
epo_te.y= ones([1 size(epo_te.x,3)]);
epo_te.className= {'test'};
epo_te.fs= 1000;
epo_te.t= 501:3500;
epo_te.clab= mnt.clab;
clear S

epo_te= proc_jumpingMeans(epo_te, 10);  %% brute force downsampling to 100Hz
epo_te= proc_selectChannels(epo_te, clab);

epo= proc_appendEpochs(epo, epo_te);
mnt= rmfield(mnt, 'box');
clear epo_te


fv= proc_selectChannels(epo, clab);
fv= proc_selectIval(fv, [1000 3500]);
fv= proc_fourierCourseOfBandEnergy(fv, [8 12], kaiser(50,2), 25);
fv= proc_logarithm(fv);
fv= proc_meanAcrossTime(fv);

chi= chanind(fv, 'ch_4_2', 'ch_6_6');
clf;
ci1= find(fv.y(1,:));
ci2= find(fv.y(2,:));
ci3= find(fv.y(3,:));
plots(fv.x(:,chi(1),ci1), fv.x(:,chi(2),ci1), 'rx'); hold on;
plots(fv.x(:,chi(1),ci2), fv.x(:,chi(2),ci2), 'bx');
plots(fv.x(:,chi(1),ci3), fv.x(:,chi(2),ci3), 'ko'); hold off;
legend(fv.className);

fv2= proc_selectClasses(fv, 'not','test');
C= trainClassifier(fv2, 'FD');

out= applyClassifier(fv, 'FD', C, [ci1 ci2]);
clf;
fv_out= fv2;
fv_out.x= out;
plotOverlappingHist(fv_out, 20);

out= applyClassifier(fv, 'FD', C, ci3);
sum(out<0)
out= out-median(out);
clf; hist(out, 20);
save([BCI_DIR 'tasks/bci_competition_iii/output_spectral'], 'out');


epo_te= proc_selectClasses(epo, 'test');
epo_te.y= [out<0; out>0];

figure(1)
scp= proc_baseline(epo_te, 250);
scp_rsq= proc_r_square(scp);
H= grid_plot(scp, mnt);
grid_addBars(scp_rsq, 'box','on', 'h_scale',H.scale);
figure(2)
scp= proc_baseline(proc_selectClasses(Epo,'not','test'),250);
scp_rsq= proc_r_square(scp);
H= grid_plot(scp, mnt);
grid_addBars(scp_rsq, 'box','on', 'h_scale',H.scale);


figure(1)
spec= proc_spectrum(epo_te, [3 35]);
spec_rsq= proc_r_square(spec);
spec= proc_average(spec);
H= grid_plot(spec, mnt);
grid_addBars(spec_rsq, 'box','on', 'h_scale',H.scale);
figure(2)
spec= proc_spectrum(proc_selectClasses(Epo,'not','test'), [3 35]);
spec_rsq= proc_r_square(spec);
spec= proc_average(spec);
H= grid_plot(spec, mnt);
grid_addBars(spec_rsq, 'box','on', 'h_scale',H.scale);










return









epo_te= proc_selectClasses(Epo, 'test');
epo_te.y= [out3<0; out3>0];
epo_te.className= fv.className(1:2);
sz= size(Epo.x);
sz(2)= 1;
epo_te.x= Epo.x .* repmat(opt_nrm.scale, sz);


fv= proc_selectIval(Epo, [1000 3500]);
fv= proc_jumpingMeans(fv, 100);
fv2= proc_selectClasses(fv, 'not','test');
%xvalidation(fv2, 'LDA');
%classy= select_model(fv2, model_RDA);
C= trainClassifier(fv2, classy);

out3= applyClassifier(fv, classy, C, ci3);
hist(out3, 15);
sum(out3<0)
















return


epo= Epo;
epo= proc_subtractMovingAverage(epo, 500, 'centered');
epo_memo= epo;
epo= proc_selectClasses(epo_memo, 'not','test');
sz= size(epo.x);
epo.x= permute(epo.x, [1 3 2]);
epo.x= reshape(epo.x, [sz(1)*sz(3) sz(2)]);
epo= proc_subtractMean(epo, 'dim',1);
epo= proc_normalize(epo, 'dim',1, 'policy','std');
epo.x= reshape(epo.x, [sz(1) sz(3) sz(2)]);
epo.x= ipermute(epo.x, [1 3 2]);
epo_nrm= epo;
epo= proc_selectClasses(epo_memo, 'test');
sz= size(epo.x);
epo.x= permute(epo.x, [1 3 2]);
epo.x= reshape(epo.x, [sz(1)*sz(3) sz(2)]);
epo= proc_subtractMean(epo, 'dim',1);
[epo, opt_nrm]= proc_normalize(epo, 'dim',1, 'policy','std');
epo.x= reshape(epo.x, [sz(1) sz(3) sz(2)]);
epo.x= ipermute(epo.x, [1 3 2]);
epo= proc_appendEpochs(epo_nrm, epo);
clear epo_memo











spec= proc_spectrum(epo, [3 95]);
spec_rsq= proc_r_square(proc_selectClasses(spec, 'not','test'));
spec= proc_average(spec);

H= grid_plot(spec, mnt);
grid_addBars(spec_rsq, 'box','on');

fit_ival= [55 75];
fit_iv= getIvalIndices(fit_ival, spec);

tr= mean(mean(spec.x(fit_iv,:,1:2), 3), 1);
te= mean(spec.x(fit_iv,:,3), 1);

spec.x(:,:,3)= spec.x(:,:,3) + repmat(tr-te, [size(spec.x,1) 1 1]);





band= [7 13];
[b, a]= butter(5, band/epo.fs*2);
epo_flt= proc_filt(epo, b, a);

erd= proc_rectifyChannels(epo_flt);
erd= proc_average(erd);
erd= proc_movingAverage(erd, 200);
erd= proc_baseline(erd, 250, 'beginning');
grid_plot(erd, mnt);



scp= proc_average(epo);
scp= proc_baseline(scp, 250, 'beginning');
grid_plot(scp, mnt);


spec= proc_spectrum(epo, [3 95]);
spec_rsq= proc_r_square(proc_selectClasses(spec, 'not','test'));
spec= proc_average(spec);

H= grid_plot(spec, mnt);
grid_addBars(spec_rsq, 'box','on');


