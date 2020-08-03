subdir= 'VPcm_05_10_11/';

is= min(find(subdir=='_'));
sbj= subdir(1:is-1);
[cnt,mrk,mnt]= eegfile_loadMatlab([subdir 'imag_lett' sbj]);

%% retrieve orginal settings
bbci= eegfile_loadMatlab([subdir 'imag_1drfb' sbj], 'vars','bbci');
csp = copy_struct(bbci.setup_opts, 'clab','band','ival','nPat','usedPat');
csp.filt_b= bbci.analyze.csp_b;
csp.filt_a= bbci.analyze.csp_a;
csp.classes= bbci.classes;

opt_csp= strukt('row_layout', 1, ...
                'contour_lineprop', {'LineWidth',0.3}, ...
                'title',0);

cnt= proc_selectChannels(cnt, csp.clab);
mrk= mrk_selectClasses(mrk, csp.classes);

nEvents= length(mrk.pos);
idx_train= 1:ceil(nEvents/2);
idx_test= ceil(nEvents/2)+1:nEvents;
mrk_train= mrk_selectEvents(mrk, idx_train);
mrk_test= mrk_selectEvents(mrk, idx_test);

[mrk_train, rClab]= reject_eventsAndChannels(cnt, mrk_train, [500 4500]);
cnt= proc_selectChannels(cnt, 'not',rClab);
cnt= proc_filt(cnt, csp.filt_b, csp.filt_a);
epo= makeEpochs(cnt, mrk_train, csp.ival);
[fv_csp, csp_w, csp_eig, csp_a]= ...
    proc_csp3(epo, 'patterns',3, 'scaling','maxto1');

figure(1);
plotCSPanalysis(epo, mnt, csp_w, csp_a, csp_eig, opt_csp);

strange_trial= 59;

fv_loo= proc_selectEpochs(epo, 'not',strange_trial);
[dmy, csp_w_loo, csp_eig_loo, csp_a_loo]= ...
    proc_csp3(fv_loo, 'patterns',3, 'scaling','maxto1');

figure(2);
plotCSPanalysis(epo, mnt, csp_w_loo, csp_a_loo, csp_eig_loo, opt_csp);


fv= proc_variance(epo);
fv= proc_logarithm(fv);

figure(3); clf;
opt_scalp= struct('colAx','range');
mt= mnt_adaptMontage(mnt, epo);
N= 15;
for k= 1:N, 
  suplot(N,k);
  fv_trial= proc_selectEpochs(fv,strange_trial+k-1);
  scalpPlot(mt, fv_trial.x, opt_scalp);
  title(sprintf('trial %d', strange_trial+k-1));
end

figure(4); clf;
for k= 1:N, 
  suplot(N,k);
  X= epo.x(:,:,strange_trial+k-1);
  imagesc(cov(X)); set(gca, 'XTick',[], 'YTick',[]); colorbar;
  title(sprintf('trial %d', strange_trial+k-1));
end


%% Calculate the mahalanobis distance (by Ryota)
epocv=proc_covariance(epo);
[T,d,n]=size(epo.x);
D=zeros(1,n);
for i=1:length(D)
  I=setdiff(find(fv.y([1 2]*fv.y(:,i),:)>0), i);
  Sigma=mean(reshape(epocv.x(1,:,I),[d,d,length(I)]),3);
  D(i)=trace(inv(Sigma)*reshape(epocv.x(1,:,i),[d,d]));
end
figure, plot(D);

[dmy,csp_w]= proc_csp3(epo, 'patterns',3, 'scaling','maxto1');
E=zeros(6,n);
for i=1:length(E),
  epo_loo= proc_selectEpochs(epo, 'not', i);
  [dmy,csp_w_loo]= proc_csp3(epo_loo, 'patterns',csp_w, 'scaling','maxto1',...
                             'selectPolicy','matchfilters');
  A= csp_w'*csp_w_loo./sqrt(csp_w'*csp_w)./sqrt(csp_w_loo'*csp_w_loo);
  E(:,i)= 100*(1-abs(diag(A)));
end
figure, plot(E')
  