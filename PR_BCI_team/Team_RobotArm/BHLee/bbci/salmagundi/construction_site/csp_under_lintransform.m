file= 'VPcm_05_10_11/imag_lettVPcm';
[cnt,mrk]= eegfile_loadMatlab(file, 'clab',{'not','E*','Fp*'});

nC= length(cnt.clab);
B= pinv(randn(nC,nC));

band= [9 13];
ival= [750 3750];
[b,a]= butter(5, band/cnt.fs*2);
cnt= proc_filt(cnt, b, a);
mrk= mrk_selectClasses(mrk, {'left','right'});
fv= makeEpochs(cnt, mrk, ival);
fvB= proc_linearDerivation(fv, B);

[fv_csp, csp_w, la]= proc_csp3(fv, 'patterns','all');
[fvB_csp, cspB_w, laB]= proc_csp3(fvB, 'patterns','all');

%% correct for the sign:
S= diag(sign(diag(csp_w'*(B*cspB_w))));
csp2_w= cspB_w * S;

%% projected signals are equal:
fv2_csp= proc_linearDerivation(fvB_csp, S);
norm(fv_csp.x(:,:)-fv2_csp.x(:,:));

%% eigenvalues are equal
subplot(1,3,1);
plot([la-laB]);
%% filter to X are equal to B*filter to B*X.
subplot(1,3,2);
imagesc(csp_w-B*csp2_w);
colorbar;

