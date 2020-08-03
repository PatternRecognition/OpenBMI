file= 'VPcm_06_06_06/imag_lettVPcm';
[cnt,mrk,mnt]= eegfile_loadMatlab(file);
cnt= proc_selectChannels(cnt, {'not','E*','Fp*','AF*','FT9,10'});

%% Select just two classes for classification
mrk= mrk_selectClasses(mrk, 'left','right');

opt_xv= strukt('sample_fcn',{'chronKfold',8}, 'std_of_means',0);

band= [9 14; 14 20];
[filt_b,filt_a]= butters(5, band/cnt.fs*2);
cnt_flt= proc_filterbank(cnt, filt_b, filt_a);
epo= makeEpochs(cnt_flt, mrk, [750 3500]);

fv1= proc_selectChannels(epo, '*flt1');
[fv1,csp_w1]= proc_csp3(fv1, 'patterns',2);
fv2= proc_selectChannels(epo, '*flt2');
[fv2,csp_w2]= proc_csp3(fv2, 'patterns',2);
%[csp_m,csp_n]= size(csp_w1);
%csp_w= [csp_w1 zeros(csp_m,csp_n); zeros(csp_m,csp_n) csp_w2];
fv= proc_appendChannels(fv1, fv2);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
xvalidation(fv, 'LSR', opt_xv);


proc= struct('memo', 'csp_w');
proc.train= ...
    ['fv1= proc_selectChannels(fv, ''*flt1''); ' ...
     '[fv1,csp_w1]= proc_csp3(fv1, ''patterns'',2); ' ...
     'fv2= proc_selectChannels(fv, ''*flt2''); ' ...
     '[fv2,csp_w2]= proc_csp3(fv2, ''patterns'',2); ' ...
     '[csp_m,csp_n]= size(csp_w1); ' ...
     'csp_w= [csp_w1 zeros(csp_m,csp_n); zeros(csp_m,csp_n) csp_w2];' ...
     'fv= proc_appendChannels(fv1, fv2); ' ...
     'fv= proc_variance(fv); ' ...
     'fv= proc_logarithm(fv);'];
proc.apply= ...
    ['fv= proc_linearDerivation(fv, csp_w); ' ...
     'fv= proc_variance(fv); ' ...
     'fv= proc_logarithm(fv);'];

xvalidation(epo, 'LSR', opt_xv, 'proc',proc);
