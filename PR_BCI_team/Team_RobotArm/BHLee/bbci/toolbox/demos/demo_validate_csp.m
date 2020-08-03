%DEMO_VALIDATE_CSP
%
%Description:
% This demo shows an example of how to validate classification using
% Common Spatial Pattern (CSP) analysis. The CSP algorithm uses label
% information. So you must not
% apply CSP in advance and then do the cross validation, as this
% procedure would bias the result. Rather the CSP has to be calculated
% within the cross-validation on each training set.

% Author(s): Benjamin Blankertz, Feb 2005

file= [EEG_MAT_DIR 'Matthias_04_03_24/imag_cursMatthias'];
classes= {'left','foot'};

[cnt, mrk, mnt]= eegfile_loadMatlab(file);
%% Select the classes, which are to be classified
mrk= mrk_selectClasses(mrk, classes);

%% Apply band-pass to the continuous signals
band= [7 32];
[b,a]= butter(5, band/cnt.fs*2);
cnt_flt= proc_filt(cnt, b, a);
%% Make the epochs (750 to 3500ms post stimulus) which are to be classified.
epo= makeEpochs(cnt_flt, mrk, [750 3500]);
epo= proc_selectChannels(epo, {'not','E*','Fp*','AF*','I*','OI*',...
                              'OPO*','TP9,10','T9,10','FT9,10'});

%% First we demonstrate the WRONG VALIDATION!!!
%% The application of csp before the cross-validation use label information
%% of trials which become test trials in the subsequent cross-validation.
%% In order to be unbiased, the test trials in the cross-validation have
%% to be completely 'unseen' (i.e., unused by the algorithm). This
%% principle is violated here and causes an underestimation of the
%% generalization error.
fv= proc_csp3(epo, 3);
fv= proc_variance(fv);
fv= proc_logarithm(fv);
xvalidation(fv, 'LDA');

%% Here comes the correct way: CSP is caluclated *within* the 
%% cross-validation on each training set (proc.train). The calculated
%% filters (projection matrix csp_w) are stored in proc.memo and
%% can be used for projecting the test samples (proc.apply).
proc= struct('memo', 'csp_w');
proc.train= ['[fv,csp_w]= proc_csp3(fv, 3); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
             'fv= proc_variance(fv); ' ...
             'fv= proc_logarithm(fv);'];
xvalidation(epo, 'LDA', 'proc',proc);
