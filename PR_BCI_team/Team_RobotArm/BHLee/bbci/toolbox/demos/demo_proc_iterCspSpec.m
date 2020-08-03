% ryotat

warning off PROC_LOG_PROJECT_CSSP2:winOvlp
%epo=loadEpochedEEG('VPcm_06_06_06/imag_lettVPcm','classes',{'left','right'});
[cnt, mrk]= eegfile_loadMatlab('VPcm_06_06_06/imag_lettVPcm');
mrk= mrk_selectClasses(mrk, {'left','right'});
epo= cntToEpo(cnt, mrk, [750 3750]);
[epoTr epoTe]=proc_splitSamples(epo, [1 1]);
[fv,W,F,la]=proc_iterCspSpec(epoTr, 20, 3, 128, [7 30], 0, 1);
C=trainClassifier(fv,'LDA');

fvTe= proc_log_project_cssp2(epoTe, W, F);
out=applyClassifier(fvTe,'LDA',C);
err=mean(loss_0_1(fvTe.y, out))
