%file= 'VPaa_09_01_29/imag_fbarrow_pmeanVPaa'
%file='VPjs_08_07_09/imag_fbarrowVPj';
file='VPjx_08_07_16/imag_fbarrowVPjx';

[cnt,mrk,mnt]=eegfile_loadMatlab(file);
load([EEG_RAW_DIR 'subject_independent_classifiers/ensemble_LR.mat'])
load([BCI_DIR '\online\ensemble\filters.mat'])
load([BCI_DIR '\online\ensemble\Opt.mat'])
cnt=proc_selectChannels(cnt,opt.clab);
iter=50000;
for i=1:iter
  show_time(i,iter)
  dat=[];
  % here is the polling:
  dat.x=cnt.x(i:i+3,:);
  dat.clab=cnt.clab;
  %size(dat.x)
  tic
  dat_flt=online_filtBank(dat,cont_proc.procParam{1}{1},cont_proc.procParam{1}{2});
  dat_flt;
  fv=online_linearDerivationE(dat_flt,cont_proc.procParam{2}{:});%,opt.clab,Nfilter)
  fv=proc_variance(fv);
  fv=proc_logarithm(fv);
  fv.x=fv.x';
  feature_bias(i)=mean(fv.x);
  output(i)=apply_separatingHyperplaneE(cls.C,fv.x);
  disp(sprintf('running at %f Hz',1/toc))
end