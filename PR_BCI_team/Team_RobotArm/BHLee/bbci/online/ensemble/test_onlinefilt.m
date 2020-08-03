file= 'VPaa_09_01_29/imag_fbarrow_pmeanVPaa'
[cnt,mrk,mnt]=eegfile_loadMatlab(file)
C.w=[1 1 -2 -3]';
C.b=0.02;
csp_pattern=randn(4,18*62)';

load filters

for i=1:100
  dat=[];
  % here is the polling:
  dat.x=cnt.x(i:i+6,:);
  dat.clab=cnt.clab;
  %size(dat.x)
  tic
  dat_flt=proc_onlineFiltBank(dat,b_array,a_array);
  fv=proc_linearDerivation(dat_flt,csp_pattern);
  fv=proc_variance(fv);
  fv=proc_logarithm(fv);
  %out=applyClassifier(fv,'LDA',C);
  disp(sprintf('running at %f Hz',1/toc))
end
