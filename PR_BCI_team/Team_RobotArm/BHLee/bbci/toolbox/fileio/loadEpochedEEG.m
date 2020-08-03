function epo=loadEpochedEEG(file, varargin)
% loadEpochedEEG - loads EEG and makes epochs
%
% epo=loadEpochedEEG(file, <opt>)
%
% INPUTS:
%  file : like 'VPcm_06_06_06/imag_lettVPcm'
%         or only 'VPcm_06_06_06' (and 'paradigm' specified as an option)
%  <opt>
%  .file_appendix   : like '_cut50'
%  .clab
%  .ival       : can be 'adaptive' for covariance=1. 
%  .filtOrder
%  .band
%  .paradigm
%  .classes
%  .covariance : calculate only the covariance within the interval.
% OUTPUT:
%  epo  : the epoched data
%
% by Ryota Tomioka Oct 2006


opt=propertylist2struct(varargin{:});
opt=set_defaults(opt,'clab', {'not','E*','Fp*','FAF*','I*','AF*'},...
                     'file_appendix' , '',...
                     'ival', [500 3500],...
                     'filtOrder',  5,...
                     'band', [7 30],...
                     'paradigm', 'imag_lett',...
                     'classes', [],...
                     'covariance', 0);

eb=filename2expbase(file);
if isempty(eb.paradigm)
  eb.paradigm=opt.paradigm;
end
file = [expbase_filename(eb) opt.file_appendix];

fprintf('paradigm [%s] classes ', eb.paradigm);
for c=1:length(opt.classes)
  fprintf('[%s]',opt.classes{c});
end

if ~isempty(opt.band)
  fprintf(' using band [%g %g]', opt.band(1), opt.band(2));
end

if strcmp(opt.ival,'adaptive')
  opt.covariance = 1;
  fprintf(' adaptive interval (covariance)\n');
else
  if opt.covariance
    fprintf(' interval [%g %g] (covariance)\n', opt.ival);
  else
    fprintf(' interval [%g %g]\n', opt.ival);
  end
end

[cnt mrk]=loadProcessedEEG(file);
if ~isempty(opt.clab)
  cnt= proc_selectChannels(cnt, opt.clab);
end
if ~isempty(opt.band)
  cnt = proc_filtButter(cnt, opt.filtOrder, opt.band);
end
if ~isempty(opt.classes)
  mrk=mrk_selectClasses(mrk, opt.classes);
end

if opt.covariance
  epo=makeCovariancedEpochs(cnt, mrk, opt.ival);
else
  epo=makeEpochs(cnt, mrk, opt.ival);
end

