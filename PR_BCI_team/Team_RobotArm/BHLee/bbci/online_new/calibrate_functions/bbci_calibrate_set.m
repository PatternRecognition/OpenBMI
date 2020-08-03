function bbci= bbci_calibrate_set(bbci, varargin)
% BBCI_CALIBRATE_SET - Set parameters for BBCI calibration
%
%bbci_calibate_set(BBCI, 'param')
%  prints the VALUE of parameter 'param' of BBCI.calibrate.settings;
%
%BBCI= bbci_calibrate_set(BBCI, 'param1', VALUE1, 'param2', VALUE2, ...)
%  sets the parameters 'param1', 'param2', ... of BBCI.calibrate.settings
%  to the values VALUE1, VALUE2, ...
%
%BBCI= bbci_calibrate_set(BBCI, DATA, 'param1', 'param2', ...)
%  copies the values of parameters 'param1', 'param2', ... from DATA.result
%  to BBCI.calibrate.settings. The calibration function should be
%  programmed such that parameters that can be automatically selected by
%  the calibration function are stored in the respective fields of
%  DATA.results. If the experimenter is satiesfied with certain parameters,
%  s/he can copy the respective values to the calibration settings such
%  they do not have to be re-selected again.
%
%Short hand: bc_set
%
%Example:
%  [bbci, data]= bbci_calibrate(bbci);
%  bbci= bc_set(bbci, data, 'band', 'ival');
%This copies the band and the time interval that was selected during
%calibration into the bbci.settings such that it will be used again in 
%next runs of bbci_calibrate without the time-consuming reselection.

% 01-2012 Benjamin Blankertz


BCS= bbci.calibrate.settings;

if length(varargin)==0,
  disp(bbci.calibrate.settings);
%  bbci_prettyPrint(bbci.calibrate.settings);
  clear bbci
  return;
end


% bbci_calibate_set(BBCI, 'Param')
if length(varargin)==1,
  param= varargin{1};
  if ~ischar(param),
    error('Argument must be a field of BBCI.calibrate.settings');
  end
  if isfield(BCS, param),
    fprintf('The value of ''%s'' is:\n', param);
%    disp(BCS.(param));
    fprintf('%s\n', toString(BCS.(param)));
  else
    fprintf('Property ''%s'' is not defined.\n', param);
  end
  if nargout>0,
    bbci= BCS.param;  % provide value as output argument
  else
    clear bbci
  end
  return
end


if isstruct(varargin{1}),
% BBCI= bbci_calibrate_set(BBCI, DATA, 'Param1', 'Param2', ...)
  data= varargin{1};
  flds= varargin(2:end);
else
% BBCI= bbci_calibrate_set(BBCI, 'Param1', Value1, ...)
  data.result= propertylist2struct(varargin{:});
  flds= fieldnames(data.result);
end
  
  
for ii= 1:length(flds),
  fld= flds{ii};
  if isfield(data.result, fld),
    bbci.calibrate.settings.(fld)= data.result.(fld);
  else
    warning(['Field <' fld '> not found in data.result']);
  end
end
