function bbci= bbci_calibrate_reset(bbci, varargin)
%BBCI_CALIBRATE_RESET - Revert calibration settings to original values
%
%Synopsis:
%  BBCI= bbci_calibrate_reset(BBCI)
%  BBCI= bbci_calibrate_reset(BBCI, 'param1', 'param2', ...)
%
%Argument:
%  BBCI: Structure holding specific settings for calibration in the
%     field 'calibrate'.
%
%Output:
%  BBCI - Updated BBCI structure in which all subfields of BBCI.settings
%     which are specified in BBCI.calibrate.auto_reset are reverted to 
%     the values in BBCI.calibrate.default_settings.
%     For BBCI.calibrate.auto_reset=1 all subfields are reverted.

% 11-2011 - Benjamin Blankertz


BC= bbci.calibrate;
if ~isfield(BC, 'default_settings'),
  error('no default settings stored in BBCI structure');
end
BCS= bbci.calibrate.default_settings;

if nargin==1,
  fld_list= fieldnames(BCS);
elseif length(varargin)==1 && iscell(varargin{1}),
  fld_list= varargin{1};
else
  fld_list= varargin;
end

for ii= 1:length(fld_list),
  fld= fld_list{ii};
  bbci.calibrate.settings.(fld)= bbci.calibrate.default_settings.(fld);
end
