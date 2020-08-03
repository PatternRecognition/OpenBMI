function bvr_sendcommand(fcn, varargin)
%BVR_SENDCOMMAND - Control the BrainVision Recorder software
%
%Synopsis:
% bvr_sendcommand(FCN, <ARGS>)
%
%Arguements:
% FCN: Name of the function to be executed. See in the bvr_*.vbs files in
%     acquistion_tools folder for a list of options. Here are some:
%     'loadworkspace' - load BV workspace into the recorder; ARG: name of
%        the workspace (extension '.rwsp' may be left out)
%     'startrecording' - Start EEG recording; ARG: name of the file with
%        full path, without extension.
%     'startimprecording' - Make Impedance measurement first and start
%        recording afterward (impedance values are saved into the EEG
%        file); ARG as above.
%     'stoprecording' - Stops the recording.
%     'viewsignals' - Switch to monitoring state
%     'viewsignalsandwait' - Switch to monitoring mode and wait (unless monitoring
%        mode is already active). Example: bvr_sendcommand('viewsignalsandwait','3000');

% blanker@cs.tu-berlin.de, Jul-2007

global BCI_DIR

warning('function bvr_sendcommand overloaded for ECoG usage');

if length(varargin)==1 & isequal(varargin{1}, 'stoprecording'),
  system([BCI_DIR 'acquisition/tools/bvr_stoprecording.vbs']);
  pause(3);
  bvr_startrecording('interim_recording');
  return;
end

if isequal(varargin{1}, 'startimprecording'),
  varargin{1}= 'startrecording';
end

system([BCI_DIR 'acquisition/tools/bvr_' fcn '.vbs' sprintf(' %s',varargin{:})]);
