function r = bvr_sendcommand_python(fcn, varargin)
%BVR_SENDCOMMAND - Control the BrainVision Recorder software
%
%Synopsis:
% bvr_sendcommand(FCN, <ARGS>)
%
% state = bvr_sendcommand('getstate')
%
%Arguements:
% FCN: Name of the function to be executed. See in the bvr_*.vbs files in
%     acquistion_tools folder for a list of options. Here are some:
%     %'loadworkspace' - load BV workspace into the recorder; ARG: name of
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
%      'checkimpedances' - Swithc to impedance check
%      'pauserecording'
%       'resumerecording'
%       'getstate' - Returns the operational state of the brain vision recorder
%           The following part was copied from RecorderRemoteControl.py
%""" The return value will describe the following state:
%        0 - nothing
%        1 - view data
%        2 - view test signal
%        3 - view impedance
%        4 - record data
%        5 - record test data
%        6 - pause record data
%        7 - pause record test data
%
%        the list may be incomplete"""
%
% AUTHOR
%    Max Sagebaum
%
%    2008/09/08 - Max Sagebaum
%                   - file created 
% (c) 2005 Fraunhofer FIRST

% blanker@cs.tu-berlin.de, Jul-2007

global BCI_DIR

[status, result] = system([BCI_DIR 'acquisition/tools/bvr_matlab_python.py' sprintf(' %s %s',fcn,varargin{:})]);
if (status == 0)
  if(strcmp(fcn,'getstate'))
    r = str2double(result);
  end
else
  disp(result)
end
  
