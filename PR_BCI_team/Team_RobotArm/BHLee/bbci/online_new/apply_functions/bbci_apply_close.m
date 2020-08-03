function bbci_apply_close(bbci, data)
%BBCI_APPLY_CLOSE - Clean closing of bbci_apply
%
%This function performs several actions of closing the bbci-apply
%operations. What actions are performed depends on how many input
%arguments are given. If the function bbci_apply exists regularly
%all actions are performed.
%
%Synopsis:
%  bbci_apply_close(<BBCI, DATA>)
%
%Arguments:
%  BBCI - Structure of bbci_apply which specifies processing and 
%      classificaiton, type 'help bbci_apply_structures' for detailed
%      information about the fields of this structure.
%  DATA - Structure of bbci_apply which holds all current data
%      (and file-IDs of the log file(s)).
%
%bbci_apply_close
%    clears the persistent variables of all bbci_* functions. This is
%    in particular important for the bbci_control_* functions.
%bbci_apply_close(BBCI)
%    additionally closes the acquisition function(s) that is/are specified
%    in BBCI.source.acquire_fcn, and
%    and closes the feedback function(s) that is/are specified in
%    BBCI.feedback.
%bbci_apply_close(BBCI, DATA)
%    additionally closes the log file(s), which are identified by the
%    field DATA.log.fid.

% 03-2011 Benjamin Blankertz


if nargin>=1,
  if nargin==1,  %% DATA not provided, try closing anyway (works for most)
    for k= 1:length(bbci.source),
      bbci.source(k).acquire_fcn('close');
    end
  end
  if isfield(bbci, 'feedback'),
    for k= 1:length(bbci.feedback),
      bbci_apply_sendControl('close', bbci.feedback(k));
    end
  end
end

if nargin>=2,
  for k= 1:length(bbci.source),
%    bbci.source(k).acquire_fcn('close', data.source(k).state);
    bbci.source(k).acquire_fcn('close');
  end
  bbci_apply_adaptation(bbci, data, 'close');
  bbci_log_close(data);
  for k= 1:length(bbci.source),
    bbci_apply_recordSignals('close', data.source(k).record);
  end
else
  fclose('all');
end

% Clear persistent variables (should not be necessary)
clear bbci_*
