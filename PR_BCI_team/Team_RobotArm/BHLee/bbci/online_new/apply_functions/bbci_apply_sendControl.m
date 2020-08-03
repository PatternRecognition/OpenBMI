function bbci_apply_sendControl(control_signal, bbci_feedback)
%BBCI_APPLY_SENDCONTROL - Send control signal to application
%
%Synopsis:
%  bbci_apply_sendControl('init', BBCI_FEEDBACK)
%  bbci_apply_sendControl('close', BBCI_FEEDBACK)
%  bbci_apply_sendControl(CONTROL_SIGNAL, BBCI_FEEDBACK)
%
%Arguments:
%  CONTROL_SIGNAL - CELL: variable/value list, e.g. {cl_output, 1.253}
%  BBCI_FEEDBACK - Structure defining where and how the control signal is sent,
%      subfield of 'data' structure of bbci_apply, see bbci_apply_structures

%?? Should this function receive more information, ??
%TODO: Init UDP before, Init Matlab (and Pyff?) Feedback before,

% 02-2011 Benjamin Blankertz


% For matlab feedback only:
persistent feedback_opt

%% - INIT
if isequal(control_signal, 'init'),
  switch(bbci_feedback.receiver),
   case '',
    % do nothing
   case 'pyff',
%    send_udp_xml('init', bbci_feedback.host, bbci_feedback.port);
   case 'tobi_c',
    send_tobi_c_udp('init', bbci_feedback.host, bbci_feedback.port);
   case 'matlab',
    if isfield(bbci_feedback, 'opt'),
      feedback_opt= bbci_feedback.opt;
    else
      feedback_opt= [];
    end
    feedback_opt.reset= 1;
   otherwise,
    error('feedback.receiver unknown');
  end
  return;
end

%% - CLOSE
if isequal(control_signal, 'close'),
  switch(bbci_feedback.receiver),
   case 'pyff',
%    We do not close this channel, as it is probably still required to
%    send signals to Pyff.
%    send_udp_xml('close');
   case 'tobi_c',
    send_tobi_c_udp('close');
  end
  return;
end

%% - Operation
switch(bbci_feedback.receiver),
 case '',
  % do nothing
 case 'pyff',
   if ~isempty(control_signal),
     send_udp_xml(control_signal{:});
   end
 case 'tobi_c',
%   error('in construction');
  if ~isempty(control_signal),
    send_tobi_c_udp('send', control_signal{2}(1));
  end
 case 'matlab',
  idx= find(cellfun(@(x)isequal(x,'cl_output'), control_signal));
  cls_output= control_signal(idx+1);
  feedback_opt= bbci_feedback.fcn(feedback_opt, cls_output{:});
 case 'matlab_gui',
  error('in construction');
  ii= max(find(cellfun(@(x)isequal(x,'cl_output'), control_signal)));
  cls_output= control_signal{ii+1};
  send_data_udp([logFileNumber(1); controlnumber; timestamp; 0;
                 cls_output]);
end
