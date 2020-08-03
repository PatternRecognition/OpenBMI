function control= bbci_apply_evalControl(cfy_out, control, bbci_control, event, marker)
%BBCI_APPLY_EVALCONTROL - Evaluate control function to classifier output
%
%This function is a wrapper, that calls the function specified in
%BBCI_CONTROL.fcn (prefix 'bbci_control_' is added) to transform the
%classifier output into a control signal (PACKET), that will be sent to
%the application via UDP. The PACKET is formatted as a variable/value list
%in a CELL. If BBCI_CONTROL.fcn is empty, the generic PACKET
%{'cl_output', CFY_OUT} is returned.
%
%Synopsis:
%  CONTROL= bbci_apply_evalControl(CFY_OUT, CONTROL, BBCI_CONTROL, EVENT, MARKER)
%
%Arguments:
%  CFY_OUT - Output of the classifier
%  CONTROL - Structure for control signals,, subfield of 'data' structure of
%           bbci_apply, see bbci_apply_structures. The subfield 'state' is a
%           state variable that can be used by the specific control functions
%           to store information. In the very first call of a run of
%           bbci_apply, this field is empty.
%  BBCI_CONTROL - Structure specifying the calculation of the control signal.
%      subfield of 'bbci' structure of bbci_apply.
%  EVENT - Structure that specifies the event (fields 'time' and 'desc')
%      that triggered the evaluation of this control.
%  MARKER - Structure of recently acquired markers
%
%Output:
% CONTROL - As input argument, but with possibly updated field 'state'
%     (see above) and with the field
% 'packet': Variable/value list in a CELL defining the control signal that
%     is to be sent via UDP to the application. In the simplest case this
%     is {'cl_output', CFY_OUT}.

% 02-2011 Benjamin Blankertz


if isempty(bbci_control.fcn),
  control.packet= cfy_out;
else
  event_info= setfield(event, 'all_markers',marker);
  if nargout(bbci_control.fcn)==1,
    % old style format (no state variable)
    control.packet= bbci_control.fcn(cfy_out, event_info, ...
                                     bbci_control.param{:});
  else
    [control.packet, control.state]= ...
        bbci_control.fcn(cfy_out, control.state, event_info, ...
                         bbci_control.param{:});
  end
end

if isnumeric(control.packet) && ~isempty(control.packet),
  control.packet= {'cl_output', control.packet};
end
