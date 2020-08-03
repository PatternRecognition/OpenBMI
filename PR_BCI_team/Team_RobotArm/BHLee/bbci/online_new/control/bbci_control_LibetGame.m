function [packet, state]= bbci_control_LibetGame(cfy_out, state, event, opt)
%BBCI_CONTROL_LIBETGAME - Generate control signal for the LibetGame feedback
%
%Synopsis:
%  [PACKET, STATE]= bbci_control_LibetGame(CFY_OUT, STATE, EVENT)
%
%Arguments:
%  CFY_OUT - Output of the classifier
%  STATE - Internal state variable
%  EVENT - Structure that specifies the event (fields 'time' and 'desc')
%      that triggered the evaluation of this control.
%
%Output:
% PACKET: Variable/value list in a CELL defining the control signal that
%     is to be sent via UDP to the application.
% STATE: Updated internal state variable

% 04-2011 Benjamin Blankertz


thiscall= event.all_markers.current_time;
if isempty(state),
  check_ival= [0 thiscall];
else
  check_ival= [state.lastcall thiscall];
end
state.lastcall= thiscall;

% Check behavioral response (button press)
events= bbci_apply_queryMarker(event.all_markers, check_ival, -13);
keypressed= ~isempty(events);

control= 2*(cfy_out<0) + keypressed;
packet= {'i:cl_output', control};
