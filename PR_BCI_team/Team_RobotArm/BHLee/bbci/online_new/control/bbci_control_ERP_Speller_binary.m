function [packet, state]= bbci_control_ERP_Speller_binary(cfy_out, state, event, opt)
%BBCI_CONTROL_ERP_SPELLER_BINARY - Generate control signal for ERP-based Hex-o-Spell
%
%Synopsis:
%  [PACKET, STATE]= bbci_control_ERP_Speller_binary(CFY_OUT, STATE, EVENT, <SETTINGS>)
%
%Arguments:
%  CFY_OUT - Output of the classifier
%  STATE - Internal state variable
%  EVENT - Structure that specifies the event (fields 'time' and 'desc')
%      that triggered the evaluation of this control.
%  SETTINGS - Structure specifying the following feedback paramters:
%    .nClasses - number of classes of stimuli from which to select
%    .nSequences - number of sequences (repetitions) that should be used
%        in order to determine one choice
%
%Output:
% PACKET: Variable/value list in a CELL defining the control signal that
%     is to be sent via UDP to the Speller application.
% STATE: Updated internal state variable

% 02-2011 Benjamin Blankertz


this_cue= opt.mrk2feedback_fcn(event.desc);
packet= [cfy_out this_cue];


%% The rest of this function is not required. It is only used to generate
%  the output of the selected class, to verify that the Pyff feedback is
%  doing the right thing.
if nargin<3 || isempty(opt.nSequences),
  return;
end

if isempty(state),
  state.counter= zeros(1, opt.nClasses);
  state.output= zeros(1, opt.nClasses);
end

state.counter(this_cue)= state.counter(this_cue) + 1;
state.output(this_cue)= state.output(this_cue) + cfy_out;

if sum(state.counter) >= opt.nClasses*opt.nSequences,
  idx= find(state.counter>0);  % avoid divide by zero
  state.output(idx)= state.output(idx) ./ state.counter(idx);
  [max_score, selected_class]= min(state.output);
  fprintf('[BBCI CONTROL]: Selected class: %d\n', selected_class);
  state.counter(:)= 0;
  state.output(:)= 0;
end
