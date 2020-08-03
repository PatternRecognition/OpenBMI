function [packet, state]= bbci_control_RSVP_Speller(cfy_out, state, event, opt)
%BBCI_CONTROL_ERP_SPELLER - Generate control signal for ERP-based Hex-o-Spell
%
%Synopsis:
%  [PACKET, STATE]= bbci_control_ERP_Speller(CFY_OUT, STATE, EVENT, SETTINGS)
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


packet= [];
thiscall= event.all_markers.current_time;

if isempty(state),
  state.counter= zeros(1, opt.nClasses);
  state.output= zeros(1, opt.nClasses);
end

if event.desc==199,
  % Feedback is requesting classifier decision
  if any(state.counter) && thiscall-state.lastcall > 1000,
    % All stimuli of the current sequence should have been evaluated.
    fprintf('\n>>> Premature classifier output requested!\n');
    fprintf('>>> Counter: %s.\n\n', vec2str(state.counter));
    packet= select_class(state.output, state.counter);
    state.counter(:)= 0;
    state.output(:)= 0;
  end
  return;
end
state.lastcall= thiscall;

this_cue= mod(event.desc-30, 40);
state.counter(this_cue)= state.counter(this_cue) + 1;
state.output(this_cue)= state.output(this_cue) + cfy_out;

if sum(state.counter) >= opt.nClasses*opt.nSequences,
  packet= select_class(state.output, state.counter);
  state.counter(:)= 0;
  state.output(:)= 0;
end



function packet= select_class(output, counter)

score= inf*output;
idx= find(counter>0);  % avoid divide by zero
score(idx)= output(idx) ./ counter(idx);
[max_score, selected_class]= min(score);
packet= {'i:cl_output', selected_class-1};
