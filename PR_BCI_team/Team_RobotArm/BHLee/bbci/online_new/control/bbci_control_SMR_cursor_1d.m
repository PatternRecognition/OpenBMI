function packet= bbci_control_SMR_cursor_1d(cfy_out, event, opt)
%BBCI_CONTROL_ERP_SPELLER_BINARY - Generate control signal for SMR cursor.
%Important: this file is needed just for simulation testing, no bbci.control is needed
%in the experiment, since the control is calculated by the feedback
%application where the classifier output is sent.
% Works with MI data acquired with matlab feedback (mrk 60 starts the control).
%Synopsis:
%  PACKET= bbci_control_ERP_Speller_binary(CFY_OUT, EVENT, <SETTINGS>)
%
%Arguments:
%  CFY_OUT - Output of the classifier
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

% 02-2011 Claudia Sannelli


persistent output this_cue flag_mrk60 this_time

opt = set_defaults(opt, 'mrk60', 0, ...
    'mrk_start', [1 2 3], ...
    'mrk_end', [11 12 13 21 22 23 31 32 33], ...
    'fix_cross', 1000);

if ismember(event.desc, opt.mrk_start)
    this_cue= 1 + mod(event.desc-11, 10);
    this_time= event.time;
    flag_mrk60= 0;
end

if isempty(flag_mrk60)
    flag_mrk60= 0;
end
if isempty(this_time)
  this_time=0;
end

if opt.mrk60
  % check whether mrk60 already arrived before starting to cumulate
  % this part is so far to produce the fb as programmed in matlab, old
  % version. The pyff version is so far like the next part
  if isequal(event.desc, 60)
    flag_mrk60= 1;
  elseif ~isempty(event.desc)
    if ismember(event.desc, opt.mrk_start) || ismember(event.desc, opt.mrk_end)
      flag_mrk60= 0;
    end
  end
  if flag_mrk60
    output= output + cfy_out;
  elseif ismember(event.desc, opt.mrk_end)
    output= output + cfy_out;
  else
    output= 0;
  end
elseif ~isempty(opt.fix_cross)
  if flag_mrk60==0
    if event.time < this_time+opt.fix_cross
      output= 0;
    elseif event.time >= this_time+opt.fix_cross
      if ~isempty(this_cue)
        flag_mrk60= 1;
        output= 0;
      else
        output= 0;
      end
    end
  else    
    output= output + cfy_out;
  end
  if ~isempty(event.desc)
    if ismember(event.desc, opt.mrk_end)
      flag_mrk60= 0;
      this_cue= [];
    end
  end
else
  if  ~isempty(event.desc)
    if ismember(event.desc, opt.mrk_start)
      flag_mrk60= 1;
      output= 0;
    elseif ismember(event.desc, opt.mrk_end)
      flag_mrk60= 0;
    end
  end
  if flag_mrk60
    output=  output + cfy_out;
  end
end
if ~isempty(event.desc) || flag_mrk60
    ishit= sign(output)/2+1.5 == this_cue;
    packet= [cfy_out this_cue output ishit];
else
    packet= [cfy_out this_cue output];
end
