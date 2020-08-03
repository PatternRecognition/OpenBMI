function [cls, varargout] = adaptation(cls, varargin)
%[cls, bbci] = adaptation(cls, bbci, 'init')
%[cls, bbci] = adaptation(cls, bbci, ts)
%[cls, feature, bbci] = adaptation(cls, feature, bbci, ts)
%[cls, feature, cont_proc, bbci] = adaptation(cls, feature, cont_proc, bbci, ts)
%[cls, feature, cont_proc, post_proc, bbci] = adaptation(cls, feature, cont_proc, post_proc, bbci, ts)
%
% This should be used by bbci_bet_apply for an update of the 
% classifier.
%
% Only the first classifier can be updated so far (i.e., 
%  multi-classifier adaptation needs extension).
% 
% bbci.adaptation should have the following fields for this to work:
%   .running: the setting from the gui. It tells us whether adaptation 
%      is requested.
%   .policy: specifies the type of adaptation strategy. This lets the function
%      bbci_adaptation_POLICY be called, default 'schnitzel'.

%% Manage different variants of input arguments
switch(length(varargin)),
 case 2,
  [bbci, ts]= deal(varargin{:});
 case 3, 
  [feature, bbci, ts]= deal(varargin{:});
 case 4, 
  [feature, cont_proc, bbci, ts]= deal(varargin{:});
 case 5, 
  [feature, cont_proc, post_proc, bbci, ts]= deal(varargin{:});
end

%% INIT case
if ischar(varargin{end}) && strcmp(varargin{end},'init')
  global TODAY_DIR TMP_DIR
  
  if ~isfield(bbci,'adaptation')
    bbci.adaptation= struct('running',false);
  end
  bbci.adaptation= set_defaults(bbci.adaptation, ...
                                'policy', 'schnitzel');
  today_vec= clock;
  today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
  if isempty(TODAY_DIR),
    tmp_folder= TMP_DIR;
  else
    tmp_folder= TODAY_DIR;
  end
  tmpfile= [tmp_folder 'adaptation_' bbci.adaptation.policy '_' today_str];
  bbci.adaptation= set_defaults(bbci.adaptation, ...
                                'tmpfile', tmpfile);
end

%% Call specific adaptation function, depending on bbci.adaptation.policy
%% and number of input/output arguments
if bbci.adaptation.running,
  adaptation_fcn= ['bbci_adaptation_' bbci.adaptation.policy];
  switch(nargout(adaptation_fcn)),
   case 2,
    [cls, bbci]= feval(adaptation_fcn, cls, bbci, ts);
    varargout= {bbci};
   case 3, 
    [cls, feature, bbci]= feval(adaptation_fcn, cls, feature, bbci, ts);
    varargout= {feature, bbci};
   case 4, 
    [cls, feature, cont_proc, bbci]= feval(adaptation_fcn, cls, feature, cont_proc, bbci, ts);
    varargout= {feature, cont_proc, bbci};
   case 5, 
    [cls, feature, cont_proc, post_proc, bbci]= feval(adaptation_fcn, cls, feature, cont_proc, post_proc, bbci, ts);
    varargout= {feature, cont_proc, post_proc, bbci};
  end
else
  varargout= {varargin{1:end-2}, bbci};
end

%% For Future extension: restart adapation on specific marker events
if false & ~bbci.adaptation.running,    %% disabled for now
  % check if the marker tells us to restart.
  % This needs to be checked if mrk_update is used!
  [toe,timeshift] = adminMarker('query', [-100 0]);
  ind = intersect([bbci.adaptation.mrk_update{:}], toe);
  if ~isempty(ind)
    fprintf('Trigger received: %i. Restarting Adaptation.\n', toe);
    restart = true;
  end
end
