function mrk= mrkodef_general_oddball(mrko, varargin)


%% Set default values
stimDef= {[21:39], [1:19];
          'target','non-target'};
respDef= {'R 16', 'R  8', 'R 24',
          'left', 'right', 'both'};
miscDef= {100, 252, 253;
          'cue off', 'start', 'end'};

%% Build opt struct from input argument and/or default values
opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'respDef', respDef, ...
                       'miscDef', miscDef, ...
                       'matchstimwithresp', 1, ...
                       'opt_match_resp', {}, ...
                       'offsetCountingResponse', [], ...
                       'maxCountingResponse', 199);

%% Get markers according to the stimulus class definitions
mrk_stim= mrk_defineClasses(mrko, opt.stimDef);

if ~isempty(opt.offsetCountingResponse),
  cmatch= cprintf('S%3d', opt.offsetCountingResponse:opt.maxCountingResponse);
  idx= strpatternmatch(cmatch, mrko.desc);
  mrk_stim.counting_response= apply_cellwise2(mrko.desc(idx), ...
          inline('str2num(x(2:end))','x')) - opt.offsetCountingResponse;
end

if isempty(opt.respDef),
  mrk= mrk_stim;
elseif opt.matchstimwithresp,
  mrk_resp= mrk_defineClasses(mrko, opt.respDef);
  [mrk, istim, iresp]= ...
      mrk_matchStimWithResp(mrk_stim, mrk_resp, ...
                            'missingresponse_policy', 'accept', ...
                            'multiresponse_policy', 'first', ...
                            'removevoidclasses', 0, ...
                            opt.opt_match_resp{:});
  ivalid= find(~mrk.missingresponse);
  if size(mrk_resp.y,1)>1,
    mrk.ishit= any(mrk.y(:,ivalid)==mrk_resp.y([1:size(mrk.y,1)],iresp));
    mrk= mrk_addIndexedField(mrk, 'ishit');
  end
else
  mrk= mrk_stim;
  mrk.resp= mrk_defineClasses(mrko, opt.respDef);
end

if ~isempty(opt.miscDef),
  mrk.misc= mrk_defineClasses(mrko, opt.miscDef);
end
