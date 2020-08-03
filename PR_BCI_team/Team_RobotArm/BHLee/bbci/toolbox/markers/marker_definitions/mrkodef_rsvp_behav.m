function mrk= mrkodef_rsvp_behav(mrko, varargin)

stimDef= {[71:100], [31:60];
          'target','non-target'};
miscDef= {254, 255, 240, 241;
          'run_start', 'run_end', 'countdown_start', 'countdown_end'};
% miscDef= { 200, 201, 105, 106;
%           'countdown_start', 'countdown_end', ...
%           'burst_start','burst_end'};
% respDef = {'R  8'; ...
%            'button'};
% 
opt= propertylist2struct(varargin{:});
% opt= set_defaults(opt, 'stimDef', stimDef, ...
%                        'respDef', respDef, ...
%                        'miscDef', miscDef);

opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'miscDef', miscDef);
                     
% mrk= mrkodef_general_oddball(mrko, 'stimDef',opt.stimDef, ...
%                              'respDef', [], ...
%                              'miscDef',opt.miscDef); 

mrk= mrkodef_general_oddball(mrko, 'stimDef',opt.stimDef, ...
                             'miscDef',opt.miscDef); 
                           
itarget= find(mrk.y(1,:));
mrk_target= mrk_chooseEvents(mrk, itarget);
% mrk_resp= mrk_defineClasses(mrko, opt.respDef);
% [tmp, istim, iresp]= ...
%         mrk_matchStimWithResp(mrk_target, mrk_resp, ...
%                               'missingresponse_policy', 'accept', ...
%                               'multiresponse_policy', 'first', ...
%                               'removevoidclasses', 0);
% mrk.latency= NaN*ones(1,length(mrk.pos));
% mrk.latency(itarget)= tmp.latency;
% mrk.multiresponse= NaN*ones(1,length(mrk.pos));
% mrk.multiresponse(itarget)= tmp.multiresponse;
% mrk.missing_response= isnan(mrk.latency);
% iTooSlow= find(mrk.latency>1500);
% mrk.missing_response(iTooSlow)= 1;
% mrk.latency(iTooSlow)= NaN;
% mrk.false_positives= sum(mrk.multiresponse) + length(iTooSlow);

mrk= mrk_addInfo_P300design(mrk, 30, 10);
mrk.stimulus= mod(mrk.toe-31,40)+1;
% mrk= mrk_addIndexedField(mrk, {'stimulus','latency','multiresponse','missing_response'});
