function mrk= mrkodef_rsvp_speller(mrko, varargin)

stimDef= {[71:100], [31:70];
          'target','non-target'};
miscDef= {252, 253, 200, 201, 105, 106;
          'run_start', 'run_end', 'countdown_start', 'countdown_end', ...
          'burst_start','burst_end'};
% miscDef= { 200, 201, 105, 106;
%           'countdown_start', 'countdown_end', ...
%           'burst_start','burst_end'};
% respDef = {'R8'; ...
%   'button'};

% mrk= mrkodef_general_oddball(mrko, 'stimDef',stimDef, ...     
%                              'respDef',respDef, ...
%                              'miscDef',miscDef);  

 mrk= mrkodef_general_oddball(mrko, 'stimDef',stimDef, ...    
                             'miscDef',miscDef);  
mrk= mrk_addInfo_P300design(mrk, 30, 10, ...
                            'nIntroStimuli',30, ...
                            'nExtroStimuli',30);
mrk.stimulus= mod(mrk.toe-31,40)+1;
mrk= mrk_addIndexedField(mrk, 'stimulus');

mrk_number= [NaN apply_cellwise2(mrko.desc(2:end), inline('str2num(x(2:end))', 'x'))];
idx_count= find(ismember(mrk_number, 150 + [-20:20]));
mrk.counting_diff= mrk_number(idx_count)-150;
