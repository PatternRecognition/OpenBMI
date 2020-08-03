function mrk= mrkodef_online_hexospell(mrko, varargin)

%     END_LEVEL2 = 245               # end of second hex level
%     COUNTDOWN_START = 240
%     STIMULUS = [ [11, 12, 13, 14, 15, 16] , [21, 22, 23, 24, 25, 26] ]
%     RESPONSE = [ [51, 52, 53, 54, 55, 56] , [61, 62, 63, 64, 65, 66] ]
%     TARGET_ADD = 20


stimDef= {[31:46], [11:26], [51:66];
          'target','nontarget','classified'};
respDef= {'R  1'; 'oscillator'};
miscDef= {252, 253, 240, 244, 245;
          'run_start', 'run_end', 'countdown_start', ...
          'end_level1', 'end_level2'};

mrk= mrkodef_general_oddball(mrko, 'stimDef',stimDef, ...     
                             'respDef',respDef, ...                      
                             'miscDef',miscDef, ...
                             'matchstimwithresp',0); 

                           

% mrk= mrk_addInfo_P300design(mrk, 30, 10, ...
%                             'nIntroStimuli',30, ...
%                             'nExtroStimuli',30);
% mrk.stimulus= mod(mrk.toe-31,40)+1;
% mrk= mrk_addIndexedField(mrk, 'stimulus');

% mrk_number= [NaN apply_cellwise2(mrko.desc(2:end), inline('str2num(x(2:end))', 'x'))];
% idx_count= find(ismember(mrk_number, 150 + [-20:20]));
% mrk.counting_diff= mrk_number(idx_count)-150;
