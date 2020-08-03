function saved_setups = training_bbci_bet(subject_file);
%TRAINING_BBCI_BET starts a gui-based routine for training classifier 
%for bbci_bet
%
% usage: 
%  saved_setups = training_bbci_bet(subject_file);
%
% input:
%  subject_file   a name of a file in bbci_bet/subjects/ with default values
%  
% output:
%  saved_setups   name of saved setup files
%
% see also:
%   apply_bbci_bet, gui_file_setup
%
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 07/03/04
% $Id: training_bbci_bet.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

global BBCI_DIR

message_box(true);
waitForSync(0);
% INTRO
h = present_intro;

if nargin==0 
  subject_file = [];
end

% LOAD A SETUP
bbci = load_subject_file(subject_file,[BBCI_DIR 'setups/']);


waitForSync(3000);

close(h);
firsttime = true;
prepare_flag = true;
analyze_flag = false;
finish_flag = false;
saved_setups = {};

% START THE LOOP
while prepare_flag + analyze_flag + finish_flag;
  if prepare_flag
    % PREPARE GUI AND PREPARATION
    aa = gui_file_setup(bbci);
    if isnumeric(aa) & (aa==-1 | firsttime) 
      prepare_flag = false;
      analyze_flag = false;
      finish_flag = false;
    elseif isnumeric(aa)
      analyze_flag = true;
    else
      h = message_box('Load THE DATA. Please wait!!!',0);
      bbci = aa;
      
      for i = 1:size(bbci.classDef,2);
        if isempty(bbci.classDef{2,i})
          if isnumeric(bbci.classDef{1,i})
            bbci.classDef{2,i} = sprintf('%i,',classDef{1,i});
            bbci.classDef{2,i} = bbci.classDef{2,i}(1:end-1);
          else
            for j = 1:length(bbci.classDef{1,i})
              if isnumeric(bbci.classDef{1,i}{j})
                bbci.classDef{2,i} = [bbci.classDef{2,i},int2str(bbci.classDef{1,i}{j})];
              else
                bbci.classDef{2,i} = [bbci.classDef{2,i},bbci.classDef{1,i}{j}];
              end
              if j<length(bbci.classDef{1,i})
                bbci.classDef{2,i} = [bbci.classDef{2,i},','];
              end
            end
          end
        end
      end
      
      try
        bbci_bet_prepare;
        analyze_flag = true; prepare_flag = false;
        active = [];
        classes = mrk.className;
        firsttime = false;
        feature_calc = false;
        handlefigures('close');
        close(h);
      catch
        close(h);
        message_box('Wrong Setup. Please Retry!!!',1);
      end
    end
  end
  
  if analyze_flag
    % ANALYZE GUI AND ANALYSIS
    if ~isfield(bbci,'analyze')
      bbci.analyze = struct('message','');
    end
    
    [vars,ac] = gui_analyze_data(bbci,active,feature_calc);
    
    if ischar(vars)
      switch vars
       case 'prepare'
        analyze_flag = false;
        prepare_flag = true;
        handlefigures('vis','off');
        reset_flag = check_yes_or_no('Save values for later analyses?');
        if ~reset_flag
          bbci = load_subject_file([],[BBCI_DIR 'setups/']);
        end
       case 'finish'
        analyze_flag = false;
        finish_flag = true;
       case 'exit'
        analyze_flag = false;
        finish_flag = false;
        prepare_flag = false;
      end
    else
      active = ac;
      bbci = vars;
      handlefigures('vis','off');
      bbci.analyze.message = '';
      
      bbci_bet_message(1);
%      try
        bbci_bet_analyze;
        feature_calc = true;
%      catch
%        h = message_box(sprintf('Analyze Method fails with error\n%s,',lasterr),1);
%        close(h);
%        feature_calc = false;
%      end
      bbci_bet_message(0);
    end
  end
  
  if finish_flag
    %finishing
    handlefigures('vis','off');
    bbci_bet_finish;
    handlefigures('vis','on');
    finish_flag = false;
    analyze_flag = true;
  end

end

% GOODBYE
handlefigures('close');
h = message_box('Goodbye!!!',0);
pause(3);
close all;
return;




