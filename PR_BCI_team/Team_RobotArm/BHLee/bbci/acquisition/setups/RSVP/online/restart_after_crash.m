
% close pyff windows
VP_CODE= 
setup_RSVP_online
VP_COUNTER_FILE= [DATA_DIR 'RSVP_online_VP_Counter'];
load(VP_COUNTER_FILE, 'VP_NUMBER');

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'],'gui',0);
pause(4)

condition_tags= {'NoColor116ms', 'Color116ms', 'Color83ms'};
order= perms(1:length(condition_tags));
conditionsOrder= order(1+mod(VP_NUMBER-1, size(order,1)),:);

phrase_practice= choose_case('QUARZ');
phrase_calibration= choose_case('BRAIN_COMPUTER_INTERFACE');
phrase_copyspelling2= choose_case('LET_YOUR_BRAIN_TALK.');
phrase_copyspelling1= choose_case('WINTER_IS_DEPRESSING');
phrase_copyspelling3= choose_case('DONT_WORRY_BE_HAPPY!');

%Trial type in RSVP speller
% 1: Count, 2: YesNo, 3: Calibration, 4: FreeSpelling, 5: CopySpelling
TRIAL_COUNT= 1;
TRIAL_YESNO= 2;
TRIAL_CALIBRATION= 3;
TRIAL_FREESPELLING= 4;
TRIAL_COPYSPELLING= 5;

jj=

  tag= condition_tags{jj};
  speller= ['RSVP_' tag];
  i= min(find(isstrprop(tag,'digit')));
  color_mode= tag(1:i-1);
  speed_mode= tag(i:end);
  bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier_', speller, VP_CODE);

% execute run-script blockwise