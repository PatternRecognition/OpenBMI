% NIRS experiment using three kinds of imagery in a visual quiz

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

%
addpath([BCI_DIR 'acquisition/setups/labrotation2011_HadiRoohani']);
addpath([BCI_DIR 'acquisition/setups/season10']);
fprintf('\n\nWelcome to NIRS Visual Quiz experiment.\n\n');


% try
%   bvr_checkparport('type','S');
% catch
%   error('Check amplifiers (all switched on?) and trigger cables.');
% end

global TODAY_DIR
acq_makeDataFolder('multiple_folders', 1);

fprintf('Type ''run_labrotation2011_HadiRoohani'' and press <RET>\n');

sessiondir = [BCI_DIR 'acquisition/setups/labrotation2011_HadiRoohani/'];

fprintf('\n****************\nIs the trigger Cable connected???\n****************\n')

%% Initialization
% VP_SCREEN = [0 0 1920 1100]; % BIGLAB
VP_SCREEN = [0 0 1280 1024]; % Psylab
general_port_fields.feedback_receiver= 'pyff';

%% HAVE A LOOK (REMOVE)
% l= load('D:\data\bbciRaw\VPgby_12_01_10\bbci_classifier_RSVP_Color116ms_VPgby')

%% BBCI online

% Fragen:
% .CLAB in Signal oder Source??


% Wird DATA automatisch (basierend auf BBCI struct) initialisiert?


% superfluous, when the new system is default
startup_new_bbci_online;

bbci= [];

% Get the channels labels
source = {'19' '1' '76' '38' '2' '95' '23' '80' '47' '28' '85' '104' ...
      '12' '29' '86' '69'};
detector = {'Ref' 'Gnd' '20' '77' '37' '21' '78' '94' '22' '79' ...
      '53' '39' '96' '110' '3' '8' '46' '9' '103' '30' '87' '11' '10' '68' };
mnt = nirs_getMontage(source ,detector,'file','equidistant128ch');


% Source
bbci.source.acquire_fcn= @bbci_acquire_nirx;
bbci.source.log.output= 'screen';
bbci.source.min_blocklength = 3000;  
bbci.source.clab = mnt.clab;
bbci.source.acquire_param={'clab', mnt.clab};

% Marker
bbci.marker.queue_length = 10;

% Signal
bbci.signal.source=1;

% Feedback
bbci.feedback.receiver = 'pyff';

bbci.control.condition.marker=4;
% bbci.control.condition.overrun=0;
bbci.quit_condition.marker= 8;

% Log
bbci.log.output= 'screen';
% bbci.log.output= 'screen&file';
bbci.log.classifier = 1;
bbci.log.clock = 1;

% Calibrate

settings=[];
settings.visu_clab= {'1-1','5-5'};
settings.lp = .2;     
settings.doLowpass = 0;     
settings.signal = 'both';
settings.LB = 0;
settings.nShuffles=5;
settings.nFolds=10;
settings.restrict=1;
settings.innerFold=0;
settings.derivative=0;
settings.baseline=1;
settings.ival = [-2000,18000];
settings.nIvals = 5;
settings.clab = bbci.source.clab;
settings.spectra = 0;
settings.plot=0;

bbci.calibrate.settings=settings;
bbci.calibrate.fcn = @bbci_calibrate_NIRS;
bbci.calibrate.read_fcn = @nirsfile_loadRaw;
bbci.calibrate.read_param = {'source',source, 'detector', detector, ...
  'LB', settings.LB,'dist',3.8,'file','equidistant128ch','restrict', settings.restrict,'verbose',1};
bbci.calibrate.marker_fcn= @mrkodef_VisualQuiz;
bbci.calibrate.montage_fcn=[];
%bbci.calibrate.montage_param={source,detector};

% feature

% bbci.feature.fcn= {@visualQuiz_feat_extract};
% bbci.feature.param={{'nIvals',5,'signal','both','baseline',1}};

% just front? or just back?

return

if opt.justFront
    fv=proc_selectChannels(fv,front_ch);
elseif opt.justBack
    fv=proc_selectChannels(fv,back_ch);
end

justFront = 0;
justBack = 0;
front_ch={'19_20','19_37','19_Gnd','19_38','1_Ref','1_Gnd','1_20','10_77', ...
              '1_21','1_78','76_GND','76_77','76_94','76_95','38_37','38_20',...
              '38_21','38_22','38_39','38_53','2_21','2_78','2_22','2_3', ...
              '2_80','2_79','95_77','95_94','95_110','95_96','95_79','95_78', ...
              '95_78','23_22','23_3','23_39','80_79','80_3','80_96'};
          
back_ch={'47_46','28_8','28_9','85_8','85_103','85_9','104_103','104_87', ...
         '12_47','12_30','12_11','29_30','29_46','29_9','86_9','86_103', ...
              '86_87','29_11','29_10','86_10','86_68','69_68','69_87','87_104'};