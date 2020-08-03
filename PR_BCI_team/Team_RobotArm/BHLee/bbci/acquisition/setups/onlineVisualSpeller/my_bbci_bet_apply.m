function my_bbci_bet_apply(setup_list,varargin);
%BBCI_BET_APPLY is the main routine for the online application of a
%classifier for BBCI FEEDBACK. After some important initializations 
%the structure is the following:
%
% INPUT NEEDS DESCRIPTION
%
% Description: 
%
% a) read out the EEG signal. A packet of data is received every 40ms (in
%    the current BrainVision system)
%
% b) buffer the marker in a marker queue and log
% 
% c) apply the processing for continuous signals to the EEG data received
%    in to a). A number of different processings can be specified.
%
% d) Evaluate the classifiers. Each classifiers has an associated condition,
%    the classifier is only evaluated if the condition gives true. If
%    necessary the required windowings and processings are calculated.
%    Finally a cell array with the classifier outputs should appear
%
% e) If necessary get some marker output
%
% f) concatenate d) and e)
% 
% f*) log f)
%
% g) apply a post processing to get a vector of numbers
%
% g2) Feedback dependent mapping, to get udp data and protocol. Each
% feedback has its own routine that writes data and instructions for send_udp
%
% h) send to udp
%
% i) log h)
%
% j) look up for new setups (udp/tcp communication)
%
% k) reset all values
% 
% l) loop end, restart with a)
%
% 
% The loop should be interruptible by j) (e.g. some variable run,
% which can be set by j))
%
% If the loop finishes (usual or by an error) - > close all important
% connections.
%
% all important variables from outside: 
%
% bbci is a struct with fields:
%     .fs     the sampling frequency of the received eeg-data [100]
%     .logfilebase  Filename for the log file. To this filename, a
%             running numbering will be added each time bbci_bet_apply is
%             called.
%     .minDataLength  Mimimum length of the EEG data packets in
%             ms. Default: 1
%     .ringBufferSize  Length of the buffer for each cont. processing
%     .maxDataLength   Maximum length of data packets received from
%             BrainAmp. Longer packets are chopped off.
%     .maxBufLength   the maximum buffer length for classifier outputs 
%                     Default: 100
%     .mrkQueueLength  the length of the marker queue (Default 100)
%
% cont_proc is a struct array with fields (array for different cnt processing)
%     .clab        the channel names (cell array of strings)
%     .procFunc        a cell array of processing which should be applied
%                  (feval!!!). Each entry in this cell array can be a
%                  function handle or function name.
%     .procParam  regaring .proc a cell array of cells for the params
%                 
%            CALL: [cnt,state] = feval(.proc,cnt,state,.proc_param{:});
%            USE OF POINTER IMPORTANT HERE!!!
%
% cls is a struct array with fields (array for different classifiers)
%     .condition   a condition if the classifier is calculated 
%                  or is nan [true] 
%     .conditionParam  further params for the condition
%     .fv          the features to use (an integer array)
%     .applyFcn      the name of the apply routine (apply_separatingHyperplane)
%     .C           the classifier
%   FORMALLY    if condition
%                  for i = 1:length(.fv)
%                     feat{i} = get_features(cnt,feature,i,timeshift(i));
%                  end
%                  fv = cat(1,feat{:});
%                  out{.} = feval(['apply_' method],C,fv.x);
%               end
%     .integrate   an integration (1). Integrate over the most recent
%                  .integrate classifier outputs. That is, with the
%                  default value 1, no integration is performed.
%     .bias        a bias. Default value: 0
%     .scale       a scaling. Default value: 1
%     .dist        setting all absolute values smaller than dist to zero, 
%                  correct for continuity and fix +1 and -1
%     .alpha       raise all values to the power of alpha
%     .range       limit all outputs in the range interval
%     The sequence of operations is: integrate, bias, scale, dist, alpha, range
%
%      timeshift is a negative number or zero 
%      (to get a windowing with a timeshift)
%
% feature is a struct array with fields (array for different processings)
%     .cnt         the cnts to use (integer array) regarding cont_proc
%     .ilen_apply  the window lengths in msec
%     .proc        a cell array of processing which should be applied
%                  (feval!!!) 
%     .proc_param  regaring .proc a cell array of cells for the params
%          fv = feval(.proc,fv,.proc_param{:});
%
% post_proc is a string
%     .proc        a function name
%     .proc_param  further params (the old fb_opt!!!)
%
% marker_output is a struct array (for each marker output) with fields
%               .marker    marker to use 
%               .value     values to map (numeric array regarding 
%                          length of the marker cell array
%               .no_marker default value if no marker exists
%
% Guido Dornhege, 30/11/2004 
%
% $Id: bbci_bet_apply.m,v 1.14 2008/03/04 14:26:03 neuro_cvs Exp $

% Requirements for conditions:
% - Marker lesen aus Queue
% - Abhaengig von Output anderes Classifiers, arithm Operationen
% - 
% $$$ out >0
% $$$ out < 0
% $$$ out < const
% $$$ out > const
% $$$ marker65 im letzten Paket. Besser: in ms-Intervall
% $$$ marker65 vor 200ms. Features umrechnen mit Timeshift!
% $$$ wenn marker65 dann schalte fuer 500ms tot?
% $$$ Timeshifts: features 2dim anlegen, features fuer jeden Timeshift

nLetters = 0;

global general_port_fields parport_on

if isempty(parport_on), parport_on = 1;end

if nargin<1
  setup_list = 1;
end
cmb_setup = 'concat';

if mod(length(varargin),2)==1
  feedback_opt = varargin{1};
  modifications = varargin(2:end);
else
  feedback_opt = {};
  modifications = varargin;
end

new_setup = '';
specified_settings= '';
for i = 1:2:length(modifications)
  specified_settings= sprintf('%s;%s = modifications{%d};',specified_settings,modifications{i},i+1);
end

parPort_list = [];

% we are optimistic.  
run = 1; % the status of the loop (run=1 -> the loop runs, run = 0->
           % the loop should stop)
errorMessage = [];


if isempty(setup_list)
  if ~isempty(general_port_fields) & isfield(general_port_fields(1),'control')
    setup_list = general_port_fields(1).control{2};
  else
    setup_list = 12470;
  end
end

BBCI_BET_APPLY = true;

general_port_fields= set_defaults(general_port_fields, ...
                                  'feedback_receiver', 'matlab');
if isnumeric(setup_list)
  
  if setup_list<=2
    if isnumeric(setup_list)
      parPort_list = 229+setup_list;
    end
    if ~isempty(general_port_fields) & isfield(general_port_fields(1),'control')
      setup_list = general_port_fields(setup_list).control{2};
    else
      setup_list = 12471-setup_list;
    end
  end
  
  %open the listening port for interaction
  control_communication = get_data_udp(setup_list);
  
  % wait for a setup if not given
  while run & isnumeric(setup_list)
    pause(0.1);
    get_setup;
  end
  get_data_udp(control_communication);
end

% set some defaults

%%%%% DO IT IN A LOOP UNTIL EXIT
while run
  
  %%% GET THE SETUPS
  if ~iscell(setup_list)
    setup_list = {setup_list};
  end
  load(setup_list{1},'bbci')
  
%  bbci.control_port = 12471-bbci.player;
  bbci = set_defaults(bbci,'player',1);
  
  if ~isfield(bbci,'control_port')
      if ~isempty(general_port_fields) & isfield(general_port_fields(1),'control')
          bbci.control_port = general_port_fields(min(bbci.player,length(general_port_fields))).control{2};
      else
          bbci.control_port = 12471-bbci.player;
      end
  end
  if length(setup_list)>1,
    [cont_proc,feature,cls,post_proc,marker_output] = feval(['cmb_setup_' cmb_setup],bbci,setup_list{:});
  else
    [cont_proc,feature,cls,post_proc,marker_output] = bbciutil_load_setup(setup_list{1});
  end
  bbci.setup_list = setup_list;
%  fprintf('\specified settings:\n')
%  fprintf([escape_printf(specified_settings) '\n\n']);
  eval(specified_settings);
%  fprintf('\nnew_setup:\n')
%  fprintf([escape_printf(new_setup) '\n\n']);
  eval(new_setup);
  bbci
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% set all defaults and check the input%%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % All defaults were already set by bbci_bet_prepare
  
  % Possible to specify a single function instead of cell array. Can't check
  % this directly in the params, convert to cell if function is not a cell.
  for pr = 1:length(cont_proc),
    %sometimes processing is not needed
    if ~isfield(cont_proc(pr),'procFunc')
      cont_proc(pr).procFunc = {};
      cont_proc(pr).procParam = {{}};    
    end
    %it is possible to specify a single string as processing function 
    %without parameters
    if ~iscell(cont_proc(pr).procFunc),
      cont_proc(pr).procFunc = {cont_proc(pr).procFunc};
      cont_proc(pr).procParam = {cont_proc(pr).procParam};
    end
  end
  
  % make sure that ringbuffer size is not too small: Extract the largest
  % relevant window & add one second buffer
  maxIlenApply = max([feature.ilen_apply])+1000;
  if bbci.ringBufferSize<maxIlenApply,
    error('Ring buffer too small');
  end
  
  % Set defaults for post_proc. Avoid strcmp in apply loop, make sure the
  % .proc field exists:
  if ~exist('post_proc','var')
    post_proc = [];
  end
  if isempty(post_proc) | ~isfield(post_proc, 'proc'),
    post_proc.proc = [];
  end
  if ~isfield(post_proc, 'proc_param'),
    post_proc.proc_param = {};
  end
  
  % Check clab for wildcards
  flag = false;
  for pr = 1:length(cont_proc)
    flag = flag | checkWildcards(cont_proc(pr).clab);
  end
  if flag
    run = 0;
    errorMessage = 'No wildcards are allowed in clabs';
  end
  
  if ~exist('marker_output','var')
    marker_output = [];
  end
  
  bbci = set_defaults(bbci,'fb_machine',general_port_fields(1).bvmachine,'fb_port',general_port_fields(min(bbci.player,length(general_port_fields))).control{3},'other_clients',{},'other_ports',[], 'prefilt', []);
  bbci= set_defaults(bbci, 'quit_marker',[], 'start_marker',[252]);
  start_marker_received= 0;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% initialize all important communication streams %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Open the TCP port to BRAINVSION with mean filtering
  if isfield(bbci, 'filt') && ~isempty(bbci.filt)
    acquire_state = acquire_bv(bbci.fs,general_port_fields(1).bvmachine, bbci.filt.b, bbci.filt.a);
  else
    acquire_state = acquire_bv(bbci.fs,general_port_fields(1).bvmachine);
  end
  acquire_state.reconnect= 1;
  acquire_state.fir_filter = ones(1,acquire_state.lag) / acquire_state.lag;
  %%%% TODO FOR MIKIO: CHECK IF CONNECTION EXISTS %%%%
  
  %%%% map for player to the cont\_proc.clab
  for i = 1:length(cont_proc)
    for j = 1:length(cont_proc(i).clab)
      if bbci.player==2 & cont_proc(i).clab{j}(1)~='x'
        cont_proc(i).clab{j} = ['x' cont_proc(i).clab{j}];
      end
      if bbci.player==1 & cont_proc(i).clab{j}(1)=='x'
        cont_proc(i).clab{j} = cont_proc(i).clab{j}(2:end);
      end
    end
  end
  
  % Optionally, we can restrict the number of returned channels to those
  % that are actually needed in feature computation.
  allChans = unique(cat(2, cont_proc.clab));
  % Tell acquire_bv to only return the channels with these indices:
  % TODO: THE FOLLOWING DOES NOT WORK!!!!!!
% $$$   acquire_state.chan_sel = chanind(acquire_state.clab, allChans);
% $$$   acquire_state.scale = acquire_state.scale(acquire_state.chan_sel);
% $$$   % Overwrite old clab data. 
% $$$   % TOCHECK: It might be good for increased performance to
% $$$   % have something sorted here - need to check in acquire_bv
% $$$   acquire_state.clab = allChans;
  
  if isempty(bbci.fb_machine)
    if ~isempty(bbci.other_clients)
      warning('other clients not supported in this mode');
    end
    feedback_opt.reset = 1;
    feedback_opt = set_defaults(feedback_opt,'parPort',1,'log',1);
    if isfield(feedback_opt,'parPort') & feedback_opt.parPort==1
      do_set(255);
    end
   
    if isfield(feedback_opt,'graphic_master') & feedback_opt.graphic_master
      error('bbci_bet_apply is not allowed to be graphic_master');
    end
    loop_fb = true;
    run_fb = true;
    fig = figure;
  else
    
    if strcmp(general_port_fields.feedback_receiver, 'pyff'),
%      global TODAY_DIR VP_CODE
      send_xml_udp('init', bbci.fb_machine, bbci.fb_port);
%      send_xml_udp('fc-signal', 's:TODAY_DIR',TODAY_DIR, 's:VP_CODE',VP_CODE);
    else
    
      % UDP for sending data
      if isempty(bbci.other_clients)
        send_data_udp(bbci.fb_machine,bbci.fb_port);
      else
        send_data_udp(cat(2,{bbci.fb_machine},bbci.other_clients),[bbci.fb_port,bbci.other_ports]);
      end
    end
  end
  
  old_port = bbci.fb_port;
  old_machine = bbci.fb_machine;
  
   
  % UDP FOR REQUIRING INTERACTION
  control_communication = get_data_udp(bbci.control_port);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% initialize important run time variables %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %form: name = value; % documentation
  
  % Each processing has its own state variable(s). Initially, these are empty
  procState = cell([1 length(cont_proc)]);
  for i = 1:length(cont_proc)
    procState{i} = cell([1, length(cont_proc(i).procFunc)]);
  end
  % check the number of outputs for each function. If it is greater than 1,
  % it is assumed that a state variable is used. If you don't like this
  % behavior, you have to supply a field 'use_state_var'.
  if ~isfield(cont_proc,'use_state_var') | isempty(cont_proc(ii).use_state_var)
    for ii = 1:length(cont_proc)
      use_state_var = [];
      for jj = 1:length(cont_proc(ii).procFunc)
        use_state_var(jj) = (nargout(cont_proc(ii).procFunc{jj})>1);
      end
      cont_proc(ii).use_state_var = use_state_var;
    end
  end
  
  
  % Init the storage module for the processed data
  storeContData('init', length(cont_proc), length(acquire_state.chan_sel),bbci,'sloppyfirstappend',1);
  
  
  % For each processing, compute the actual channel indices from the string
  % labels
  for pr = 1:length(cont_proc)
    cont_proc(pr).chans = chanind(acquire_state.clab, cont_proc(pr).clab);
    if length(cont_proc(pr).clab) ~= length(cont_proc(pr).chans)
      run = 0;
      errorMessage = 'clabs mismatch';
    end
  end
  
  % Initialize the classifier!!!
  cls = getClassifier('init',cls,bbci);
  
  getFeature('init',feature,bbci,cont_proc);
  
  % Initialize the marker queue
  adminMarker('init',bbci);
  
  % Initialize the standardPostProc
  standardPostProc('init',cls,bbci);
  
  
  % Open the log file.
  important_log_file_vars = struct('cont_proc',{cont_proc},'cls',{cls},'feature',{feature},'post_proc',{post_proc});
  logFileNumber = writeClassifierLog('init',bbci,important_log_file_vars);
  
  % init marker Output
  marker_output = performMarkerOutput('init',bbci,marker_output);
  
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
  %%%%% other important initializations %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  [cls,bbci]=adaptation(cls,bbci,'init');
  
  if isfield(bbci,'initialize_functions')
    for bb = 1:length(bbci.initialize_functions)
      feval(bbci.initialize_functions{bb},bbci.initialize_params{bb}{:});
    end
  end					
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% THE MAIN WORKING LOOP %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  loop = true;
  controlnumber = 0;

  clabs = acquire_state.clab;
  for i = 1:length(clabs)
    if clabs{i}(1)=='x'
      clabs{i} = clabs{i}(2:end);
    end
  end
  while loop
    %%% get the EEG %%%
    % To get a data packet of desired size, build a loop here:
    data = [];
    while size(data, 1)<(bbci.minDataLength/1000*bbci.fs),
      % acquire_bv returns a data packet of size [time channels], and all
      % the marker info
%      [currData, block,markerPos,markerToken] = ...
%          acquire_bv(acquire_state);
      %%%% TODO FOR MIKIO: FUTURE VERSION: DOES IT WORK WITH DESCR????
      [currData, block, markerPos, markerToken, markerDescr] = ...
           acquire_bv(acquire_state);
      
      % TODO FOR MIKIO: Error handling: If communication is closed, default values are
      % returned (BrainAmp down). Check in Acquire_bv! 
      
      % BUFFER TIMESTAMP
      timestamp = block*1000/bbci.fs;
      
      % TRANSFER MARKERPOS CORRECTLY
      markerPos = markerPos-size(currData,1)+1;
      markerPos = markerPos*1000/bbci.fs;
      
      %%% save the marker in the marker queue %%%
      adminMarker('add',timestamp,markerPos,markerToken,markerDescr);
      
      data = cat(1, data, currData);
    end
    % For a case like Windows/Matlab/ crashed, we might receive huge data
    % packets. Keep only the last part of this data
    keep = round(bbci.maxDataLength/bbci.fs);
    if size(data, 1)>keep,
      data = data((end-keep+1):end,:);
    end

    % Make struct from data because most processings require structs.
    % for player 2 kill initial x in channel names
    
    data = struct('fs',bbci.fs,'x',data,'clab',{clabs});
    
    %%% apply the cnt processing %%%
    for pr = 1:length(cont_proc)
      theProc = cont_proc(pr);

      % Do this only on the selected subset of channels
      procData = proc_selectChannels(data, theProc.chans);
      
      % Each processing step can have a list of functions to call
      % sequentially. 
      for i = 1:length(theProc.procFunc),
        if ~theProc.use_state_var(i)
          procData = feval(theProc.procFunc{i}, procData, theProc.procParam{i}{:});
        else
          % Each processing step has its own set of state variables
          [procData,procState{pr}{i}] = ...
              feval(theProc.procFunc{i}, procData, procState{pr}{i}, theProc.procParam{i}{:});
        end
      end
      % Store via container function. This also handles the case that a huge block
      % is written (larger than ringBufferSize)
      storeContData('append', pr, procData.x);
    end % end cnt processing

    %%% calculate the classifiers %%%
    [out,out_aa] = getClassifier('apply',size(data.x,1)*1000/bbci.fs,cls);
    
    %%% if necessary perform mrk_out %%%
    mrkOut = performMarkerOutput('apply',marker_output,size(data.x,1)*1000/bbci.fs,timestamp);
    
%%% log out %%%
    writeClassifierLog('cls',timestamp,cat(2,out_aa,mrkOut));

    %%% Adapt the classifier, if necessary %%%
    [cls,bbci] = adaptation(cls,bbci,timestamp);
    
    %%% individual post_process the data. If no function is given as
    % post_proc.proc, then do what we always do. Otherwise, call a
    % feedback-dependent postprocessing
    if isempty(post_proc.proc),
      % Concatenate all classifier outputs into a huge vector. Each cell
      % entry is a column vector
      udp = vertcat(out{:}, mrkOut{:});
    else
      % Feedback-dependent postprocessing with parameters
      udp = feval(post_proc.proc, out, mrkOut, post_proc.proc_param{:});
    end
    % This routine needs to convert the postprocessed classifier output
    % to something that can be sent over upd. This routine might need
    % things like player number, so give it all the setup 
    if isfield(bbci,'feedback') & ~isempty(bbci.feedback)
      udp = feval(['bbci_bet_feedbacks_' bbci.feedback], udp, bbci);
    end
    % The feedback routine needs to return a double array

    %%% send udp%%%
    controlnumber = controlnumber+1;
    if isempty(bbci.fb_machine)
      if run_fb
        if ~loop_fb; feedback_opt.reset = 1; loop_fb = true; end
        dat = num2cell(udp);
        for bla = 1:length(parPort_list)
          do_set(parPort_list(bla));
          do_set('counter',controlnumber,timestamp,logFileNumber(1));
        end
        if isfield(feedback_opt,'type') & ~isempty(feedback_opt.type)
          feedback_opt = feval(feedback_opt.type,fig,feedback_opt,dat{:});
        end
      else
        set(fig,'Visible','off');
      end
      drawnow;
    else
      switch(general_port_fields.feedback_receiver),
       case 'matlab',
        if parport_on
          send_data_udp([logFileNumber(1); controlnumber;timestamp;length(parPort_list); parPort_list';udp]);
        else
          send_data_udp([logFileNumber(1); controlnumber;timestamp;0;udp]);
        end
       case 'pyff',
        if ~isnan(udp(1)),
          send_xml_udp('i:controlnumber', controlnumber, ...
                       'timestamp', timestamp, ...
                       'cl_output', udp);
        end
       otherwise,
        error('feedback receiver unknown');
      end
    end
    
    parPort_list = [];
    
    %%% log udp %%%
    writeClassifierLog('udp',timestamp,udp);
    
    setupchanged = 0;
    get_setup;
    %%% GIVES VARIABLE setupchanged BACK IF SOMETHING CHANGES %%%
    if setupchanged
      %%%% SEND MARKER TO PARALLEL PORT IF FEEDBACK IS CHANGED %%%%
      parPort_list = [parPort_list,229+bbci.player];
      if old_port~=bbci.fb_port | ~strcmp(old_machine,bbci.fb_machine)
        send_data_udp;
        
        if isempty(bbci.fb_machine)
          if ~isempty(bbci.other_clients)
            warning('other clients not supported in this mode');
          end
          feedback_opt.reset = 1;
          if isfield(feedback_opt,'parPort') & feedback_opt.parPort==1
            do_set(255);
          end
   
          if isfield(feedback_opt,'graphic_master') & feedback_opt.graphic_master
            error('bbci_bet_apply is not allowed to be graphic_master');
          end
          loop_fb = true;
          run_fb = true;
        else
          if isempty(bbci.other_clients)
            send_data_udp(bbci.fb_machine,bbci.fb_port);
          else
            send_data_udp(cat(2,{bbci.fb_machine},bbci.other_clients),[bbci.fb_port,bbci.other_ports]);
          end
          old_port = bbci.fb_port;
          old_machine = bbci.fb_machine;
        end
      end
      
      writeClassifierLog('change', timestamp, new_setup);
    end
    
    %% bbci_bet_apply will automatically stop when specified markers are
    %% received
    toe= adminMarker('query', [-100 0]);
    if ~isempty(intersect(toe, bbci.start_marker)),
      fprintf('[my_bbci_bet_apply:] Start marker received.\n');
      start_marker_received= 1;
      quit_counter = 0;
    end
    if ~isempty(intersect(toe, bbci.quit_marker)) & start_marker_received,
      quit_counter = quit_counter + 1;
      fprintf('[my_bbci_bet_apply:] EndLevel2 marker received.\n');
      if quit_counter == nLetters
        fprintf('[my_bbci_bet_apply:] Quitting...\n');
        loop= false;
        run= false;
        pause(5);
        bvr_sendcommand('stoprecording');
      end
    end
  end  % end while loop schleife
%catch
%  % inform about the error
%  errorMessage = lasterr;
%end


  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%%%% Close all communications %%%%%
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  % Close port to BrainVision
  acquire_bv('close');
  
  if strcmp(general_port_fields.feedback_receiver, 'pyff'),
    send_xml_udp('close');
  else
    send_data_udp; 
  end
  if exist('fig','var'), try,close(fig), end; end
  get_data_udp(control_communication);
  
end

if ~isempty(errorMessage)
  writeClassifierLog('message',timestamp, ['Error: ' errorMessage]);
  writeClassifierLog('exit',timestamp);
  error(errorMessage);
end

writeClassifierLog('exit',timestamp);
fprintf('bbci_bet finishes\n');
