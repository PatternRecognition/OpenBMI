%function validate_bbci_classifier(varargin)

%global DATA_DIR EEG_RAW_DIR

S= load([DATA_DIR 'results/vitalbci_season1/performance']);
decent_fb= find(S.acc_f>70);
default_subdir_list= S.subdir_list(decent_fb);

%opt= propertylist2struct(varargin{:});
opt=[];
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'skip_trials', 0, ...
                 'cfydir', [EEG_RAW_DIR ...
                    'subject_independent_classifiers/'], ...
                 'cfy_template', 'ensemble_$CLSTAG', ...
                 'save_name', '', ...
                 'subdir_list', default_subdir_list, ...
                 'verbose', 0);
if isdefault.save_name,
  opt.save_name= strrep(opt.cfy_template, '$CLSTAG', '');
  is= max(find(~ismember(opt.save_name,'_')));
  opt.save_name= opt.save_name(1:is);
end
subdir_list= opt.subdir_list;

load([opt.cfydir  strrep(opt.cfy_template, '$CLSTAG', 'LR')]);
fprintf('\nUsing the following classifier:\n%s\n\n', toString(cls));
if length(cont_proc)>1,
  error('sorry, this function is restricted to length(cont_proc)==1.');
end
requ_clab= cont_proc.clab;
%14
for vp= 1:1%length(subdir_list),
  vp
  %% Load EEG Data
  subdir='VPkp_08_08_27/' %
  %subdir=subdir_list{vp};
  sbj= subdir(1:find(subdir=='_',1,'first')-1);
  file_name= [subdir '/imag_fbarrow' sbj];
  [cnt, mrk_orig, bbci]= eegfile_loadMatlab(file_name, 'clab',requ_clab, ...
					    'vars',{'cnt','mrk_orig','bbci'});
  
  %% Cut EEG signals to relevant part
  blk= blk_segmentsFromMarkers(mrk_orig, 'start_marker','S 71', ...
                               'end_marker','S254');
  if isempty(blk.ival),
    warning('\n** markers missing for %s **\n', sbj);
  else
    [cnt, blkcnt, mrk]= proc_concatBlocks(cnt, blk, mrk_orig);
  end
  
  %% Load matching classifier
  classes= bbci.classes;
  clstag= [upper(classes{1}(1)), upper(classes{2}(1))]
  if ~strcmp(clstag,'LR'),
    %break;
    continue;
  end
  
  
  if clstag=='RF',
    fprintf('bad hack for %s (classes switched).\n', sbj);
    clstag= 'FR';
    idx1= strmatch('S  1', mrk.desc);
    idx2= strmatch('S  2', mrk.desc);
    mrk.desc(idx1)= {'S  2'};
    mrk.desc(idx2)= {'S  1'};
  end
  cfyname= strrep(opt.cfy_template, '$CLSTAG', clstag);
  load([opt.cfydir cfyname]);
  %% sorry - this is needed due to a Matlab bug
  S= load([opt.cfydir cfyname], 'feature');
  feat= S.feature;
  %% --
  bbci.log= 0;
  
  procState= {cell([1, length(cont_proc.procFunc)])};
  cont_proc.use_state_var= [];
  for jj = 1:length(cont_proc.procFunc)
    cont_proc.use_state_var(jj) = (nargout(cont_proc.procFunc{jj})>1);
  end
  
  % Init the storage module for the processed data
  storeContData('init', length(cont_proc), length(cnt.clab), ...
                bbci, 'sloppyfirstappend',1);
  
  % Initialize the classifier and other functions
  cls = getClassifier('init',cls,bbci);
  getFeature('init', feat, bbci, cont_proc);
  adminMarker('init', bbci, 'log',0);
  standardPostProc('init', cls, bbci);
  %  marker_output= performMarkerOutput('init', bbci, []);
  [cls,bbci]= adaptation(cls, bbci, 'init');
  
  bci_output= [];
  bci_target= [];
  trial_no= 0;%-opt.skip_trials;
  ishit= [];
  washit= [];
  curr_mrk= [];
  blocksize= 40;  %% or bbci.minDataLength
  cnt_step= blocksize/1000*cnt.fs;
  cnt_idx= [1 cnt_step];
  block= 0;
  loop= true;

  while loop,   
    
    %%% get the EEG %%%
    block= block + cnt_step;
    timestamp = block*1000/bbci.fs;    
    data= struct('fs', bbci.fs, ...
                 'x', cnt.x([cnt_idx(1):cnt_idx(2)], :), ...
                 'clab', {cnt.clab});
    
    %%% save the marker in the marker queue %%%
    mrk_idx= find(mrk.pos>=cnt_idx(1) & mrk.pos<=cnt_idx(end));
    markerPos= ( mrk.pos(mrk_idx)-cnt_idx(end)+1 ) * 1000/cnt.fs;
    markerToken= mrk.desc(mrk_idx);
    markerDescr= mrk.type(mrk_idx);
    adminMarker('add', timestamp, markerPos, markerToken, markerDescr);
    toe= apply_cellwise2(markerToken, inline('str2double(x(2:end))','x'));
    %   if ~isempty(toe)
    %       toe
    %     end
    

    cnt_idx= cnt_idx + cnt_step;
    if cnt_idx(end) > size(cnt.x,1),
      loop= false;
    end

    %%% apply the cnt processing %%%
    theProc = cont_proc;
    procData = data;
    
    % Each processing step can have a list of functions to call
    % sequentially. 
    for i = 1:length(theProc.procFunc),
      if ~theProc.use_state_var(i)
        procData = feval(theProc.procFunc{i}, procData, ...
                         theProc.procParam{i}{:});
      else
        % Each processing step has its own set of state variables
        [procData, procState{1}{i}] = ...
            feval(theProc.procFunc{i}, procData, procState{1}{i}, ...
                  theProc.procParam{i}{:});
      end
      %    if i==1
      % 	sprintf('data size after filterbank ')
      % 	size(procData.x)
      %       else
      % 	sprintf('data size after csp')
      % 	size(procData.x)
      %       end
    end
    storeContData('append', 1, procData.x);
    mean(cls.C.b);
    [out,out_aa] = getClassifier('apply', size(data.x,1)*1000/bbci.fs, cls);
    %out{1}=out{1}-2;
    %out;
    %out_aa;
    %    mrkOut = performMarkerOutput('apply', marker_output, ...
    %                                 size(data.x,1)*1000/bbci.fs, timestamp);
    
    [cls,bbci]= adaptation(cls, bbci, timestamp);
    ind_startoftrial= intersect([bbci.adaptation.mrk_start], toe);
    if ~isempty(toe)
      toe;
      bbci.adaptation.mrk_start;
      intersect([bbci.adaptation.mrk_start], toe);
    end
    
    if ~isempty(ind_startoftrial),
      if opt.verbose,
        fprintf('[validate:] Trigger received: %s\n', vec2str(toe));
      end
      curr_mrk= 1 + ~isempty(intersect(bbci.adaptation.mrk_start(2), toe));
      trial_no= trial_no + 1;
      trial_t= 0;
      if trial_no>0,
        bci_target(trial_no)= curr_mrk;
      end
    end
    if isempty(curr_mrk) | trial_no<=0,  %% not inside a trial or initial trial
      continue;
    end
    trial_t= trial_t + 1;
    bout= vertcat(out{:});
    %mean(bout)
    if trial_t<25,  %% strictly: wait for Marker 'S 60'
      bci_output{trial_no}(trial_t)= 0;
    else
      bout;
      bci_output{trial_no}(trial_t)= bci_output{trial_no}(trial_t-1) + bout;
      bout;
       bci_output{trial_no}(trial_t);
    end
    bbci.adaptation;
    ind_endoftrial= intersect(bbci.adaptation.mrk_end, toe);
    if ~isempty(ind_endoftrial),
      if opt.verbose,
        %fprintf('[validate:] Endmarker: %s\n', vec2str(toe));
 
      end
      
      ishit(trial_no)= sign(bci_output{trial_no}(trial_t))/2+1.5 == curr_mrk;
      washit(trial_no)= any(ismember(toe, [11 12]));
         fprintf('|raw %3.1f|out %i |y %i',...
	      bci_output{trial_no}(trial_t)/100, ...
	      sign(bci_output{trial_no}(trial_t)), ...
	      (curr_mrk-1.5)*2)
   
      fprintf('|%03d:%03d|%03d:%03d|\n', ...
              sum(ishit), sum(ishit==0), sum(washit), sum(washit==0));
      curr_mrk= [];
    end
  end
  bci_control(vp).target= bci_target;
  bci_control(vp).trace= bci_output;
  hit_rate(vp)= 100*mean(ishit);
  hit_orig(vp)= 100*mean(washit);
  storeContData('cleanup');
  standardPostProc('cleanup');
  %end
  if ~isempty(opt.save_name),
    save([DATA_DIR 'results/ensemble/vp',num2str(vp),'.mat'],'hit_rate', 'hit_orig', 'bci_control', 'ishit','washit');
  end
end
