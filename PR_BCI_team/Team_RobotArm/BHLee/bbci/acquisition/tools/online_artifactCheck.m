function online_artifactCheck(varargin)
%
% online_artifactCheck_irene(<opt>);
%
%  new options added by irene(06/08):
%  opt:
%  .trial_wise          - 0: variance is calculated for intervals of length
%                         opt.winlen while eeg is recorded, breaks excluded.
%                         1: variance is calculated for each trial specified by
%                         'trial_start_marker' and 'trial_end_marker' or
%                         opt.trial_length,
%
%  .trial_start_marker  - vector of startmarkers, default [1 2 3 4], only
%                         used if opt.trialwise=1; 
%  
%  .trial_end_marker    - vector of endmarkers, default [100 11 12 21 22]
%  .run_start_marker    - marker that indicates the beginning of a run,
%                         default 'S252', only used if opt.trialwise=0; 
%  .run_end_marker      - marker that indicates teh end of a run, default 'S253' 
%  .pause_start_marker  - marker that indicates the beginning of a break, default 'S249' 
%  .pause_end_marker    - marker that indicates the end of a break, default 'S250'
%  .winlen              - length of interval for which variance is
%                         computed, only if opt.trialwise = 0;
%  .time_between_trials - if time between two trial startmarkers is longer than
%                         opt. time_between_trials, a break is indicated by
%                         a black column, default 1000.
%
% for simulation:
% send-data (matlab 1), receive_data.m (matlab 2) in
% .../bbci/investigation/personal/irene/artifact_check

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'bv_host', 'localhost', ...
                  'fs', 100, ...
                  'whiskerperc', 10, ...
                  'whiskerlength', 3, ...
                  'refresh', 200, ...
                  'clab', {'not','E*'}, ...
                  'trial_start_marker',1:4, ...
                  'trial_end_marker',[100 11 12 21 22], ...
                  'run_start_marker','S252', ...
                  'run_end_marker','S253', ...
                  'pause_start_marker','S249', ...
                  'pause_end_marker','S250', ...
                  'trialwise',0, ...
                  'time_between_trials', 1000, ...
                  'winlen', 100);
               


state= acquire_bv(opt.fs, opt.bv_host);
dispclab= chanind(state.clab, opt.clab);
iClab= chanind(state.clab, scalpChannels);
iClab= iClab(find(ismember(iClab, dispclab)));
nChan= length(iClab);
majorClab= chanind(state.clab(iClab), '*z');
minorClab= setdiff(1:nChan, majorClab);

clf;
set(gcf, 'Color',[1 1 1]);
global fig_char
set(gcf, 'KeyPressFcn','global fig_char; fig_char= [fig_char,double(get(gcbo,''CurrentCharacter''))];set(gcbo,''CurrentCharacter'','' '');');
nCols= 51;
cmap= [0 0 0; jet(nCols); 1 1 1];
colormap(cmap);
set(gca, 'TickLength',[0 0]);

X= zeros(0, nChan);
winlen=opt.winlen;
bti=[];
Xv=[];
m_type=cell(0);
m_pos=[];
start_end_pos=zeros(2,0);
last_nWin= NaN;
running=0; 
pausing=0;

waitForSync;
%for i=1:150
%    i
while ~ismember(27, fig_char),  %% do until ESC is pressed
  waitForSync(opt.refresh);
  fig_char= [];
  [data,bn,mp,mt,md]= acquire_bv(state); 
  
  %% alternative 1: calculate variance trialwise
  if (opt.trialwise)
    
    num_mark=size(mp,2);
 
    % store marker types
    if (~isempty(mt))
        m_type{end+1}=mt;
    end
    m_type=cell_flaten(m_type);
  
    % store absolute marker positions
    m_pos_temp=mp+ size(X,1);
    m_pos=[m_pos m_pos_temp];
  
    X= cat(1, X, data(:,iClab));
 
    % convert m_type into numbers as in mrk.toe (for convenience)
    toe=zeros(1,num_mark);
    for k=1:num_mark
      mty=mt{k};
      if (strmatch('S',mty))
          marker_type=1;
      elseif (strmatch('R',mty))
          marker_type=-1;
      else
          marker_type=NaN;
      end 
      event_type=str2num(mty(2:end));
      if ~isempty(event_type)
          toe(k)=marker_type*event_type;
      end
    end 
  
  
    % get indeces of trial_start_marker
    trial_start=[];
    for i=1:size(opt.trial_start_marker,2)      
        trial_start=[trial_start (find(toe==opt.trial_start_marker(i)))]; 
    end
    trial_start=sort(trial_start);
 
    %look for end marker before first start marker (completion of last trial of preceding data block)
  
    if ~isempty(trial_start)
        prec_trial_markers=toe(1:(trial_start(1)-1));
    else
        prec_trial_markers=toe;
    end
    tail=find(ismember(prec_trial_markers,opt.trial_end_marker));
  
    % if there was an unfinished trial at the beginning of the data block
    % replace 0 with position of trial_end_marker
    if ~isempty(tail)
       if (start_end_pos(end)==0)
       start_end_pos(end)=m_pos_temp(tail);
       end    
    end    
  
 
     % store positions of start markers in first row of start_end_pos. look for end_markers between two start
     % markers and add positions in second row of start_end_markers
     if isfield(opt,'trial_length')
     start_end_pos=[start_end_pos [m_pos_temp(trial_start);m_pos_temp(trial_start)+opt.trial_length]];
     else  
     for l=1:length(trial_start)
      
      start_pos=m_pos_temp(trial_start(l));
      start_end_pos=[start_end_pos [start_pos;0]];
      
      if (l==size(trial_start,2))
          snum=size(toe,2);
      else
          snum=trial_start(l+1);
      end    
  
      for m=trial_start(l):snum
          if (ismember(toe(m),opt.trial_end_marker)) 
              end_pos=m_pos_temp(m);
              start_end_pos(end)=end_pos;
              
              break
          end    
          
      end
    end
     end  
 
       
   % take only complete trials
   complete_trials=start_end_pos(:,(start_end_pos(2,:)~=0));
   complete_trials=complete_trials(:,(complete_trials(2,:)<=size(X,1)));
   
   % check intervals between trials
   bti=find(diff(complete_trials(1,:))>opt.time_between_trials);
  
   % calculate variance for each complete trial
   for k=1:size(complete_trials,2)
      Xv(:,k)= var(X(complete_trials(1,k):complete_trials(2,k),:),1);
   end
 
   nWin=size(Xv,2);
 
%%alternative 2: calculate variance for intervals of fixed size between
%%start and end marker, breaks excluded
 else %if ~opt.trialwise
    if (~running&&~pausing)
      %if there's a startmarker, get data after start_marker
       if (~isempty(strmatch(opt.run_start_marker,mt)))
          running=1;
          sp=mp(strmatch(opt.run_start_marker,mt));
          X= cat(1, X, data(sp:end,iClab));
       end
    
    elseif (running&&~pausing)
      %if there's a pausemarker ignore data after
      %that and insert dark column
       if (~isempty(strmatch(opt.pause_start_marker,mt)))
          pausing=1;
          
          sp=mp(strmatch(opt.pause_start_marker,mt));
          X= cat(1, X, data(1:sp,iClab));
          bti=[bti floor(size(X,1)/winlen)];
     
       elseif (~isempty(strmatch(opt.run_end_marker,mt)))
           %if there's a endmarker ignore data after
           %that and return
          running=0;
          pausing=0;
          sp=mp(strmatch(opt.run_end_marker,mt));
          X= cat(1, X, data(1:sp,iClab));
          return
       else   
          X= cat(1, X, data(:,iClab));
       end 
      
    elseif (running&&pausing)
        % if there's a pause-endmarker continue 
        if (~isempty(strmatch(opt.pause_end_marker,mt)))
          pausing=0;
          running=1;
          sp=mp(strmatch(opt.pause_end_marker,mt));
          X= cat(1, X, data(sp:end,iClab));
        elseif (~isempty(strmatch(opt.run_end_marker,mt)))
          
          return
        end  
    end  
  
  nWin= floor(size(X,1)/winlen);  
  if  (nWin>0) 
    %% calculate matrix of short term variances
    Xw= reshape(X(1:nWin*winlen,:), [winlen nWin nChan]);
    Xv= squeeze(var(Xw, 1))';
  end
  end
 
 
  %doesn't work yet
  % display only opt.number_of_columns trials
  %if (nWin>opt.number_of_columns)
  %    Xv=Xv(:,nWin-opt.number_of_columns+1:end);
  %end    
  %nWin= floor(size(X,1)/winlen);
  
  % display variance
  if ~isequal(nWin, last_nWin) && nWin>0,
    Vint_temp =[];  
    mi= min(Xv(:));
    peak= max(Xv(:));
    perc= percentiles(Xv(:), [0 100] + [1 -1]*opt.whiskerperc);
    thresh= perc(2) + opt.whiskerlength*diff(perc);
    ma= max(Xv(find(Xv < thresh)));
    Vint= 2 + floor(nCols*(Xv-mi)/(ma+1e-2-mi));
    
 % insert black columns to show breaks between trials  
    if~isempty(bti)
        Vint_temp= [Vint(:,1:bti(1)) ones(nChan,1)];
        if (length(bti)>1)         
             for v=2:length(bti)
                 Vint_temp=[Vint_temp Vint(:,(bti(v-1)+1):bti(v)) ones(nChan,1)]; 
             end
        end  
     
        Vint_temp=[Vint_temp Vint(:,bti(end)+1:end)];
        Vint=Vint_temp;
    end
     
    Vdisp= ones([nChan+4 nWin+4+size(bti,2)]);
    Vdisp(3:end-2, 3:end-2)= Vint;
    image([-1:size(Xv,2)+2], [1:nChan+4], Vdisp);
    axis_yticklabel(state.clab(iClab(minorClab)), 'ytick',2+minorClab, ...
                    'hpos', -0.01, 'color',0.9*[1 1 1], 'fontsize',7);
    axis_yticklabel(state.clab(iClab(majorClab)), 'ytick',2+majorClab, ...
                    'hpos', -0.01, 'color',[0 0 0], ...
                    'fontsize',10, 'fontweight','bold');
    xTick= get(gca, 'XTick');
    xTick= setdiff(xTick, 0);
    if (opt.trialwise)
        xlabel('trials');
    end    
    set(gca, 'XTick',xTick, 'YTick',[]);
    

    drawnow;
  end
  last_nWin=nWin;
  end

