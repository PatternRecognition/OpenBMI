function onlineScalpAndSpectra(varargin)
% onlineScalpAndSpectra(opt)
% 
% IN: opt  - struct specifying 
%            .colormap  - string description of the colormap (see
%            get_colormap)
%            .range     - a range in which the data are supposed to be.
%            .target_fs - sampling rate.
%            .var_length - array specifying the regions for variance
%            calculation of the EEG: variance over var_length(2) ms
%            is subtracted from variance over var_length(2) ms.
%

% TODO: extended help (installation of OpenGL, Java etc)
%
% 1. Find the directory of your Java Runtime Enviroment from Matlab (e.g.
% $matlabroot\sys\java\jre\win32\jre), ScalpPlot works fine with jre1.3
%
% 2. Install gl4java (www.jausoft.com/gl4java/) and use the directory from
% above, perhaps you must mkdir the lib\ext directory
%
% 3. Set the Classpath, in Matlab 6 edit classpath.txt and paste
% "$BCI_DIR\eegVisualize\ScalpPlot.jar" and restart Matlab
% in Matlab 7 exist dynamic classpath setting
% 
% 4. now it should work.
%
% (5. After the installation of Java SDK (Software Development Kit),
% don't forget to set the PATH to the $j2sdk\bin directory,
% install gl4java in $j2sdk\jre)
%
% Parameter:
% (help setElectrodeMontage)

global EEG_TEMP_DIR
if length(varargin)>1
    opt = propertylist2struct(varargin{:});
elseif length(varargin)>0
     opt = varargin{1};
else    
    opt = struct;
end 
% initialization of the arguments
opt = set_defaults(opt,'colorMap','jet',...
  'range',[-2 2],...
  'target_fs',100,...
  'buffer_length',10000,... 
  'var_length',[2000 30000],...
  'title',{{'BBCI Visualization','Michael'},...
      {'BBCI Visualization','Guido'}},...
  'spec_length',3000,...
  'host','brainamp',...
  'cls_file',{'imag_Michael_setup_001.mat',...
      'imag_Guido_setup_001.mat'},...
  'disp_csp',{[1],[1]},...
  'fft_length',512,...
  'spec_Hz_disp',[5 33],...
  'var_band',[8 15],...
  'spec_band',[8 30],...
  'wincount',5,...
  'ar_order',4,...
  'fps',25,...
  'matlab_figure',1,...
  'java_figure',1,...
  'spec_ylim',[-30 30],...
  'spec_ytickdelta',10,...
  'time_ylim',[-20 20],...
  'time_ytickdelta',10,...
  'integrate',5,...
  'fig_position',{[0 -20],[1000 -20]},...
  'fig_javacorrection',[8 5],...
  'fig_size',[280,790],...
  'fig_bgcolor',[0 0 0],...
  'fig_textcolor',[1 1 1],...
  'com_port',12345);
% Check for classifier setup file. Try looking in EEG_TEMP_DIR.
for ii = 1:length(opt.cls_file)
    if ~exist(opt.cls_file{ii},'file')
        opt.cls_file{ii} = [EEG_TEMP_DIR opt.cls_file{ii}];
    end
end

% Establish communication to the amplifier
bbciclose
state = acquire_bv(100,opt.host);
% Establish communication to the feedback
com_hand = get_data_udp(opt.com_port);
% Initialize current number of players
curr_playernum = 1;

% get the colormap
colMap = loadColormap(opt.colorMap);

% determine how many players are involved
if ~isempty(strfind([state.clab{:}],'x'))
    % two players
    playernum = 2;
else
    % one player only
    playernum = 1;
end 

if opt.matlab_figure
  close all
end

dat_chans = {};
% Prepare windows for spectral estimation:
spec_ind = round(opt.fft_length/100*opt.spec_Hz_disp);
spec_ind = spec_ind(1):spec_ind(2);
[dum,f] = pyulear(zeros(1,opt.spec_length/1000*100),opt.ar_order,...
  opt.fft_length,100);
spec_num = length(opt.disp_csp{1});
Pxx = cell(spec_num,playernum);

% generate ticklabels for spectra
spec_f= f(spec_ind);
spec_tick_ind = zeros(size(spec_f));
spec_tick_lab = [];
for ii = 0:5:(max(opt.spec_Hz_disp+1))
  %find the closest indices to multiples of 5.
  if sum(spec_f>ii)&sum(spec_f<ii)
    % generate a tick index.
    [dum,mi] = min(abs(spec_f-ii));
    spec_tick_ind(mi)=1;
    spec_tick_lab(end+1) = ii;
  end
end

%generate ticklabels for timecourse
time_f = (-opt.spec_length/1000*100+1):0;
time_tick_ind = zeros(size(time_f));
time_tick_lab = [];
for ii = length(time_f):(-1):1
  %find the closest indices to multiples of 100.
  if mod(time_f(ii),100)==0
    % generate a tick index.
    time_tick_ind(ii)=1;
    time_tick_lab(end+1) = time_f(ii)/100;
  end
end
time_tick_lab = time_tick_lab(end:(-1):1);

for i = 1:playernum
  if i==1
    % load the classifier logfile
    try 
      s = load(opt.cls_file{1});
    catch
      error(['Loading ' opt.cls_file{1} ' failed.']);
    end
    %opt.clab = s.cont_proc.clab;
    csp_projection{i} = s.cont_proc.procParam{i}{1};
    csp_projection{i} = csp_projection{i}(:,opt.disp_csp{i});
    % player one: select only the non-x-channels
    %clab = {state.clab{chanind(state.clab,{'not','x*'})}};
    clab = {state.clab{chanind(state.clab,s.cont_proc.clab)}};
    if length(clab)~=length(s.cont_proc.clab)
      error('Less channels than specified in this electrode setup!');
    end
    % kick out some values which are not in the grid:
    mnt(i) = setElectrodeMontage(clab,'C3,C4');
    ind = find(~isnan(mnt(i).x));
    mnt(i).clab = {mnt(i).clab{ind}};
    mnt(i).x = mnt(i).x(ind);
    mnt(i).y = mnt(i).y(ind);
    
    clab = {clab{ind}};
    chans{i} = chanind(state.clab, clab);    
  else
    % load the classifier logfile
    try 
      s = load(opt.cls_file{2});
    catch
      error(['Loading ' opt.cls_file{2} ' failed.']);
    end
    %opt.clab = s.cont_proc.clab;
    csp_projection{i} = s.cont_proc.procParam{1}{1};
    csp_projection{i} = csp_projection{i}(:,opt.disp_csp{i});
    %for jj=1:length(opt.clab)
    %  if ~strcmp(opt.clab{jj},'not')&~strcmp(opt.clab{jj}(1),'*')&~strcmp(opt.clab{jj}(1),'x')
    %    % add an x
    %    opt.clab{jj} = ['x' opt.clab{jj}];
    %  end
    % end
    % player two: select only the x-channels.
    %xclab = {state.clab{chanind(state.clab,{'x*'})}};
    xclab = {state.clab{chanind(state.clab,s.cont_proc.clab)}};
    if length(xclab)~=length(s.cont_proc.clab)
      error('Less channels than specified in this electrode setup!');
    end
    clab = {};
    for j = 1:length(xclab)
      % remove trailing x.
      clab{j} = [xclab{j}(2:end)];
    end 
    mnt(i) = setElectrodeMontage(clab,'C3,C4');
    % kick out some values which are not in the grid:
    ind = find(~isnan(mnt(i).x));
    mnt(i).clab = {mnt(i).clab{ind}};
    mnt(i).x = mnt(i).x(ind);
    mnt(i).y = mnt(i).y(ind);
    
    xclab = {xclab{ind}};
    chans{i} = chanind(state.clab, xclab);
  end
  
  % triangulation will be given to the visualisation.
  x = mnt(i).x;
  y = mnt(i).y;
  tri = delaunay(x,y);
  
   
  if opt.matlab_figure
    %for test purposes only: matlab output.
    h_fig{i} = figure(i);
    clf
    h_par = get(get(h_fig{i},'Parent'),'ScreenSize');
    set(h_fig{i},'DoubleBuffer','on',...
      'MenuBar','none',...
      'Position',[opt.fig_position{i}(1), h_par(4)-opt.fig_position{i}(2)-opt.fig_size(2),...
        opt.fig_size(1) opt.fig_size(2)-opt.fig_size(1)],...
      'Color',opt.fig_bgcolor);
    for jj = 1:spec_num
      %spectra
      h_ax{i,jj,1} = subplot(2,spec_num,2*(jj-1)+1);
      hold on;
      lin{i,jj,1} = plot([spec_f(1) spec_f(end)],[0 0],'--');
      p{i,jj,1} = plot(spec_f,zeros(1,length(spec_ind)));
      tit{i,jj,1} = title('Frequency Spectrum');
       set(h_ax{i,jj,1},...%'LineWidth',2,...
        'XColor',opt.fig_textcolor,...
        'YColor',opt.fig_textcolor,...
        'Color',opt.fig_bgcolor);
      set(lin{i,jj,1},'Color',opt.fig_textcolor,...
        'LineWidth',.5);
      set(p{i,jj,1},'Color',opt.fig_textcolor,...
        'LineWidth',2);
      set(tit{i,jj,1},'Color',opt.fig_textcolor);
      axis([min(spec_f) max(spec_f) opt.spec_ylim]);
      yl{i,jj,1}=ylabel('[dB]');
      set(yl{i,jj,1},'Position',get(yl{i,jj,1},'Position')+[1 0 0]*2);
      xl{i,jj,1}=xlabel('[Hz]');
      %timecourse
      h_ax{i,jj,2} = subplot(2,spec_num,2*(jj-1)+2);
      hold on;
      lin{i,jj,2} = plot([-opt.spec_length/1000 0],[0 0],'--');
      p{i,jj,2} = plot((-opt.spec_length/1000+.001):.01:0,...
        ones(1,opt.spec_length/1000*opt.target_fs));
      tit{i,jj,2} = title('Time Course');
      set(h_ax{i,jj,2},...%'LineWidth',2,...        'XColor',opt.fig_textcolor,...
        'YColor',opt.fig_textcolor,...
        'Color',opt.fig_bgcolor);
      set(lin{i,jj,2},'Color',opt.fig_textcolor,...        'LineWidth',.5);
      set(p{i,jj,2},'Color',opt.fig_textcolor,...        'LineWidth',1);
      set(tit{i,jj,2},'Color',opt.fig_textcolor);
     axis([-opt.spec_length/1000+.001 0 opt.time_ylim]);
     yl{i,jj,2}=ylabel('[\mu{}V]');
     set(yl{i,jj,2},'Position',get(yl{i,jj,2},'Position')+[1 0 0]*.25);
     xl{i,jj,2}=xlabel('[sec]');
    end
    %keyboard
   end

  if opt.java_figure==1
    % get a new graphics object
    javaPlot{i} = jScalpPlot(opt.title{i}{:},350,350);
    % initialize some values of the object
    javaPlot{i}.initScalp(colMap,[mnt(i).x,mnt(i).y],tri,opt.range);
    pause(.5);%allow for drawing and generation of GLcontext.
    javaPlot{i}.setSize(java.awt.Dimension(opt.fig_size(1)+opt.fig_javacorrection(1),...
        opt.fig_size(1)));
    javaPlot{i}.setLocation(opt.fig_position{i}(1)-opt.fig_javacorrection(2),...
        opt.fig_position{i}(2));
  elseif opt.java_figure==2
    colMap = colMap';
    colMap = single(colMap(:));

    jBCI{i} = de.fhg.first.jBCIPlot.BCIFrame;
    jProp{i} = de.fhg.first.jBCIPlot.BCIPlotProperties;
    jTrans{i} = de.fhg.first.jBCIPlot.TransferObject;
    
    % initialize some values of the object
    %Scalp Object
    jProp{i}.colorMap = colMap;
    jProp{i}.scalpMap = [mnt(i).x,mnt(i).y];
    jProp{i}.delaunay = tri;
    jProp{i}.scalpRange = opt.range;
    jProp{i}.scalpName = 'Alpha Band Power';
    %Spec Object
    jProp{i}.specName = 'Frequency Spectrum';
    jProp{i}.specXTicks = spec_tick_ind(:)';
    jProp{i}.specXLabel = spec_tick_lab;
    jProp{i}.specYRange = opt.spec_ylim;
    jProp{i}.specYTickDelta = opt.spec_ytickdelta;
    jProp{i}.specYUnit = '[dB]';
    jProp{i}.specXUnit = '[Hz]';
    %Timecourse Object
    jProp{i}.timeName = 'TimeCourse';
    jProp{i}.timeXTicks = time_tick_ind;
    jProp{i}.timeXLabel = time_tick_lab;
    jProp{i}.timeYRange = opt.time_ylim;
    jProp{i}.timeYTickDelta = opt.time_ytickdelta;
    jProp{i}.timeYUnit = '[muV]';
    jProp{i}.timeXUnit = '[s]';
    
    jBCI{i}.setLocation(java.awt.Point(opt.fig_position{i}(1),opt.fig_position{i}(2)));
    jBCI{i}.setBCIPlotProperties(jProp{i});
    jBCI{i}.initAndStart;
  end
  dat_chans{i} = length([dat_chans{:}])+1:length([dat_chans{:}])+length([chans{i}]);
 end

%%%%% EEG data acquisition %%%%%%%%

% initialize a data buffer for EEG
cnt= struct('fs', opt.target_fs);
cnt.clab= {state.clab{[chans{:}]}};
bN= ceil(opt.buffer_length/1000*cnt.fs);
cnt_buff(1,bN,playernum,spec_num,opt.spec_length/1000*cnt.fs);
% initialize a data buffer for the spectra:
spec_buff(1,opt.integrate, playernum, spec_num, length(spec_ind));

online_variance(length(cnt.clab),opt.var_length(2)/1000*cnt.fs,opt.var_length);
[b_var,a_var]=getButterFixedOrder(opt.var_band,opt.target_fs,5);
state_var = [];
[b_spec,a_spec]=getButterFixedOrder(opt.spec_band,opt.target_fs,5);
state_spec = {[],[]};
run = 1;

%%%%% Figures arrangement loop %%%%%
while run
    visualize = 1;
    switch curr_playernum
        case 1
            % first player only
            disp('Setting mode 1');
            if opt.matlab_figure
                % arrange player 1
                set(h_fig{1},'Visible','on',...
                    'Position',...
                    [opt.fig_position{1}(1), ...
                        h_par(4)-opt.fig_position{1}(2)-opt.fig_size(2),...
                        opt.fig_size(1) opt.fig_size(2)-opt.fig_size(1)]);
                figure(h_fig{1});drawnow
                % hide player 2
                set(h_fig{2},'Visible','off');
            end
            if opt.java_figure==1
                % arrange player 1
                javaPlot{1}.setVisible(1);
                javaPlot{1}.toFront;
                javaPlot{1}.setSize(java.awt.Dimension(opt.fig_size(1)+...                    opt.fig_javacorrection(1),opt.fig_size(1)));
                javaPlot{1}.setLocation(opt.fig_position{1}(1)-...
                    opt.fig_javacorrection(2),opt.fig_position{1}(2));
                % hide player 2
                javaPlot{2}.setVisible(0);
            end
        case 2
            % second player only
            disp('Setting mode 2');
            if opt.matlab_figure
                % arrange player 2
                set(h_fig{2},'Visible','on',...
                    'Position',...
                    [opt.fig_position{1}(1), ...
                        h_par(4)-opt.fig_position{1}(2)-opt.fig_size(2),...
                        opt.fig_size(1) opt.fig_size(2)-opt.fig_size(1)]);
                figure(h_fig{2});drawnow
                % hide player 1
                set(h_fig{1},'Visible','off');
            end
            if opt.java_figure==1
                % arrange player 2
                javaPlot{2}.setVisible(1);
                javaPlot{2}.toFront;
                javaPlot{2}.setSize(java.awt.Dimension(opt.fig_size(1)+...
                    opt.fig_javacorrection(1),opt.fig_size(1)));
                javaPlot{2}.setLocation(opt.fig_position{1}(1)-...                    opt.fig_javacorrection(2),opt.fig_position{1}(2));
                % hide player 1
                javaPlot{1}.setVisible(0);                
            end
        case 3
            % two players simultaneously
            disp('Setting mode 3');
            if opt.matlab_figure
                % arrange player 1
                set(h_fig{1},'Visible','on',...
                    'Position',...
                    [opt.fig_position{2}(1), ...
                        h_par(4)-opt.fig_position{2}(2)-opt.fig_size(2),...
                        opt.fig_size(1) opt.fig_size(2)-opt.fig_size(1)]);
                figure(h_fig{1});drawnow
                % arrange player 2
                set(h_fig{2},'Visible','on',...                    'Position',...
                    [opt.fig_position{1}(1), ...                        h_par(4)-opt.fig_position{1}(2)-opt.fig_size(2),...
                        opt.fig_size(1) opt.fig_size(2)-opt.fig_size(1)]);
                figure(h_fig{2});drawnow
            end
            if opt.java_figure==1
                % arrange player 1
                javaPlot{1}.setVisible(1);
                javaPlot{1}.toFront;
                javaPlot{1}.setSize(java.awt.Dimension(opt.fig_size(1)+...
                    opt.fig_javacorrection(1),opt.fig_size(1)));
                javaPlot{1}.setLocation(opt.fig_position{2}(1)-...
                    opt.fig_javacorrection(2),opt.fig_position{2}(2));
                % arrange player 2
                javaPlot{2}.setVisible(1);
                javaPlot{2}.toFront;
                javaPlot{2}.setSize(java.awt.Dimension(opt.fig_size(1)+...                    opt.fig_javacorrection(1),opt.fig_size(1)));
                javaPlot{2}.setLocation(opt.fig_position{1}(1)-...                    opt.fig_javacorrection(2),opt.fig_position{1}(2));
            end
        otherwise
            % something's funny.
            warning(sprintf('Unknown player number %i',curr_playernum));
    end
    
    %%%%% Visualization loop %%%%%
    while visualize
        [cnt.x]= acquire_bv(state);
        cnt.x = cnt.x(:,[chans{:}]);
        % extract the variance estimates:
        [bl,state_var] = online_filt(cnt,state_var,b_var,a_var);
        vari = online_variance(bl,bN,opt.var_length);
        data = log(vari.x(1,:)./vari.x(2,:));
        
        % filter data spatially:
        for ii = 1:playernum
            %write directly into the buffer:
            cnt_csp = proc_linearDerivation(proc_selectChannels(cnt,dat_chans{ii}),csp_projection{ii});
            [cnt_csp,state_spec{ii}] = online_filt(cnt_csp,state_spec{ii},b_spec,a_spec);
            cnt_buff(2,cnt_csp.x,ii);
        end
        % Retrieve the spatially filtered data with the sufficient windowlength:
        cnt_spatiotemp_flt = cnt_buff(3);
        % extract the spectral estimates:
        for jj = 1:playernum
            for kk = 1:spec_num
                [Pxx{jj,kk},f] = pyulear(cnt_spatiotemp_flt(:,(jj-1)*spec_num + kk),...
                    opt.ar_order,opt.fft_length,cnt.fs);
                %[Pxx{jj,kk},fs]= pwelch(cnt_spatiotemp_flt(:,(jj-1)*spec_num + kk),...
                %[],[],opt.fft_length,100);
                % enter the estimates into the buffer:
                spec_buff(2,Pxx{jj,kk}(spec_ind),jj,kk);
            end
        end
        
        % the important part of the spectrum: Pxx(spec_ind,:).
        for ii=1:playernum
            if opt.java_figure==1
                scalpreturn{ii} = javaPlot{ii}.setData(data(dat_chans{ii}));
                run = run&scalpreturn{ii}.isAlive;
            elseif opt.java_figure==2
                jTrans{ii}.ScalpData = data(dat_chans{ii});
                jTrans{ii}.SpecData = max(opt.spec_ylim(1),min(opt.spec_ylim(2),10*log10(spec_buff(3,jj,kk)+eps)));
                jTrans{ii}.TimeData = max(opt.time_ylim(1),min(opt.time_ylim(2),...
                    cnt_spatiotemp_flt(:,spec_num*(jj-1)+kk)));
                
                keyboard
                jTrans{ii} = jBCI{ii}.setData(jTrans{ii});
                keyboard
                run = run&jTrans{ii}.isAlive;
            end
        end 
        
        if opt.matlab_figure
            for jj = 1:playernum
                for kk = 1:spec_num
                    % draw a spectrum
                    set(p{jj,kk,1},'YData',10*log10(spec_buff(3,jj,kk))+eps);
                    % draw the timecourse
                    set(p{jj,kk,2},'YData',cnt_spatiotemp_flt(:,spec_num*(jj-1)+kk));
                end
            end
            drawnow
        end
        % Look for new player arrangements:
        sig = get_data_udp(com_hand,0,1);
        if ~isempty(sig)
            if sig~=curr_playernum
                % setup has changed.
                curr_playernum = sig;
                visualize =0;
            end
        end
        visualize = visualize*run;
        waitForSync(1000/opt.fps);
    end %%%% Visualization loop %%%%%
end %%%% Figures arrangement loop %%%%%
acquire_bv('close');
% clear the buffers
cnt_buff(4);
spec_buff(4);
if opt.java_figure==1
  for ii =1:playernum
    javaPlot{ii}.setVisible(false);
    javaPlot{ii}.dispose;
  end
  clear javaPlot
end
if opt.matlab_figure
  close all
end
return




function data = cnt_buff(action,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stores and retrieves spatially projected data in several CSP channels per player.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent poi bN buffer hist_length spec_num playernum
switch action
  case 1
    % init case
    poi = 1;
    bN = varargin{1};
    playernum = varargin{2};
    spec_num = varargin{3};
    hist_length = varargin{4};
    buffer = zeros(bN,spec_num*playernum);
  case 2
    % fill the buffer with data.
    data = varargin{1};
    s = size(data,1);
    player = varargin{2};
    if s>bN
      warning('Too much data for buffer');
    end
    fill_ind = mod((poi:(poi+s-1))-1,bN)+1;
    buffer(fill_ind,spec_num*(player-1)+(1:spec_num)) = data;
    if player==playernum
      % we expect that first player 1, then player 2 is updated.
      % if the second player is following, the pointer is not moved.
      poi = mod((poi+s-1),bN)+1;
    end
  case 3
    % query some values.
    query_ind = mod(((poi-hist_length):(poi-1))-1,bN)+1;
    data = buffer(query_ind,:);
  case 4
    % cleanup
    clear poi bN buffer hist_length spec_num
end
return



function data = spec_buff(action,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% stores and retrieves spectra of several CSP channels per player.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
persistent poi bN buffer integrate spec_size spec_num playernum
switch action
  case 1
    % init case
    poi = 1;
    integrate = varargin{1};
    playernum = varargin{2};
    spec_num = varargin{3};
    spec_size = varargin{4};
    bN = integrate;
    buffer = zeros(bN,spec_num*spec_size*playernum);
  case 2
    % fill the buffer with data.
    data = varargin{1};
    player = varargin{2};
    chan_num = varargin{3};
    fill_ind = mod(poi-1,bN)+1;
    buffer(fill_ind,spec_size*spec_num*(player-1)+(1:spec_size)+ spec_size*(chan_num-1)) = data';
    if player==playernum
      % we expect that first player 1, then player 2 is updated.
      % if the second player is following, the pointer is not moved.
      poi = mod(poi,bN)+1;
    end
  case 3
    % query some values.
    %query_ind = mod(((poi-integrate):(poi-1))-1,bN)+1;
    player = varargin{1};
    chan_num = varargin{2};
    data = mean(buffer(:,spec_size*spec_num*(player-1)+(1:spec_size)+ spec_size*(chan_num-1)),1);
  case 4
    % cleanup
    clear poi bN buffer integrate spec_size playernum
end
return
