function Data= showGuiEEG(data, mrk, chans)

% showGuiEEG visualizes continuous or epoched data, 
% Features: setting the time interval, channel selection, epoch removal,
% etc.
% 
% IN    data    - data structure of continuous or epoched data
%       mrk     - marker structure
%       chans   - channels which should be visualized (optional) 
%
% OUT   data    - updated data structure
%
% SEE showEEG 

% Christine Carl 23/10/06, extended by Simon Scholler 06/05/08
% Please report bugs to simon.scholler(at)gmail.com

persistent GUI DATA STATUS

if nargin>3
    error(['Too many input arguments (3 required, ' int2str(nargin) ' given).']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INTIALIZE THE GUI
elseif nargin<=3 && isstruct(data)  
    
    if length(data.clab) ~= size(data.x,2)
        error(['Number of channels in the data matrix is not consistent ' ...
               'with the number given in the .clab structure'])
    end
    
    % close the GuiEEG if it was already open
    try
       close(GUI.FIG);
    end
    
    % init variables
    DATA = data;   
    STATUS.EPOCH= 1;
    
    if exist('mrk', 'var')
        STATUS.MRK = mrk;
    else
        STATUS.MRK=[];
    end
    
    if exist('chans','var')
        chans= data.clab(chanind(data, chans));  % resolve abbrev like 'C3,z,4'
        STATUS.clab= chans;
        STATUS.initClab= chans;
    else
        STATUS.clab= DATA.clab;
        STATUS.initClab= DATA.clab;
    end
    
    % open a new figure
    GUI.FIG = figure;
    sz = get(0,'ScreenSize');     % get screen resolution
    set(GUI.FIG,'Position',[0,0,sz(3),sz(4)],'NumberTitle','off','Name','EEGDATA');
    %set(GUI.FIG,'Units', 'Normalized', 'Position',[0.05,0,0.9,0.85],'NumberTitle','off','Name','EEGDATA');
    axes( 'Position',[0.1, 0.1, 0.65 ,0.85],...
        'Parent',GUI.FIG);

    % check if data is epoched or continuous & adjust settings accordingly
    if ndims(DATA.x)==2
        % continuous data specific settings
        STATUS.CNT= 1;
        if size(DATA.x, 1)>=5000
            STATUS.IVAL= [0 5000];        % default time interval shown
        else
            STATUS.IVAL= [0 size(DATA.x,1)];
        end
        stepsize = STATUS.IVAL(1,2) -STATUS.IVAL(1,1);      % default stepsize
    elseif ndims(DATA.x)==3
        % epoch data specific settings
        STATUS.CNT= 0;
        STATUS.nEPOCHS= size(DATA.x,3);
        STATUS.REMOVE_EPOCHS= zeros(1, size(DATA.x, 3));
        STATUS.IVAL= [DATA.t(1) DATA.t(end)];
    else
        error('Invalid data format. Expected continuous or epoched data')
    end
    
    STATUS.scaling= getScalingFactor();   % determine scaling factor for visualization
    setMuVScale();                        % get the maximal value for the muV-bar
    updateShownEEG(DATA, STATUS);         % & show EEG
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CREATE UIPANELS
    y= 1/38; x= 0.8; w= 0.17;
    if ~STATUS.CNT
        GUI.REMOVE_EPOCHS_PANEL= uipanel('Title', 'Remove Epochs', 'FontWeight', 'bold', 'Position', [x y*2 w y*6]);
        time_sel_panel_title= 'Epoch Selection';
    else
        time_sel_panel_title= 'Time Interval Selection';
    end
        
    GUI.SELECT_CHANNELS_PANEL= uipanel('Title', 'Channel Selection', 'FontWeight', 'bold', 'Position', [x y*9 w y*4]);
    GUI.SCALE_VOLTAGE_PANEL= uipanel('Title', 'Scale Voltage', 'FontWeight', 'bold', 'Position', [x y*14 w y*4]);
    GUI.TIME_SELECTION_PANEL= uipanel('Title', time_sel_panel_title, 'FontWeight', 'bold', 'Position', [x y*19 w y*8]);
    GUI.INTERVAL_SELECTION_PANEL= uipanel('Title', 'Interval Selection', 'FontWeight', 'bold', 'Position', [x y*28 w y*8]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% CREATE UICONTROLS
    x=2/15;
    x4= 1/30;
    h1= 0.55;            % height of component if there is only component in one column
    y1= (1-h1)/3;
    h2= 0.3;            % height of component if there are two components in one column
    y2= (1-2*h2)/6;  
    w1= 1-2*x;          % width of component if there is only one component in one row
    w2= (1-3*x)/2;      % width of component if there are two components in one row
    w4= (1-5*x4)/4;       % width of component if there are four components in one row
    f= 8;               % fontsize normal
    F= 9;               % fontsize large
    
    % GUI components for choosing a specific interval
    GUI.INTERVAL= uicontrol('Parent', GUI.INTERVAL_SELECTION_PANEL, ...
                            'Style','Edit','Units','normalized','Position',[x y2*3+h2 w1 h2],...
                            'FontSize',f,'Tooltipstring','Type in the interval you want to plot',...
                            'String', int2str(STATUS.IVAL));
    GUI.UPDATE= uicontrol('Parent', GUI.INTERVAL_SELECTION_PANEL, ...
                          'Style','pushbutton','Units','normalized','Position',[x y2*2 w1 h2],...
                          'FontSize',f,'Tooltipstring','Update the plot','String','Update', ...
                          'Callback','showGuiEEG(1)');                 
    % GUI components for scaling (y-axis)
    GUI.SCALE_LARGER= uicontrol('Parent', GUI.SCALE_VOLTAGE_PANEL, ...
                                'Style','pushbutton','Units','normalized','Position',[x*2+w2 y1 w2 h1],...
                                'FontSize',F,'String','+', 'FontSize',14,'Callback', 'showGuiEEG(12)');
    GUI.SCALE_SMALLER= uicontrol('Parent', GUI.SCALE_VOLTAGE_PANEL, ...
                                 'Style','pushbutton','Units','normalized','Position',[x y1 w2 h1],...
                                 'FontSize',F,'String','-', 'FontSize',14,'Callback', 'showGuiEEG(13)');
    % GUI components for removing or hiding channels
    GUI.CHANNEL_SELECTION= uicontrol('Parent', GUI.SELECT_CHANNELS_PANEL, ...
                                     'Style','pushbutton','Units','normalized','Position',[x y1 w1 h1],...
                                     'FontSize',f,'String','Channel Selection','Callback', 'showGuiEEG(5)');
  
    % Special GUI components for continuous and epoched data
    if STATUS.CNT
        % GUI components for scaling (x-axis)
        GUI.STEPSIZE= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                                'Style','Edit','Units','normalized','Position',[x y2*3+h2*0.75 w1 h2*0.75],...
                                'FontSize',f,'Tooltipstring','Type in the length (in ms) of the data interval you want to visualize.','String', stepsize);
        GUI.DUR_LABEL= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                                 'Style','Text','Units','normalized','Position',[x y2*3+h2*1.5 w1 h2*0.5],...
                                 'FontSize',f,'String',' Stepsize [ms]');
                             
    % GUI components for going through the selected time interval
    GUI.LEFT= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                        'Style','pushbutton','Units','normalized','Position',[x4 y2 w4 h2*0.9],...
                        'FontSize',F,'Tooltipstring','Show previous interval','String','-1',...
                        'Callback', 'showGuiEEG(3)');        
    GUI.LEFT_HALF= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                        'Style','pushbutton','Units','normalized','Position',[x4*2+w4 y2 w4 h2*0.9],...
                        'FontSize',F,'Tooltipstring','Go 1/2 interval back','String','-1/2',...
                        'Callback', 'showGuiEEG(31)'); 
    GUI.RIGHT_HALF= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                        'Style','pushbutton','Units','normalized','Position',[x4*3+w4*2 y2 w4 h2*0.9],...
                        'FontSize',F,'Tooltipstring','Go 1/2 interval further','String','+1/2',...
                        'Callback', 'showGuiEEG(21)');
    GUI.RIGHT= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                         'Style','pushbutton','Units','normalized','Position',[x4*4+w4*3 y2 w4 h2*0.9],...
                         'FontSize',F,'Tooltipstring','Show following interval','String','+1',...
                         'Callback', 'showGuiEEG(2)');
    

                      
    else
        % GUI component for removing (bad) epochs
        GUI.TICK_EPOCH= uicontrol('Parent' , GUI.REMOVE_EPOCHS_PANEL, ...
                                  'Style', 'checkbox', 'Units', 'normalized', 'Position', [x y2*3+h2*1.5 w1 h2*0.5],...
                                  'FontSize',f, 'String', 'Remove epoch', 'Callback', 'showGuiEEG(10)','Min', 0, 'Max', 1);
        GUI.REMOVE_EPOCHS= uicontrol('Parent' , GUI.REMOVE_EPOCHS_PANEL, ...
                                     'Style','pushbutton','Units','normalized','Position',[x y2*2 w1 h2*1.5],...
                                     'FontSize',f,'String', 'Remove epochs','Callback', 'showGuiEEG(11)',...
                                     'Tooltipstring', 'Removes all ticked epochs from the dataset');
                                 
        % GUI component for choosing a epoch
        GUI.EPOCH_SLIDER= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                                    'Style','Slider','Units','normalized','Position',[x y2*3+h2*0.75 w1 h2*0.75],...
                                    'FontSize',f,'Tooltipstring','Type in the epoch number you want to visualize', ...
                                    'Min', 1, 'Max', size(DATA.x, 3), 'Callback', 'showGuiEEG(4)', 'Value', 1);
        GUI.EPOCH_LABEL= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                                   'Style','Text','Units','normalized','Position',[x y2*3+h2*1.5 w1 h2*0.5],...
                                   'FontSize',f,'String',['Epoch ' int2str(STATUS.EPOCH)]);

        % GUI components for going through the selected time interval
        GUI.RIGHT= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                             'Style','pushbutton','Units','normalized','Position',[x*2+w2 y2*2 w2 h2*0.75],...
                             'FontSize',F,'Tooltipstring','Show following epoch','String','-->',...
                             'FontWeight', 'bold', 'Callback', 'showGuiEEG(2)');
        GUI.LEFT= uicontrol('Parent', GUI.TIME_SELECTION_PANEL,...
                            'Style','pushbutton','Units','normalized','Position',[x y2*2 w2 h2*0.75],...
                            'FontSize',F,'Tooltipstring','Show previous epoch','String','<--',...
                            'FontWeight', 'bold', 'Callback', 'showGuiEEG(3)');                                
    end
    waitfor(GUI.FIG);
    Data= DATA;
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RESPOND TO USER INPUT (this part is only called internally via callbacks)
else                        
    usercommand= data;   
    switch usercommand;        
        case 1
            updateInterval() % choose an interval within the loaded data
        case 2
            rightShift(1)     % if data is epoched, step one epoch further
                              % if data is continuous, step one interval further
        case 21
            rightShift(0.5)     % only for continuous data: step 1/2 interval further 
        case 3
            leftShift(1)      % if data is epoched, step one epoch back
                              % if data is continuous, step one interval
                              % back    
        case 31
            leftShift(0.5)      % only for continuous data: step 1/2 interval back   
        case 4
            sliderEpochChange()
        case 5            
            createChannelSelectionListbox()
        case 6
            removeChannels()      % remove channels (remove from DATA.clab and STATUS.clab)
        case 7
            hideChannels()        % hide channels (remove only from STATUS.clab but not from DATA.clab) 
        case 8
            showAllInitChannels() % shows all initially visible channels 
        case 9
            showAllChannels()
        case 10
            tickEpoch()
        case 11
            removeTickedEpochs()
        case 12
            increaseVoltScaling()     % scale y-axis of data (muV)    
        case 13
            decreaseVoltScaling()     % scale y-axis of data (muV)
    end
end


    % case 1
    function updateInterval()   
            ival= round(str2num(get(GUI.INTERVAL, 'String')));
            if ~STATUS.CNT
                if ival(2)<=ival(1)||ival(1)<DATA.t(1)||ival(2)>DATA.t(end)||size(ival,2)~=2
                    err_win= error_handling('The specified interval is not in the range of the recorded data. Press ''OK'' to continue');
                    ival= STATUS.IVAL;
                    %set(GUI.FIG, 'Visible', 'off');
                    waitfor(err_win);
                    figure(GUI.FIG);
                    %set(gcf, 'Visible', 'on');
                end
            else
                if ival(2)<=ival(1)||ival(1)<0||ival(2)>(size(DATA.x,1)*1000/DATA.fs-(1000/DATA.fs))||size(ival,2)~=2
                    err_win = error_handling('The specified interval is not in the range of the recorded data. Press ''OK'' to continue');
                    ival = STATUS.IVAL;
                    %set(GUI.FIG, 'Visible', 'off');
                    waitfor(err_win);
                    figure(GUI.FIG);
                    %set(gcf, 'Visible', 'on');
                end
            end
            STATUS.IVAL= ival;
            updateShownEEG(DATA, STATUS);
            set(GUI.INTERVAL, 'String',num2str(ival));
    end

    % case 2
    function rightShift(factor)
        if STATUS.CNT
            stepsize = round(str2num(get(GUI.STEPSIZE, 'String')));
            ival = round(str2num(get(GUI.INTERVAL,'String')));
            if factor~=1
                ival= ival-factor*stepsize;
            end
            if (ival(2)+stepsize)>(size(DATA.x,1)*1000/DATA.fs)
                ival = [ival(2) (size(DATA.x,1)*1000/DATA.fs-1000/DATA.fs)];
                err_win = error_handling('End of data reached. Press ''OK'' to continue.');
                waitfor(err_win);
                figure(GUI.FIG);
            else
                ival = [ival(2) ival(2)+stepsize];
            end
            STATUS.IVAL= ival;
            updateShownEEG(DATA, STATUS);
            set(GUI.INTERVAL, 'String',num2str(ival));
        else
            if STATUS.EPOCH==size(DATA.x,3)
                %err_win = error_handling('Last epoch reached. Press ''OK'' to continue');
                %waitfor(err_win);
                %figure(GUI.FIG);
                STATUS.EPOCH= STATUS.EPOCH-1;
            end
            STATUS.EPOCH= STATUS.EPOCH+1;
            updateShownEEG(DATA, STATUS);
            set(GUI.EPOCH_SLIDER, 'Value', STATUS.EPOCH');
            set(GUI.EPOCH_LABEL, 'String', ['Epoch ' int2str(STATUS.EPOCH)]);
            set(GUI.TICK_EPOCH, 'Value', STATUS.REMOVE_EPOCHS(1, STATUS.EPOCH));
        end
    end

    % case 3
    function leftShift(factor)
        if STATUS.CNT
            stepsize = round(str2num(get(GUI.STEPSIZE, 'String')));
            ival = round(str2num(get(GUI.INTERVAL,'String')));
            if factor~=1
                ival= ival+factor*stepsize;
            end
            if ival(1)~=0
                ival = [ max(0,ival(1)-stepsize) ival(1)];
            else
                ival= [0 stepsize];
            end
            STATUS.IVAL= ival;
            updateShownEEG(DATA, STATUS);
            set(GUI.INTERVAL, 'String',num2str(ival));
        else
            if STATUS.EPOCH==1
                %err_win = error_handling('First epoch reached. Press ''OK'' to continue');
                %waitfor(err_win);
                %figure(GUI.FIG);
                STATUS.EPOCH= STATUS.EPOCH+1;
            end
            STATUS.EPOCH= STATUS.EPOCH-1;
            updateShownEEG(DATA, STATUS);
            set(GUI.EPOCH_SLIDER, 'Value', STATUS.EPOCH');
            set(GUI.EPOCH_LABEL, 'String', ['Epoch ' int2str(STATUS.EPOCH)]);
            set(GUI.TICK_EPOCH, 'Value', STATUS.REMOVE_EPOCHS(1, STATUS.EPOCH));
        end
    end

    % case 4
    function sliderEpochChange()
        STATUS.EPOCH= ceil(get(GUI.EPOCH_SLIDER, 'Value'));
        set(GUI.EPOCH_LABEL, 'String', ['Epoch ' int2str(STATUS.EPOCH)]);
        updateShownEEG(DATA, STATUS);
        set(GUI.TICK_EPOCH, 'Value', STATUS.REMOVE_EPOCHS(1, STATUS.EPOCH));
    end

    % case 5
    function createChannelSelectionListbox()
        % create listbox with a listing of all channels
        GUI.CHANNEL_SELECTION= figure;
        w= 1/23*4; % button width
        h= 0.08; % button height
        d= (1-4*w)/7; % button distance
        y= 0.01; % vertical position of buttons
        set(gcf, 'NumberTitle','off','Name','Channel Selection', 'Units', 'normalized', 'OuterPosition', [0.3 0.2 0.4 0.6])
        GUI.CHAN_SELECT_LISTBOX= uicontrol('Style', 'listbox', 'Units', 'normalized', 'Position', [0 0.1 1 0.9], 'String', STATUS.clab, 'max', length(STATUS.clab)-1);
        % add 'remove' button
        uicontrol('Style', 'pushbutton', 'String', 'Remove', 'Units', 'normalized', 'Position', [d*2 y w h], 'Callback', 'showGuiEEG(6)', ...
                  'TooltipString', 'Selected channels are deleted from the dataset.');
        % add 'hide channels' button
        uicontrol('Style', 'pushbutton', 'String', 'Hide', 'Units', 'normalized', 'Position', [d*3+w y w h], 'Callback', 'showGuiEEG(7)', ...
                  'TooltipString', 'Selected channels are hidden from view but not deleted.');
        % show 'all channels' button
        uicontrol('Style', 'pushbutton', 'String', 'Init Chans', 'Units', 'normalized', 'Position', [d*4+2*w y w h], 'Callback', 'showGuiEEG(8)', ...
                  'TooltipString', 'Visualizes all channels shown initially but not the removed ones.');
        % show 'all channels' button
        uicontrol('Style', 'pushbutton', 'String', 'Show all', 'Units', 'normalized', 'Position', [d*5+3*w y w h], 'Callback', 'showGuiEEG(9)', ...
                  'TooltipString', 'Shows all channels in the dataset.');
    end

    % case 6
    function removeChannels()
        channels = STATUS.clab(get(GUI.CHAN_SELECT_LISTBOX, 'Value'));
        % make sure that at least one channel was selected
        if ~isempty(channels)
            DATA = proc_selectChannels(DATA, 'not', channels);
            [sd, idx]= setdiff(STATUS.clab, channels);
            STATUS.clab= STATUS.clab(sort(idx));
            [sd_i, idx_i]= setdiff(STATUS.initClab, channels);
            STATUS.initClab= STATUS.initClab(sort(idx_i));
            setMuVScale();
        end
        close(gcf);
        STATUS.scaling= getScalingFactor();
        updateShownEEG(DATA, STATUS);
    end

    % case 7
    function hideChannels()
        channels = STATUS.clab(get(GUI.CHAN_SELECT_LISTBOX, 'Value'));
        % check if no channel was selected
        if ~isempty(channels)
            % remove selected channels from STATUS.clab
            [sd, idx]= setdiff(STATUS.clab, channels);
            STATUS.clab= STATUS.clab(sort(idx));
            setMuVScale();
        end
        close(gcf);
        STATUS.scaling= getScalingFactor();
        updateShownEEG(DATA, STATUS);
    end

    % case 8
    function showAllInitChannels()
        if ~isempty(STATUS.initClab)
            STATUS.clab= STATUS.initClab;
            close(GUI.CHANNEL_SELECTION);
            STATUS.scaling= getScalingFactor();
            setMuVScale();
            updateShownEEG(DATA, STATUS);
        else
            error_handling('All initial channels have already been removed.')
        end
    end

    % case 9
    function showAllChannels()        
        STATUS.clab= DATA.clab;
        close(gcf);
        STATUS.scaling= getScalingFactor();
        setMuVScale();
        updateShownEEG(DATA, STATUS);
    end

    % case 10
    function tickEpoch()
        if (get(GUI.TICK_EPOCH,'Value')==1)
            % checkbox is checked
            STATUS.REMOVE_EPOCHS(1, STATUS.EPOCH)= 1;
        else
            % checkbox is not checked
            STATUS.REMOVE_EPOCHS(1, STATUS.EPOCH)= 0;
        end
    end

    % case 11
    function removeTickedEpochs()
        idx= find(STATUS.REMOVE_EPOCHS);   % get indices of the epochs
        STATUS.EPOCH= STATUS.EPOCH-sum(idx<=STATUS.EPOCH);
        if STATUS.EPOCH==0
            STATUS.EPOCH= 1;
        end
        DATA= proc_removeEpochs(DATA, idx);  % remove ticked epochs
        
        % change the GUI appropriately
        setMuVScale();
        STATUS.REMOVE_EPOCHS= zeros(1, size(DATA.x, 3));
        set(GUI.TICK_EPOCH, 'Value', 0);
        set(GUI.EPOCH_SLIDER, 'Value', STATUS.EPOCH, 'Max', size(DATA.x, 3));
        set(GUI.EPOCH_LABEL, 'String', ['Epoch ' int2str(STATUS.EPOCH)]);
        updateShownEEG(DATA, STATUS);
    end

    % case 12
    function increaseVoltScaling()
        STATUS.scaleFactor= STATUS.scaleFactor*sqrt(2);
        updateShownEEG(DATA, STATUS);
    end

    % case 13
    function decreaseVoltScaling()
        STATUS.scaleFactor= STATUS.scaleFactor/sqrt(2);
        updateShownEEG(DATA, STATUS);
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% UTILITY FUNCTIONS:
    function scale= getScalingFactor()
        dat= proc_selectChannels(DATA, STATUS.clab);
        resetScaleFactor();
        if STATUS.CNT
            scale= 1./nanstd(dat.x(1:10:end,:));
            scale(find(isinf(scale)))= 1;   %% exclude 'constant' channels from scaling
            scale= median(scale)/10;
        else
            scale= 1./mean(squeeze(nanstd(dat.x(:,:,1:10:end))),2);
            scale(find(isinf(scale)))= 1;   %% exclude 'constant' channels from scaling
            scale= median(scale)/10;
        end
    end 

    function resetScaleFactor()
        STATUS.scaleFactor= 4;
    end

    function setMuVScale()
        dat= proc_selectChannels(DATA, STATUS.clab);
        if STATUS.CNT
            STATUS.muVScale= ceil(mean(nanstd(dat.x(1:10:end,:)))/10)*10;
        else
            STATUS.muVScale= ceil(squeeze(mean(mean(nanstd(dat.x(:,:,1:10:end)))))/10)*10;
        end
    end

end


%% popup window for error messages
function h = error_handling(error_msg)
h= figure;
uicontrol('Style','Text','Units','normalized','Position',[0.25 0.5,0.5,0.2],'FontSize',14,'String','Backward', 'String', error_msg);
uicontrol('Style','pushbutton','Units','normalized','Position',[0.8 0.1,0.08,0.1],'FontSize',14,'String','Ok','Callback','close(gcf)');
end


%% show EEG data
function updateShownEEG(data, STATUS)
data= proc_selectChannels(data, STATUS.clab);
ival= STATUS.IVAL;
mrk= STATUS.MRK;
if ndims(data.x)==2
    showEEG(data, ival, mrk, STATUS);
elseif ndims(data.x)==3
    data.x= squeeze(data.x(:,:,STATUS.EPOCH));
    showEEG(data, ival, mrk, STATUS, 'baseline',0);
else
    error('Invalid data format. Expected continuous or epoched data')
end
end
