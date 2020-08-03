
function scrollPlots(userinput)
    % Call this function to scroll through the plots which have been
    % previously added by sp_addPlots
    %
    % LOOKUP test_scrollPlots for a usage example
    %
    % 'userinput'  is an internal callback variable to respond to the 
    % users actions such as button presses, but it can also be used to 
    % delete all previously stored figures by calling: scrollPlots('close')

    global SP    
    global GUI
    
    if nargin==0
        % open a new figure
        GUI.FIG = figure(100);
        if ~isfield(SP,'AX') || isempty(SP.AX)
            fig = figure;
            GUI.AXES = gca;
            sp_addPlots(fig)
            GUI.currAX = 1;
        else
            if ~isfield(GUI, 'currAX')
                GUI.currAX = 1;
            end
            GUI.AXES = SP.AX{GUI.currAX};
        end

        sz = get(0,'ScreenSize');     % get screen resolution
        set(GUI.FIG, ...%'Position',[0,0,sz(3),sz(4)], 
                    'Color', [0.5 0.5 0.7], ...
                    'NumberTitle','off', ...
                    'Name', ['ScrollPlots (' int2str(GUI.currAX) '/' int2str(length(SP.AX)) ')'], ...
                    'DeleteFcn', 'scrollPlots(''close'')');
                
        GUI.AXISPOS = [0.1, 0.1, 0.7 ,0.8];
        set(GUI.AXES, 'Position', GUI.AXISPOS, 'Parent', GUI.FIG);
        addLegend()
        addColorbar()

        w = 0.15;
        h = 0.25;
        x = 0.825;
        % init panels
        GUI.SCROLL_PANEL= uipanel('Parent', GUI.FIG, 'Title', 'Scroll', 'FontWeight', 'bold', ...
                                  'Position', [x 0.1 w h*2], 'BackgroundColor', [0.5 0.5 0.7]);
        GUI.DELETE_PANEL= uipanel('Parent', GUI.FIG, 'FontWeight', 'bold', 'Position', [x 0.65 w h]);

        % init buttons
        % GUI components for going through the selected time interval
        F = 9;
        GUI.LEFT= uicontrol('Parent', GUI.SCROLL_PANEL,...
            'Style','pushbutton','Units','normalized','Position',[0.1 0.53 0.6 0.4],...
            'FontSize',20,'Tooltipstring','Show previous plot','String','<',...
            'FontWeight', 'bold', 'Callback', 'scrollPlots(''left'')');
        GUI.RIGHT= uicontrol('Parent', GUI.SCROLL_PANEL,...
            'Style','pushbutton','Units','normalized','Position',[0.1 0.07 0.6 0.4],...
            'FontSize',20,'Tooltipstring','Show next plot','String','>',...
            'FontWeight', 'bold', 'Callback', 'scrollPlots(''right'')');
        GUI.FIG_SLIDER= uicontrol('Parent', GUI.SCROLL_PANEL,...
            'Style','Slider','Units','normalized','Position',[0.75 0.07 0.2 0.86],...
            'Min', 1, 'Max', max(1,length(SP.AX)), 'Callback', 'scrollPlots(''slide'')',... 
            'SliderStep', [1 5], 'Value', 1);
        GUI.DEL= uicontrol('Parent', GUI.DELETE_PANEL,...
            'Style','pushbutton','Units','normalized','Position', [0 0 1 1], ... %[0.2 0.6 0.6 0.4],...
            'FontSize',9,'String','Delete',...
            'FontWeight', 'bold', 'Callback', 'scrollPlots(''del'')');
        %waitfor(GUI.FIG)

    else
       if strcmpi(userinput, 'close')
           try
            close(SP.FIG)
           catch
           end
           SP.FIG = [];
           if ~isempty(GUI)
              close(GUI.FIG)
              GUI = [];
           end
           SP.AX = [];
           SP.LEG = [];
           SP.CB = [];
       else
           colorbar off
           legend('off')
           set(GUI.FIG_SLIDER, 'Max', length(SP.AX))
           set(GUI.AXES, 'Parent', SP.FIG);
           if strcmpi(userinput, 'left')
               GUI.currAX = max(1,GUI.currAX-1);
               set(GUI.FIG_SLIDER,'Value',GUI.currAX)
           elseif strcmpi(userinput, 'right')
               GUI.currAX = min(length(SP.AX),GUI.currAX+1);
               set(GUI.FIG_SLIDER,'Value',GUI.currAX)
           elseif strcmpi(userinput, 'slide')
               currAX = round(get(GUI.FIG_SLIDER,'Value'));
               GUI.currAX = max(1,min(length(SP.AX),currAX));
           elseif strcmpi(userinput, 'del')
               if GUI.currAX ~= 1
                   set(GUI.FIG_SLIDER, 'Value', get(GUI.FIG_SLIDER,'Value')-1)
               end
               if length(SP.AX)>1
                   SP.AX(GUI.currAX) = [];
                   SP.LEG{GUI.currAX} = [];
                   SP.CB(GUI.currAX) = [];                   
                   GUI.currAX = max(1, GUI.currAX-1);
               end
               set(GUI.FIG_SLIDER, 'Max', get(GUI.FIG_SLIDER,'Max')-1)
           else
               error('userinput string unknown')
           end
           set(GUI.AXES, 'Visible', 'off');
           GUI.AXES = SP.AX{GUI.currAX};
           set(GUI.AXES, 'Position', GUI.AXISPOS, 'Parent', GUI.FIG);
           set(GUI.AXES, 'Visible', 'on');           
           addLegend()
           addColorbar()
           set(GUI.FIG, 'Name', ['ScrollPlots (' int2str(GUI.currAX) '/' int2str(length(SP.AX)) ')']);
       end
    end
    
    function addLegend()
        if ~isempty(SP.LEG) && ~isempty(SP.LEG{GUI.currAX})
            legend(SP.LEG{GUI.currAX}{:})
        end
    end

    function addColorbar()
        if SP.CB(GUI.currAX)
            colorbar
        end
    end
end