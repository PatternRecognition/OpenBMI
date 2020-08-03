function dis = feedback_flipper_init(fig,opt)
    clf();
	set(fig,'Position',opt.position);
	
	% all bars are created with the functin createBar() create bar sets the local coordinates system
	% to a size of 800 x 640 so all positions and sizes are in this space

	borderPos = opt.threshold * 380;		% the x position for the threshold borders
	% creates the bar for the current classifier data we will recieve
	currentBar = createBar(fig,20,560,780,620,[1 1 1]);			
   	currentStatus = createBar(fig,400,565,775,615,[0 0 1]);
    % the threshold bars
	createBar(fig,400 + borderPos,560,400 + borderPos + 1,620,[0 0 0]);
	createBar(fig,400 - borderPos,560,400 - borderPos - 1,620,[0 0 0]);
    
	% creates the bar for the internal classifier data we produce
	internalBar = createBar(fig,20,480,780,540,[1 1 1]);
   	internalStatus = createBar(fig,400,485,775,535,[1 1 0]);
    % the threshold bars
	createBar(fig,400 + borderPos,480,400 + borderPos + 1,540,[0 0 0]);
	createBar(fig,400 - borderPos,480,400 - borderPos - 1,540,[0 0 0]);

	% the bars for the left and right flippers
	% they are rotated into idle mode
	flipLeft = createBar(fig,40,40,150,50,[1 0 0]);
    flipRight = createBar(fig,650,40,760,50,[1 0 0]);
	rotate(flipLeft,[0 0 1],-22.5,[45 45 0]);
	rotate(flipRight,[0 0 1],22.5,[755 45 0]);
    

	dis = struct('currentBar',currentBar,...
					'currentStatus',currentStatus,...
					'internalBar',internalBar,...
					'internalStatus',internalStatus,...
					'flipLeft',flipLeft,...
					'flipRight',flipRight);
    
	% creates a rectangle with the given color
	function bar = createBar(parent,xStart,yStart,xEnd,yEnd,color)
        xValues = [xStart xStart;xEnd xEnd;xStart xEnd];
        yValues = [yStart yEnd;yStart yEnd;yEnd yStart];
        flip = patch('XData',xValues,'YData',yValues,'FaceColor',color,'EdgeColor',color);
        flipParent = get(flip,'Parent');
        set(flipParent,'Position',[0 0 1 1],'XLim',[0,800],'YLim',[0,640],'Visible','off');
        
        bar = flip;
   end
end
