%% Run Oddball experiments 
% Two conditions: 
% 1) Same Shape Different Color 
% 3) Different Shapes Same Color 
%
% Test all three conditions once, in a random order.


% conditions
% shape list:
% 0. Circle (Standard)
% 1. Rotated rectange 45° 
% 2. Rotated rectange -45 °
% 3. Triangle up
% 4. Triangle down
% 5. Hourglass


shape=[1:7];
dev_color={[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255]};
std_color=[255,255,255];

conditions=[ shape(1) dev_color{1};
             shape(1) dev_color{2};
             shape(1) dev_color{3};
             shape(1) dev_color{4};
             shape(1) dev_color{5};
             shape(2) [255,255,255];
             shape(3) [255,255,255];
             shape(6) [255,255,255];
             shape(7) [255,255,255];];
%             shape(4) [255,255,255];
%             shape(5) [255,255,255];
%             shape(1) dev_color{6};
conditionsOrder= randperm(size(conditions,1));

condition_tags={'ColorRed','ColorGreen','ColorBlue','ColorYellow','ColorMagenta',...
   'ShapeCircle','ShapeRectangleForward','ShapeTriangleUp','ShapeHourglass','ShapeRectangleBackward',...
   'ShapeTriangleDown','ColorCyan'}
pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], ...
     'gui',0);

% practice
for ii= [1 7],
    fprintf('Press <RETURN> to practice %s Oddball experiment.\n',condition_tags{ii}), pause;
    setup_oddball
    pyff('set', 'nStim', 20);
    pyff('setdir','');
    pyff('play')
    stimutil_waitForMarker({'S253', 'R  2', 'R  4', 'R  8'});
    pyff('quit');
end

for ii=conditionsOrder
    fprintf('Press <RETURN> to start %s Oddball experiment.\n',condition_tags{ii}), pause;
    setup_oddball
    pyff('setdir','basename',['oddball' condition_tags{ii}]);
    pyff('play')
    stimutil_waitForMarker({'S253', 'R  2', 'R  4', 'R  8'});
    pyff('quit');
end
