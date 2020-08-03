function onlineScalpPlot(varargin)
% onlineScalpPlot(opt)
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
% Types of Constructors (jScalpPlot.jar):
%           jScalpPlot(int width, int height)
%           jScalpPlot(String description, int width, int height)
%           jScalpPlot(String title, String description, int width, int height)
%
%           .initScalp(colorMap,[mnt.x,mnt.y],delaunay,opt.range);
%           .setData(float[] daten) returns a ScalpReturn object
%

% How to install the jar-files properly:
% 1. Find the directory of your Java Runtime Enviroment from Matlab (e.g.
% $matlabroot\sys\java\jre\win32\jre), ScalpPlot works fine with jre1.3
%
% 2. Install gl4java (www.jausoft.com/gl4java/) and use the directory from
% above, perhaps you must mkdir the lib\ext directory
%
% 3. Set the Classpath, in Matlab 6 edit classpath.txt and paste
% "$BCI_DIR\eegVisualize\jScalpPlot.jar" and restart Matlab
% in Matlab 7 exist dynamic classpath setting
% 
% 4. now it should work.
%
% (5. After the installation of Java SDK (Software Development Kit),
% don't forget to set the PATH to the $j2sdk\bin directory,
% install gl4java in $j2sdk\jre
% 
% 6. After a change in jScalpPlot.java, compile with "javac jScalpPlot.java"
% and pack with "jar cvfm jScalpPlot.jar mani.txt *.java *.class *.txt")
%

if length(varargin)>0
     opt = varargin{1};
else    
    opt = struct;
end 
% initialization of the arguments
opt = set_defaults(opt,'colorMap','jet',...
    'range',[-2 2],...
    'target_fs',100,...
    'buffer_length',10000,... 
    'var_length',[2000 10000],...
    'host','brainamp', ...
    'clab', {'not','E*','xE*'},...
    'title',{{'BBCI Visualization','Player 1'},{'BBCI Visualization','Player 2'}},...
    'fps',25);

bbciclose
state = acquire_bv(100,opt.host);
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
dat_chans = {};
for i = 1:playernum
    if i==1
        % player one: select only the non-x-channels
        clab = {state.clab{chanind(state.clab,{'not','x*'})}};
        clab = {clab{chanind(clab,opt.clab)}};
        % kick out some values which are not in the grid:
        mnt(i) = setElectrodeMontage(clab,'C3,C4');
        ind = find(~isnan(mnt(i).x));
        mnt(i).clab = {mnt(i).clab{ind}};
        mnt(i).x = mnt(i).x(ind);
        mnt(i).y = mnt(i).y(ind);
        
        clab = {clab{ind}};
        chans{i} = chanind(state.clab, clab);    
    else
        % player two: select only the x-channels.
        xclab = {state.clab{chanind(state.clab,{'x*'})}};
        xclab = {xclab{chanind(xclab,opt.clab)}};
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

    % get a new graphics object
    javaPlot{i} = jScalpPlot(opt.title{i}{:},350,350);

    % initialize some values of the object
    javaPlot{i}.initScalp(colMap,[mnt(i).x,mnt(i).y],tri,opt.range);
    dat_chans{i} = length([dat_chans{:}])+1:length([dat_chans{:}])+length([chans{i}]);
end

%%%%% EEG data acquisition %%%%%%%%

% initialize a data buffer for EEG
cnt= struct('fs', opt.target_fs);
cnt.clab= {state.clab{[chans{:}]}};
bN= ceil(opt.buffer_length/1000*cnt.fs);
cnt.x= zeros(2*bN, length(cnt.clab));

online_variance(length([chans{:}]),bN,opt.var_length);
[b,a]=getButterFixedOrder([8 15],opt.target_fs,5);
state_flt = [];
run = 1;
waitForSync(0);

%%%%% Visualization loop %%%%%
while run
    
    [cnt.x]= acquire_bv(state);
    cnt.x = cnt.x(:,[chans{:}]);
    [bl,state_flt] = online_filt(cnt,state_flt,b,a);
    vari = online_variance(bl,bN,opt.var_length);
    data = log(vari.x(1,:)./vari.x(2,:));
    % display new data on the screen
    for i=1:length(mnt)
        scalpreturn{i} = javaPlot{i}.setData(data(dat_chans{i}));
        run = run&scalpreturn{i}.isAlive;
    end 
    waitForSync(1000/opt.fps);
end 
acquire_bv('close');
for i = 1:length(mnt)
    javaPlot{i}.setVisible(false)
end 
clear javaPlot