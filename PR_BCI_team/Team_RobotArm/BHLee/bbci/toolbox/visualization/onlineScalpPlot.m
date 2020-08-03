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

% TODO: extended help (installation of OpenGL, Java etc)
%
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
% Parameter:
% (help setElectrodeMontage)


% if length(varargin)>0
%     opt = varargin{1};
% else    
    opt = struct;
    % end 
% initialization of the arguments
opt = set_defaults(opt,'colorMap','jet',...
    'range',[-2 2],...
    'target_fs',100,...
    'buffer_length',10000,... 
    'var_length',[2000 10000],...
    'host','127.0.0.1', ...
    'clab', {'not','E*'});

bbciclose
state = acquire_bv(100,opt.host);
mnt = setElectrodeMontage(state.clab,'C3,C4');

% kick out some values which are not in the grid:
chans = chanind(mnt.clab, opt.clab);
chans = chans(find(~isnan(mnt.x(chans))));
mnt.clab = {mnt.clab{chans}};
mnt.x = mnt.x(chans);
mnt.y = mnt.y(chans);

% get the colormap
colMap = loadColormap(opt.colorMap);
% replace this by get_colormap(opt.colorMap)

% triangulation will be given to the visualisation.
x = mnt.x;
y = mnt.y;
tri = delaunay(x,y);

% get a new graphics object
javaPlot = jScalpPlot(1000,1000);

% initialize some values of the object
javaPlot.initScalp(colMap,[mnt.x,mnt.y],tri,opt.range);

%%%%% EEG data acquisition %%%%%%%%

% initialize a data buffer for EEG
cnt= struct('fs', opt.target_fs);
cnt.clab= mnt.clab;
bN= ceil(opt.buffer_length/1000*cnt.fs);
cnt.x= zeros(2*bN, length(cnt.clab));
%newBlock = copyStruct(cnt,'x');

online_variance(length(chans),bN,opt.var_length);
[b,a]=getButterFixedOrder([8 15],opt.target_fs,5);
state_flt = [];
state_var = [];
run = 1;
%keyboard
% just for testing:
%data = rand(size(mnt.x));
%ind = ceil(rand*.1*length(mnt.x));
%ind = ceil(rand(ind,1)*length(mnt.x));
%alpha = .5;

%%%%% Visualization loop %%%%%
while run
    
    [cnt.x]= acquire_bv(state);
    cnt.x = cnt.x(:,chans);
    [bl,state_flt] = online_filt(cnt,state_flt,b,a);
    %    [vari,state_var] = online_variance(bl,state_var,bN,opt.var_length);
    vari = online_variance(bl,bN,opt.var_length);
    data = log(vari.x(1,:)./vari.x(2,:));
     
    % generate new data
    %data(ind) = max(opt.range(1),min(opt.range(2),data(ind) + randn(size(ind))*alpha));
    % display new data on the screen

    scalpreturn = javaPlot.setData(data);
    run = scalpreturn.isAlive;
    % read out other interesting return values from object.
    %fs = scalpreturn.getFs;
    %pause(1/fs);
    %keyboard
end 
acquire_bv('close');

javaPlot.setVisible(false)
clear javaPlot