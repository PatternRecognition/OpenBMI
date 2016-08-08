clear all; close all; clc;
OpenBMI('E:\Test_OpenBMI\OpenBMI') % Edit the variable BMI if necessary
global BMI;
BMI.EEG_DIR=['E:\Test_OpenBMI\BMI_data\RawEEG'];

%% DATA LOAD MODULE
file=fullfile(BMI.EEG_DIR, '\calibration_motorimageryVPkg');
marker={'1','left';'2','right';'3','foot';'4','rest'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', [100]});
% 
% %% if you can redefine the marker information after Load_EEG function
% %% use  [marker, markerOrigin]=prep_defineClass(EEG.marke, marker)
% 
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
CNT=prep_selectClass(CNT,{'class',{'right', 'left'}});
SMT=prep_segmentation(CNT, {'interval',[-2000 5000]});

flt_CNT=prep_filter(CNT,{'frequency', [7 13]});

%% Setting for real-time topoplot
MNT = opt_getMontage(SMT);
center = [0 0];                   
theta = linspace(0,2*pi,360);  
x = cos(theta)+center(1);  
y = sin(theta)+center(2);  
oldUnit = get(gcf,'units');
set(gcf,'units','normalized');
H = struct('ax', gca);
set(gcf,'CurrentAxes',H.ax);
tic
xe_org = MNT.x';
ye_org = MNT.y';
resolution = 58;
maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));
xx = linspace(-maxrad, maxrad, resolution);
yy = linspace(-maxrad, maxrad, resolution)';

%====================================================================== 
tmpfig2 = prep_envelope(flt_CNT);
s_1=1;
b_size1=100;
s_size=10;
for i = 1:10000
    %figure 1
    subplot(1,2,1)
    tm2.x=flt_CNT.x(s_1:s_1+b_size1,:);
    tm2.fs = 100;
    tmp3 = prep_envelope(tm2);
    tm1=mean(tmp3.x(b_size1-50:b_size1,:));
    visual_topoplot(tm1, xe_org, ye_org, xx, yy);
    drawnow;
    % figure 2
    subplot(1,2,2)
    tmpfig1=tmpfig2.x(s_1:s_1+b_size1,:);
    tmpfig3=mean(tmpfig1(b_size1-50:b_size1,:));
    visual_topoplot(tmpfig3, xe_org, ye_org, xx, yy);
    drawnow;
    s_1=s_1+s_size;
end
%% Setting plot
% ----------------------------------------------------------------------
% nose plot
% tic
% nose = [1 1.2 1];
% nosi = [83 90 97]+1;
% H.nose = plot(nose.*x(nosi), nose.*y(nosi), 'k', 'linewidth', 2.4); 
% hold on;
% 
% % ----------------------------------------------------------------------
% % ears plot
% earw = .08; earh = .3;
% H.ears(1) = plot(x*earw-1-earw, y*earh, 'k',  'linewidth', 2);
% H.ears(2) = plot(x*earw+1+earw, y*earh, 'k',  'linewidth', 2);
% hold on;
% ----------------------------------------------------------------------
% main circle plot
% H.main = plot(x,y, 'k', 'linewidth', 2.2);                 
% set(H.ax, 'xTick',[], 'yTick',[]);
% axis('xy', 'tight', 'equal', 'tight');
% hold on;
% ----------------------------------------------------------------------
% Rendering the contourf
% xe_add = cos(linspace(0,2*pi,resolution))'*maxrad;
% ye_add = sin(linspace(0,2*pi,resolution))'*maxrad;
% % xe = [xe_org; xe_add];
% % ye = [ye_org; ye_add];
% 
% xe = [xe_org; xe_add];
% ye = [ye_org; ye_add];
% 
% xe_add = cos(linspace(0,2*pi,resolution))';
% ye_add = sin(linspace(0,2*pi,resolution))';
% 
% xe = [xe;xe_add];
% ye = [ye;ye_add];





