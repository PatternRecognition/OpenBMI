%DEMO_CUSTOMIZING_GRID_PLOTS
%
%Description:
% This script demonstrates how to use the function 'gird_plot' in order
% to achieve various kinds of visualzations.

% Author(s): Benjamin Blankertz, Mar 2005

file= [DATA_DIR 'siemensMat/Ben_05_01_07/einzelobjekteBen'];
[cnt, Mrk, mnt]= loadProcessedEEG(file, 'display');

%% Plot the stimulus-aligned ERPs. Markers of stimuli are in
%% the substructure 'trg'.
mrk= Mrk.trg;

%% Define some channel layout.
grd= sprintf('EOGh,legend,Fz,scale,EOGv\nT7,C3,Cz,C4,T8\nTP7,P3,Pz,P4,TP8\nP7,PO3,POz,PO4,P8');
mnt= mnt_setGrid(mnt, grd);

%% The first plot is like the one in 'demo_visualize_ERPs'.
epo= makeEpochs(cnt, mrk, [-200 800]);
epo= proc_baseline(epo, [-200 0]);
erp= proc_average(epo);
grid_plot(erp, mnt);
fprintf('Press a key to continue.\n');
pause;

%% Shrink the axes of the EOG channels
mnt= mnt_shrinkNonEEGchans(mnt);
%% Define colors for the two classes
grid_opt= struct('colorOrder',[0 0.7 0; 1 0 0]);
%% All line properties are applied to the curves. If you want, e.g., thicker
%% lines, just define
grid_opt.lineWidth= 2;
%% To change the appearance of the axis titles you can set the properties
%% axisTitle* where * is a text property like FontSize, FontWeight, Color,
%% HorizontalAlignment.
grid_opt.axisTitleFontWeight= 'bold';
grid_plot(erp, mnt, grid_opt);
fprintf('Press a key to continue.\n');
pause;

%% To have xticks at specified locations just define property 'xTick'.
%% This automatically shrinks the axes in the vertical direction in order
%% to have space for the xticklabels. Internally the shrinking is done
%% by setting property 'shrinkAxes' to 0.8, which is interpreted as [1 0.8],
%% i.e., horizontal shinkage by factor 1 (i.e. no shrinkage) and vertical
%% shrinkage by factor 0.8.
grid_opt.xTick= [0 200 400 600];
grid_plot(erp, mnt, grid_opt);
fprintf('Press a key to continue.\n');
pause;

%% In order to shink axes also horizontally set
grid_opt.shrinkAxes= [0.9 0.8];
grid_plot(erp, mnt, grid_opt);
fprintf('Press a key to continue.\n');
pause;

%% Remove xticks.
grid_opt= rmfield(grid_opt, {'xTick','shrinkAxes'});

%% Apart from the box-type axes, you can also get cross-type axes by setting:
grid_opt.axisType= 'cross';
%% To make all the background white:
grid_opt.figure_color= [1 1 1];
grid_plot(erp, mnt, grid_opt);
fprintf('Press a key to continue.\n');
pause;

%% In the cross-type axes it might look cool to have the curve potentially
%% exceed the axes
grid_opt.oversizePlot= 1.5;
grid_plot(erp, mnt, grid_opt);
fprintf('Press a key to continue.\n');
pause;

%% The axes can also easily be arranged as the electrodes on the head
%% with the function 'mnt_scalpToGrid'. Since the electrodes PO1 and PO2
%% too much in the crowd, we remove them first.
head= mnt;
kick_out= chanind(head, 'PO1,2');
head.x(kick_out)= NaN;
head= mnt_scalpToGrid(head);
%% Again, let there be cross-type axis with oversized plots.
grid_opt= struct('axisType','cross');
grid_opt.oversizePlot= 1.5;
%% Axis title shall be small and left aligned.
grid_opt.axisTitleFontSize= 8;
grid_opt.axisTitleHorizontalAlignment= 'left';
%% Tick marks on the axes should be small.
grid_opt.zeroLineTickLength= 1;
grid_plot(erp, head, grid_opt);
