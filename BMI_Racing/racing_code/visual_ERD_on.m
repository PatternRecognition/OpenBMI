function [ output_args ] = visual_ERD_on( band, scale, mode, plotChannels )
%VISUAL_ERD_ON Summary of this function goes here
%   Detailed explanation goes here

bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
orig_Dat=[];


% ---------------------------------------------------------------------- %
% --------------- Frequency bands for comparison ----------------------- %
% ---------------------------------------------------------------------- %
basic = [7 13];
visualFlag = true;
cIdx = 1;
colororder = [
	0.00  0.00  1.00
	0.00  0.50  0.00 
	1.00  0.00  0.00 
	0.00  0.75  0.75
	0.75  0.00  0.75
	0.75  0.75  0.00 
	0.25  0.25  0.25
	0.75  0.25  0.25
	0.95  0.95  0.00 
	0.25  0.25  0.75
	0.75  0.75  0.75
	0.00  1.00  0.00 
	0.76  0.57  0.17
	0.54  0.63  0.22
	0.34  0.57  0.92
	1.00  0.10  0.60
	0.88  0.75  0.73
	0.10  0.49  0.47
	0.66  0.34  0.65
	0.99  0.41  0.23
];

buffer_size=5000;
data_size=1500;
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));

%% Real-time visualization
%% Setting for real-time topoplot
load mnt
SMT.chan=aa;
SMT.chan=SMT.chan(1:31);
MNT = opt_getMontage(SMT);
% center = [0 0];
% theta = linspace(0,2*pi,360);
% x = cos(theta)+center(1);
% y = sin(theta)+center(2);
% oldUnit = get(gcf,'units');
% set(gcf,'units','normalized');
% H = struct('ax', gca);
% set(gcf,'CurrentAxes',H.ax);
tic
xe_org = MNT.x';
ye_org = MNT.y';
resolution = 100;
maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));
xx = linspace(-maxrad, maxrad, resolution);
yy = linspace(-maxrad, maxrad, resolution)';

% % nose plot
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
% % ----------------------------------------------------------------------
% % main circle plot
% H.main = plot(x,y, 'k', 'linewidth', 2.2);
% set(H.ax, 'xTick',[], 'yTick',[]);
% axis('xy', 'tight', 'equal', 'tight');
% hold on;
escapeKey = KbName('esc');
waitKey=KbName('s');
resetKey=KbName('r');
figure
play=true;
resetFlag = false;
x=-2;y=30;
while play
    
    [ keyIsDown, seconds, keyCode ] = KbCheck;
    if keyIsDown
        if keyCode(escapeKey)
            ShowCursor;
            break;
        elseif keyCode(resetKey)
            resetFlag = true;
        else
            
        end
    end
    
    
    [data, markertime, markerdescr, state] = bbci_acquire_bv(state);
    orig_Dat=[orig_Dat; data];
    if length(orig_Dat)>buffer_size % prevent overflow
        tic
        Dat=orig_Dat(end-buffer_size+1:end,:);
        orig_Dat=Dat;  %%
        Dat2.x=Dat;
        Dat2.fs=state.fs;
        %                 Dat=prep_resample(Dat2,500);
        Dat=Dat2.x;
        fDat=prep_filter(Dat, {'frequency', band;'fs',1000});%state.fs });
        switch mode
            case 1
                tm.x=fDat;tm.fs=state.fs;
                tm2 = prep_envelope(tm);
                vDat=mean(tm2(end-1000:end, :));
                vDat=vDat*-1;
            case 2
                tm=fDat(end-2000:end,:);
                vDat=var(tm)
                %                 vDat=vDat*-1;
        end
        figure(1);
        visual_topoplot(vDat(1:31), xe_org, ye_org, xx, yy, scale);
        % ---------------------------------------------------------------------- %
        % ---------------------- Feature visualization ------------------------- %
        % ---------------------------------------------------------------------- %
        cDat = prep_filter(Dat, {'frequency', basic; 'fs', state.fs});
        switch mode
            case 1
                cm.x=cDat;cm.fs=state.fs;
                tm2 = prep_envelope(cm);
                bDat=mean(tm2(end-1000:end, :));
                bDat=bDat*-1;
            case 2
                cm=cDat(end-2000:end,:);
                bDat=var(cm)
                %                 vDat=vDat*-1;
        end

        if visualFlag
            plotIdx = [];
            for i = 1 : length(plotChannels)
                for j = 1 : length(state.clab)
                    if strcmp(plotChannels{i}, state.clab{j})
                        plotIdx = [plotIdx j];
                        break;
                    end
                end
            end
            
            h = figure(2);
            set(h, 'position', [300 50 1200 450]);
            h(1)= subplot(1, 3, 1);
            set(h(1), 'position', [0.03 0.05 0.3 0.9]);
            title(h(1), plotChannels{1});
            set(h(1),'Nextplot','add');axis([x y x y]);
            grid on;
            h(2)= subplot(1, 3, 2);
            set(h(2), 'position', [0.36 0.05 0.3 0.9]);
            title(h(2), plotChannels{2});
            set(h(2),'Nextplot','add');axis([x y x y]);
            grid on;
            h(3)= subplot(1, 3, 3);
            set(h(3), 'position', [0.69 0.05 0.3 0.9]);axis([-2 6 -2 6]);
            title(h(3), plotChannels{3});
            set(h(3),'Nextplot','add');axis([x y x y]);
            grid on;
            visualFlag = false;
        end
        
        if resetFlag
            set(h(1),'Nextplot','replace');axis([x y x y]);
            set(h(2),'Nextplot','replace');axis([x y x y]);
            set(h(3),'Nextplot','replace');axis([x y x y]);
        end
        
        pDat = vDat(plotIdx);
        bDat = bDat(plotIdx);
        
        
        plot(h(1), pDat(1), bDat(1), 'x', 'LineWidth', 10, 'MarkerEdgeColor', colororder(cIdx, :));
        plot(h(2), pDat(2), bDat(2), 'x', 'LineWidth', 10, 'MarkerEdgeColor', colororder(cIdx, :));
        plot(h(3), pDat(3), bDat(3), 'x', 'LineWidth', 10, 'MarkerEdgeColor', colororder(cIdx, :));
        
        axis(h(1), [x y x y]);
        axis(h(2), [x y x y]);
        axis(h(3), [x y x y]);
        grid(h(1), 'on');
        grid(h(2), 'on');
        grid(h(3), 'on');
        
        title(h(1), plotChannels{1});
        title(h(2), plotChannels{2});
        title(h(3), plotChannels{3});
        
        
        if resetFlag
            set(h(1),'Nextplot','add');
            set(h(2),'Nextplot','add');
            set(h(3),'Nextplot','add');
            resetFlag = false;
        end
        
        cIdx = cIdx + 1;
        if cIdx > 20
            cIdx = 1;
        end
        
% -----------------------------------------------------------------------%
        
        drawnow;
        toc
        
    end
end


end

