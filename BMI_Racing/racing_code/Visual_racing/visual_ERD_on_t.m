function [ output_args ] = visual_ERD_on( band, scale, mode )
%VISUAL_ERD_ON Summary of this function goes here
%   Detailed explanation goes here

bbci_acquire_bv('close');
params = struct;
state = bbci_acquire_bv('init', params);
orig_Dat=[];

buffer_size=5000;
data_size=1500;
orig_Dat=zeros(buffer_size, size(state.chan_sel,2));

%% Real-time visualization
%% Setting for real-time topoplot
load mnt
SMT.chan=aa;
SMT.chan=SMT.chan(1:32);
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

figure
play=true;
while play
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
        visual_topoplot(vDat(1:32), xe_org, ye_org, xx, yy, scale);
        drawnow;
        toc
        
        [ keyIsDown, seconds, keyCode ] = KbCheck;
        if keyIsDown
            if keyCode(escapeKey)
                ShowCursor;
                play=false;
            elseif keyCode(waitKey)
                warning('stop')
                GetClicks(w);
                Screen('Close',tex1);
            else                
            end
        end
        %         pause(0.05);
    end
end


end

