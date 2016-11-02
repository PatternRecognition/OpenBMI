function visual_scalpPlot(SMT, CNT, varargin)
% Description:
%   Draw  scalp topographies for all selected intervals,separately for each each class.
%   Scalp topographies of each classe are plotted in one row, and shared the same color map
%   scaling in each classes.
%
% Example Code:
%    visual_scalpPlot(SMT,CNT, {'Ival' , [start interval : time increase parameter: end intercal]});
%
% Input:
%   visual_scalpPlot(SMT,CNT, <OPT>);
%   SMT: Data structrue (ex) Epoched data structure
%   CNT: Continuous data structure
%
% Option:
%      .Ival - Selecting the interested time interval depending on time increase parameter
%                 (e.g. {'Ival' [ [-2000 : 1000: 4000]])
%
% Return:
%   Scalp topographies
%
% See also:
%    opt_getMontage, opt_cellToStruct
%
% Ji Hoon, Jeong
% jh_jeong@korea.ac.kr

%%
MNT = opt_getMontage(SMT);
opt = opt_cellToStruct(varargin{:});
opt.Interval = abs(opt.Ival(1)-opt.Ival(2));
for i = 1: size(SMT.class,1)
    for seg = 1: size(opt.Ival, 2)-1
        %     figure()
        subplot(size(SMT.class,1),size(opt.Ival,2)-1,((i-1)*(size(opt.Ival, 2)-1))+seg)
        
        center = [0 0];
        theta = linspace(0,2*pi,360);
        
        x = cos(theta)+center(1);
        y = sin(theta)+center(2);
        
        oldUnit = get(gcf,'units');
        set(gcf,'units','normalized');
        
        H = struct('ax', gca);
        set(gcf,'CurrentAxes',H.ax);
        
        % ----------------------------------------------------------------------
        % nose plot
        nose = [1 1.2 1];
        nosi = [83 90 97]+1;
        H.nose = plot(nose.*x(nosi), nose.*y(nosi), 'k', 'linewidth', 2.4 );
        
        hold on;
        
        % ----------------------------------------------------------------------
        % ears plot
        earw = .08; earh = .3;
        H.ears(1) = plot(x*earw-1-earw, y*earh, 'k',  'linewidth', 2);
        H.ears(2) = plot(x*earw+1+earw, y*earh, 'k',  'linewidth', 2);
        hold on;
        
        % ----------------------------------------------------------------------
        % main circle plot
        H.main = plot(x,y, 'k', 'linewidth', 2.2);
        set(H.ax, 'xTick',[], 'yTick',[]);
        axis('xy', 'tight', 'equal', 'tight');
        hold on;
        %         if (i == 1 && seg == 1)
        %             ylabel([SMT.class{i,2},' class'])
        %             hold on;
        %         elseif (i == 2 && seg == 1)
        %             ylabel([SMT.class{i,2},' class'])
        %             hold on;
        %         elseif (i == 3 && seg == 1)
        %             ylabel([SMT.class{i,2},' class'])
        %             hold on;
        %         elseif (i == 4 && seg == 1)
        %             ylabel([SMT.class{i,2},' class'])
        %             hold on;
        %         end
        % ----------------------------------------------------------------------
        % Rendering the contourf
        xe_org = MNT.x';
        ye_org = MNT.y';
        avgSMT= prep_average(SMT);
        
        SMTintervalstart = find(avgSMT.ival == opt.Ival(seg));
        SMTintervalEnd = SMTintervalstart+opt.Interval/10;
        UpdatedSMT{seg} = avgSMT.x(SMTintervalstart:SMTintervalEnd,:,:);
        w{seg,i} = UpdatedSMT{seg}(:,i,:);
        w{seg,i} = squeeze(w{seg,i});
        inputx{seg,i}  = mean(w{seg,i},1);
        
        
        % w = w(:);
        % resolution = 101;
        resolution = 500;
        
        maxrad = max(1,max(max(abs(MNT.x)),max(abs(MNT.y))));
        
        xx = linspace(-maxrad, maxrad, resolution);
        yy = linspace(-maxrad, maxrad, resolution)';
        
        xe_add = cos(linspace(0,2*pi,resolution))'*maxrad;
        ye_add = sin(linspace(0,2*pi,resolution))'*maxrad;
        w_add = ones(length(xe_add),1)*mean(inputx{seg,i});
        
        xe = [xe_org; xe_add];
        ye = [ye_org; ye_add];
        inputx{seg,i} = [inputx{seg,i}'; w_add];
        
        % xe_add = cos(linspace(0,2*pi,resolution))';
        % ye_add = sin(linspace(0,2*pi,resolution))';
        %
        % xe = [xe;xe_add];
        % ye = [ye;ye_add];
        % w = [w; zeros(length(xe_add),1)];
        
        [xg,yg,zg] = griddata(xe, ye, inputx{seg,i}, xx, yy);
        contourf(xg, yg, zg, 50, 'LineStyle','none'); hold on;
        
        % ----------------------------------------------------------------------
        % disp electrodes
        for j = 1:size(xe_org,1)
            plot(xe_org(j), ye_org(j), 'k*'); hold on;
            set(0,'defaultfigurecolor',[1 1 1])
            
        end
        
        axis off;
        title({[SMT.class{i,2},' class'];['[' , num2str(opt.Ival(seg)), ' ~ ' , num2str(opt.Ival(seg+1)) , '] ms']})
        %         colorbar('vert');
    end
    
    
end


end

