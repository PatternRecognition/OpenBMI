function h = fig2subplot(hfigs, varargin)
% FIGS2SUBPLOT - Arrange figures in subplots
%
%Description:
% Takes a number of figure handles and arranges figures in a single subplot.
% The order of figure handles should correspond to the numbering of the
% axes in the subplot.
%
%Usage:
% H = figs2subplot(HFIGS, <OPT>)
%
%Input:
% HFIGS: a vector of figure handles
% OPT: struct or property/value list of optional properties. It can be used
% in two ways to generate a figure.
%(1): Give the number of rows and columns, a subplots with size equal to the
%     maximum height and width of the figures are created. The sizes 
%     of the figures are preserved
%  rowscols:     the number of rows and columns, eg. [2 1]
%  innerMargin:  minimum margin between plots in pixels [horizontal vertical]
%  outerMargin:  outer margins [left right top bottom] of the whole plot in
%                pixels
%
%(2): Provide a predefined subplot figure
%  hmain:        handle of a predefined subplot figure. The positions of
%                 the subplots will be used to place the figures.
%(3): Make a subplot of rows and columns
%  rows:         normalized sizes of the rows (e.g., [.1 .3 .4])
%  cols:         normalized sizes of the cols. If there is space over, the
%                 rows and columns are evenly spread.
%  margin:       outer margins [left right top bottom] of the whole plot
%
%(4): Make an arbitrary subplot
%  positions:    a n x 4 matrix of normalized axes position data 
%                 [left bottom width height]
%
% If none of these arguments is set, a default n x 1 subplot is created, 
% where n is the number of figure handles
%
% Other options
%  .deleteFigs -   the original figures are deleted after they were copied
%                  into the new figure
%  .label      -   automatically label the subfigures by running numbers or letters.
%                  Specify label type by a string, eg. '(a)' (for (a) (b), etc), 
%                  'a', 'a.', capital letters, or numerical variants 
%                  '(1)' '1', '1.'  (default []). 
%                  Alternatively, you can provide a cell array of
%                  strings representing custom labels.
%  .labelPos   -   positions of the labels, the values correspond to the
%                  values used for legend positions (default 'NorthWest')
%  .labelOpt   -   formatting options for label as cell array 
%                  (default {'FontSize' 12 'FontWeight' 'bold'})
%                 
%
%Output:
% H = Handle to the new subplot figure and its children
% .axes     - axes wherein the new subplots are placed
% .children - the copied graphics objects
%
%Example: (4 figures with four different colormaps arranged in a 2x2 subplot)
%
% close all
% [X,Y,Z] = peaks(30); % fig 1
% surf(X,Y,Z), colormap jet
% figure,pcolor(rand(20)) % fig 2
% colormap copper
% figure,contourf(peaks(40),10),colormap winter % fig 3
% figure,plot(sin(1:.1:pi)'*[1:22],'LineWidth',3); % fig 4
% H = fig2subplot([1:4],'rowscols',[2 2],'label','(a)')

% Author(s): Matthias Treder, Benjamin Blankertz Nov 2009
% Aug 2010: Added automatic labeling (mt)

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'hmain',[],...
                 'rowscols',[],...
                 'rows',[],...
                 'cols',[],...
                 'innerMargin',[0 0],...
                 'outerMargin',[10 10 10 10],...
                 'margin',[.05 .05 .05 .05],...
                 'positions',[],...
                 'deleteFigs',0, ...
                 'label',[], ...
                 'labelPos','northwest', ...
                 'labelOpt',{'FontSize' 14 'FontWeight' 'bold'});

if isdefault.hmain && isdefault.rowscols && isdefault.rows && ... 
    isdefault.positions
  % Nothing was set
  opt.rowscols = [numel(hfigs) 1];
end

% Prepare labels
if ~isempty(opt.label) && ~iscell(opt.label)
  ll = cell(1,numel(hfigs));
  for ii=1:numel(hfigs)
    switch(opt.label)
      case '(a)', ll{ii} = ['(' char(96+ii) ')'];
      case '(A)', ll{ii} = ['(' char(64+ii) ')'];
      case 'a',   ll{ii} = char(96+ii);
      case 'A',   ll{ii} = char(64+ii);
      case 'a.',  ll{ii} = [char(96+ii) '.'];
      case 'A.',  ll{ii} = [char(64+ii) '.'];
      case '(1)', ll{ii} = ['(' num2str(ii) ')'];
      case '1',   ll{ii} = num2str(ii);
      case '1.',  ll{ii} = [num2str(ii) '.'];
    end
  end
  opt.label = ll;
end

%% Prepare the axes of the new figure
h = strukt('main',[],'axes',[]);
h.cmap_start_idx = []; % denotes the start indices of the separate colormaps in the compound colormap
if isempty(opt.hmain)
    % No figure yet, make one
    h.main = figure;
    if ~isempty(opt.rowscols)
        % option 1
        % Get height and width in px of figures
        set(hfigs,'units','pixels');
        pos = get(hfigs,'position');
        pos = cat(1,pos{:});
        widmat=[];heimat=[]; % width and height matrix
        idx=1;
        for rr = 1:opt.rowscols(1)  % all rows
          for cc=1:opt.rowscols(2)  % all cols
            if numel(hfigs)>=idx
              widmat(rr,cc) = pos(idx,3); heimat(rr,cc)=pos(idx,4);
            else
              widmat(rr,cc) = 0; heimat(rr,cc)=0;
            end
            idx=idx+1;
          end
        end
        rows = max(heimat,[],2);
        cols = max(widmat,[],1);
        % Set figure size
        figwid = opt.outerMargin(1)+opt.outerMargin(2)+opt.innerMargin(1)*(numel(cols)-1);
        fighei = opt.outerMargin(3)+opt.outerMargin(4)+opt.innerMargin(2)*(numel(rows)-1);
        figwid = figwid + sum(cols);
        fighei = fighei + sum(rows);
        set(h.main,'units','pixels');
        mainpos = get(h.main,'position');
        set(h.main,'position',[mainpos(1:2) figwid fighei]);
        % Get back to normalized units
        set(h.main,'units','normalized');
        widmat = widmat/figwid;
        cols=cols/figwid;
        heimat = heimat/fighei;
        rows=rows/fighei;
        opt.outerMargin(1) = opt.outerMargin(1)/figwid;
        opt.outerMargin(2) = opt.outerMargin(2)/fighei;
        opt.outerMargin(3) = opt.outerMargin(3)/figwid;
        opt.outerMargin(4) = opt.outerMargin(4)/fighei;
        opt.innerMargin(1) = opt.innerMargin(1)/figwid;
        opt.innerMargin(2) = opt.innerMargin(2)/fighei;
        % Place new axes
        h.axes = []; hidx = 1;
        rowStart = 1-opt.outerMargin(3)-rows(1); % start at top 
        for rr = 1:numel(rows)
            colStart = opt.outerMargin(1); % start left
            for cc = 1:numel(cols)
                h.axes(hidx) = axes('position', ...
                    [colStart rowStart widmat(rr,cc) heimat(rr,cc)], ...
                    'parent',h.main,'visible','off');
                axis off;
                colStart = colStart + cols(cc) + opt.innerMargin(1);
                hidx = hidx+1;
                if hidx > numel(hfigs); break;end;
            end
            if rr<numel(rows)
                rowStart = rowStart - opt.innerMargin(2) - rows(rr+1);
            end
            if hidx > numel(hfigs); break;end;  % break when rows*cols > number of figures
            numel(hfigs)
        end
    elseif isempty(opt.positions)
        % option 2
        % Inner space left for rows and cols
        colSpace = 1-opt.margin(1)-opt.margin(2);
        rowSpace = 1-opt.margin(3)-opt.margin(4);
        % Calculate space left over between axes
        rowOver = (rowSpace-sum(opt.rows)) / (numel(opt.rows)-1);
        colOver = (colSpace-sum(opt.cols)) / (numel(opt.cols)-1);
        h.axes = []; hidx = 1;
        rowStart = 1-opt.margin(3)-opt.rows(1); % start at top 
        for rr = 1:numel(opt.rows)
            colStart = opt.margin(1); % start left
            for cc = 1:numel(opt.cols)
                h.axes(hidx) = axes('position', ...
                    [colStart rowStart opt.cols(cc) opt.rows(rr)], ...
                    'parent',h.main,'visible','off');
                axis off;
                colStart = colStart + opt.cols(cc) + colOver;
                hidx = hidx+1;
            end
            if rr<numel(opt.rows)
                rowStart = rowStart - rowOver - opt.rows(rr+1);
            end
        end
    else
        % option 3
        for ii=1:size(opt.positions,1)
            h.axes(ii) = axes('position',opt.positions(ii,:),'parent',h.main);
        end
    end
else
    % Option 1, nothing much to do ..
    h.main = opt.hmain;   % figure
    h.axes = findobj(h.main,'Type','Axes');   % its children axes
    h.axes = flipud(h.axes);  % to start with ax 1
end

%% Copy colormap from first figure to the new one
set(h.main, 'Colormap', get(hfigs(1),'Colormap'));

%% Traverse old figures and place them in the new plot
for ii=1:numel(hfigs)
  % Save connection between Colorbars and their Parent Axes
  hcb= findobj(hfigs(ii), 'Tag','Colorbar');
  for jj= 1:length(hcb),
    hhcb= handle(hcb(jj));
    if isfield(hhcb,'axes')
      hpa= double(hhcb.axes); % that's the axes the colorbar is refering to
    else
      hpa = double(hhcb);
    end
    if ~isempty(hpa),
      ud= get(hcb(jj), 'UserData');
      ud.ParentAxis= hpa;
      set(hcb(jj), 'UserData',ud);
    end
  end
  
  % Check if colormap's different
  newColmap = get(hfigs(ii),'colormap');
  if ii>1 && ~isequal(newColmap,get(h.main,'colormap'))
     acm = fig_addColormap(newColmap,'colormap');
  end
  % Copy all objects from old fig to new
  % use findall not findobj to also find hidden objects like fake axes
  % produced by axespos()
  oldax = findall(hfigs(ii),'Type','Axes'); 
  
  % Sometimes colors are clipped in the original images (because colors
  % fall out of CLim color limits). To ensure clipping is preserved after a
  % new colorbar is being added, these values should be set to the CLim
  % limits.
  for oo=1:numel(oldax)
    cLim = get(oldax(oo),'CLim');
    child = get(oldax(oo),'Children');
    % Patch objects include scalp maps
    pat = findobj(child,'Type','patch');
    for pp=1:numel(pat)
      cd = get(pat(pp),'Cdata');
      cd(cd(:)<cLim(1))=cLim(1);   % Lower than min is set to min
      cd(cd(:)>cLim(2))=cLim(2);   % Greater than max is set to max
      set(pat(pp),'Cdata',cd);    
    end
    % Image objects include time-frequency plots
    im = findobj(child,'Type','image','Tag','');
    for jj=1:numel(im)
      cd = get(im(jj),'Cdata');
      cd(cd(:)<cLim(1))=cLim(1);   % Lower than min is set to min
      cd(cd(:)>cLim(2))=cLim(2);   % Greater than max is set to max
      set(im(jj),'Cdata',cd);
    end
  end
  
  % Copy objects from old to new figure
  newax = copyobj(oldax,h.main); % Copy figure
  % Get position of subplot and positions of old plot
  set(hfigs(ii),'units','normalized');
  oldpos = get(oldax,'position');
  subpos = get(h.axes(ii),'position');
  % Make new positions
  if ~iscell(oldpos), oldpos = {oldpos}; end;
  for kk=1:numel(oldpos)
    oo = oldpos{kk};  % copy into oo for convenience
    newl = [subpos(1)+subpos(3)*oo(1) subpos(2)+subpos(4)*oo(2) ...
        subpos(3)*oo(3) subpos(4)*oo(4)];
    set(newax(kk),'Position',newl);
  end
  % Adjust colormap if necessary
  if  ii>1 && ~isequal(get(hfigs(ii),'colormap'),get(h.main,'colormap')),
    newcb= findobj(newax, 'Tag', 'Colorbar');
    newaxes= setdiff(newax, newcb);
    fig_acmAdaptCLim(acm, newaxes);
    % Adjust the displaying of the new colorbar [wieder einkommentiert]
    scnew = size(get(hfigs(ii),'colormap'),1);
    sc = size(colormap,1);
    for kk=1:numel(newcb)
%      set(get(newcb(kk),'Children'),'CData',(sc-scnew+1:sc)');
      set(get(newcb(kk),'Children'),'CData',(sc-acm.nColors+1:sc)');
    end
%     % Adjust graphic objects whereby color mapping is direct
%     % (ie not controlled via clim)
%     axes_direct = findall(newaxes,'CDataMapping','direct');
%     for kk=1:numel(axes_direct)
%       cdata = get(axes_direct(kk),'CData');
%       set(axes_direct(kk),'CData',cdata+abs(diff([size(colormap,1) acm.nColors])));
%     end
  end   
  % Place label
  if ~isempty(opt.label)
    set(gcf,'CurrentAxes',h.axes(ii))
    xl = get(gca,'XLim');
    yl = get(gca,'YLim');
    switch(lower(opt.labelPos))
      case 'northwest'
        h.label(ii) = text(xl(1),yl(2),opt.label{ii}, ...
          'VerticalAlignment','top','HorizontalAlignment','left', ...
          opt.labelOpt{:});
      case 'north'
        h.label(ii) = text(mean(xl),yl(2),opt.label{ii}, ...
          'VerticalAlignment','top','HorizontalAlignment','center', ...
          opt.labelOpt{:});
      case 'northeast'
        h.label(ii) = text(xl(2),yl(2),opt.label{ii}, ...
          'VerticalAlignment','top','HorizontalAlignment','right', ...
          opt.labelOpt{:});
      case 'west'
        h.label(ii) = text(xl(1),mean(yl),opt.label{ii}, ...
          'VerticalAlignment','middle','HorizontalAlignment','left', ...
          opt.labelOpt{:});
      case 'east'
        h.label(ii) = text(xl(2),mean(yl),opt.label{ii}, ...
          'VerticalAlignment','middle','HorizontalAlignment','left', ...
          opt.labelOpt{:});
      case 'southwest'
        h.label(ii) = text(xl(1),yl(1),opt.label{ii}, ...
          'VerticalAlignment','bottom','HorizontalAlignment','left', ...
          opt.labelOpt{:});
      case 'south'
        h.label(ii) = text(mean(xl),yl(1),opt.label{ii}, ...
          'VerticalAlignment','bottom','HorizontalAlignment','center', ...
          opt.labelOpt{:});
      case 'southeast'
        h.label(ii) = text(xl(2),yl(1),opt.label{ii}, ...
          'VerticalAlignment','bottom','HorizontalAlignment','right', ...
          opt.labelOpt{:});
    end
  end
end

% delete(h.axes)  % these axes were only placeholders for the new data
set(h.axes,'visible','off')
% h.axes = get(gcf,'Children');
h.children = setdiff(get(gcf,'Children'),h.axes); % Get all children except for the new axes

if opt.deleteFigs
    delete(hfigs);
end
