function fb_opt= animate_fb_classes(fb_opt, ptr, dscr_out, dtct_out, comb_out,error_out)
%fb_opt= animate_fb_speller(fb_opt, ptr, dscr_out, dtct_out,
%comb_out,error_out)

global BCI_DIR mrk pp first_test_event time_line trigger 

persistent ha1 ha2 ht_marker is_bard child childvert

if isempty(trigger)
  trigger = cell(2,2);
  trigger{1,1} = 'L';
  trigger{1,2} = [1,65,70];
  trigger{2,1} = 'R';
  trigger{2,2} = [2,74,192];
end

MARKER_VIEW= 3;

colors = [1 0 0;0 1 0; 0 0 1];

if isequal(ptr,'init')
  clf;
  ha1= subplot('position', [0 .01 .999 .09]); box on;
  set(gca, 'xTick',0, 'xTickLabel','', 'xLim',[-1000 1000]*MARKER_VIEW, ...
           'yTick',[], 'yLim',[-1 1]);
  maxMarkers= 10*MARKER_VIEW;           %% 5 markers per second is maximum
  ht_marker= text(NaN*ones(1,maxMarkers), zeros(1,maxMarkers), '');
  set(ht_marker, 'fontSize',28, 'fontWeight','bold');
  
  ha2 = subplot('position',[0.1 .2 .8 .78]); box on;
  if isfield(fb_opt,'YLim'),
    set(ha2,'YLim',fb_opt.YLim);
  end
  drawnow
  is_bard = 0;
else
  
  vec = dscr_out(:,ptr);
  if length(vec)>1
    clas = mrk.className;
  else
    clas = {'dscr'};
  end
  
  if exist('dtct_out','var') & ~isempty(dtct_out);
    vec = [vec;dtct_out(ptr)];
    clas = {clas{:},'no_detect'};
  end
  
  if exist('ep_out','var') & ~isempty(ep_out);
    vec = [vec;ep_out(ptr)]
    clas = {clas{:},'ishit?'};
  end
  
  if exist('comb_out','var') & ~isempty(comb_out);
    vec = [vec;comb_out(ptr)]
    clas = {clas{:},'combiner'};
  end
  
  
  subplot(ha2);
  if is_bard==0
    bar(vec);
    set(ha2,'XTick',1:length(vec));
    set(ha2,'XTickLabel',clas);
    is_bard = 1;
    if isfield(fb_opt,'YLim'),
      set(ha2,'YLim',fb_opt.YLim);
    end
    child = get(ha2,'Children');
    childvert = get(child,'Vertices');
  else
    childvert(3:5:end,2) = vec;
    childvert(4:5:end,2) = vec;
    set(child,'Vertices',childvert);
  end
  
  subplot(ha1);
  inWindow= find(abs(mrk.pos-pp)/mrk.fs<MARKER_VIEW);
  nMark= length(inWindow);
  to = mrk.toe(inWindow);
  trig = cat(1,trigger{:,2});
  for ii= 1:nMark,
    [iii,jjj] = find(to(ii)==trig);
    if ~isempty(iii)
      x= (mrk.pos(inWindow(ii))-pp)*1000/mrk.fs;
      set(ht_marker(ii), 'position', [x 0 0], 'string',trigger{iii,1}, ...
			'color',colors(iii,:));
    end
  end
  set(ht_marker(nMark+1:end), 'position',[NaN 0 0]);

  drawnow
  
end

