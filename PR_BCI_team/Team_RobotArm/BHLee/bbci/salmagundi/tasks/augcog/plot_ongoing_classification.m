function good = plot_ongoing_classification(traces,dsply);
% PLOT_ONGOING_CLASSIFICATION plots AUGCOG traces
%
% usage: 
%    good = plot_ongoing_classification(traces,dsply);
% 
% input: 
%    traces     traces given back by get_ongoing_classification
%    dsply      a struct with entries
%
% output:
%    good       the portion of correct time windows
%
% Guido DOrnhege, 04/05/2004

if ~exist('dsply','var') | isempty(dsply)
  dsply = struct;
end

dsply = set_defaults(dsply,...
                     'outputColor',[0 0 0.8],...
                     'labelColor',[1 0 0],...
                     'overlappingColor',[0.7 0.7 1],...
                     'wrongColor',[1 0.8 0.8],...
                     'trueColor',[0.6 1 0.6],...
                     'yLim',max(abs(traces.x))*[-1 1],...
                     'zero_line',[0 0 0]);

cols = cat(1,dsply.wrongColor,dsply.overlappingColor,dsply.trueColor);

traces.t = traces.t/1000;
ind = find(isnan(traces.x));
traces.x(ind)=0;

goodie = traces.y.*(abs(traces.y)==1);
goodie = sign(sign(traces.x.*goodie));

ind = [1,find(diff(goodie')~=0),length(traces.t)];

are = zeros(length(ind)-1,3);

for i = 1:length(ind)-1
  are(i,:) = [goodie(ind(i)+1),traces.t(ind(i)),traces.t(ind(i+1)-0)];
end



clf;
hold on;
for i = 1:3
  h = fill(are(1,[2,3,3,2,2]),dsply.yLim([1,1,2,2,1]),cols(i,:));
  set(h,'EdgeColor',cols(i,:));
end

for i = 1:size(are,1)
  h = fill(are(i,[2,3,3,2,2]),dsply.yLim([1,1,2,2,1]),cols(are(i,1)+2,:));
  set(h,'EdgeColor',cols(are(i,1)+2,:));
end

h = plot(traces.t,traces.x);
set(h,'Color',dsply.outputColor,'LineWidth',1)

h = plot(traces.t,0.6*min(abs(dsply.yLim))*traces.y);
set(h,'Color',dsply.labelColor,'LineWidth',1)

set(gca,'XLim',[0,traces.t(end)]);
set(gca,'YLim',dsply.yLim);

legend('real output','block',sprintf('false (%2.1f%%)',100*sum(goodie==-1)/sum(abs(goodie))),'overlap',sprintf('right (%2.1f%%)',100*sum(goodie==1)/sum(abs(goodie))),2);


good = sum(goodie==1)/sum(abs(goodie));

xlabel('in seconds');
h = line([0,traces.t(end)],[0,0]);
set(h,'Color',dsply.zero_line);
hold off;


set(gca,'YTick',0.5*min(abs(dsply.yLim))*[-1 0 1]);
set(gca,'YTickLabel',{traces.className{1},'0',traces.className{2}});



% $$$ 
% $$$ 
% $$$ subplot(3,1,1);
% $$$ plot(traces.t,traces.y,dsply.labelColor);
% $$$ set(gca,'XLim',[traces.t(1),traces.t(end)]);
% $$$ set(gca,'YLim',[-1 1]);
% $$$ %axis off
% $$$ title('Real block label');
% $$$ 
% $$$ subplot(3,1,3);
% $$$ goodie = traces.y.*(abs(traces.y)==1);
% $$$ goodie = sign(sign(xx.*goodie));
% $$$ plot(traces.t,goodie,dsply.resultColor);
% $$$ set(gca,'XLim',[traces.t(1),traces.t(end)]);
% $$$ set(gca,'YLim',[-1 1]);
% $$$ %axis off
% $$$ title(sprintf('Successful areas (True: %2.1f%% of the nonoverlapping area)',100*length(find(goodie==1))/sum(abs(goodie))));
% $$$ 


% $$$ ind = find(abs(traces.y)==1);
% $$$ xx = traces.x(ind);
% $$$ 
% $$$ [i,j] = ind2sub(size(traces.y),ind);
% $$$ 
% $$$ plot(j,xx,dsply.testColor);
% $$$ hold on;
% $$$ ind = find(abs(traces.y)<1);
% $$$ keyboard
% $$$ 
% $$$ [i,j] = ind2sub(size(traces.y),ind);
% $$$ 
% $$$ [j,in] = sort(j);
% $$$ ind = ind(in);
% $$$ xx = traces.x(ind);
% $$$ 
% $$$ jj = diff(j);
% $$$ pl = [0,find(jj'>1),length(j)];
% $$$ 
% $$$ for i = 1:length(pl)-1
% $$$   plot(j(pl(i)+1:pl(i+1)),xx(pl(i)+1:pl(i+1)),dsply.overlapcol);
% $$$ end
% $$$ 
% $$$ po = zeros(size(traces.y,1),1);
% $$$ for i = 1:size(traces.y,1)
% $$$   bi = find(traces.bidx==i);
% $$$   po(i) = mean(bi);
% $$$ end
% $$$ 
% $$$ set(gca,'XTick',po);
% $$$ set(gca,'XTickLabel',traces.block);
% $$$ 
% $$$ h = diff(traces.bidx);
% $$$ 
% $$$ ind = find(h~=0)+0.5;
% $$$ 
% $$$ for i = 1:length(ind)
% $$$   l = line([ind(i),ind(i)],get(gca,'YLim'));
% $$$   set(l,'Color',dsply.block_line);
% $$$ end
% $$$ hold off;
% $$$ set(gca,'XLim',[1,size(traces.x,2)]);
