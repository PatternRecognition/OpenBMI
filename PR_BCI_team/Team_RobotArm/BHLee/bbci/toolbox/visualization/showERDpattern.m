function [h,val] = showERDpattern(cnt, mrk, mnt, seg, refIval, class, frequencies, opt)
%
% Plot ERD dependent on frequencies and time (see Graimann B,et al. Visualization of significant ERD/ERS patterns in multichannel EEG ...)
% cnt,mrk,mnt is the normal data, marker and mnt struct
% seg is the interval where the classes are
% refIval: refIval for ERD
% class: a array of number whichs determines the classes taken for ERD (visualized is been able only for one class), default all classes
% frequencies has different opportunities:
%     [a;b]: all frequencies between a and b with frequency window length 1 (default 5 30), Mean Points will be plot
%     [a;b;c]: all frequencies between a and b with frequency window length c
%     a row vector: frequency windows between a(n-1) and a(n) to the mean point
%     {a,b}: a is a vector with the frequency points, b is a number or a b 2-dim. array or a matrix with the intervals around the points given by a.   
%     {a,b,c}: a is a 2-dim vector as range, b as given above, and c is the step length in a.
% opt is a struct with the following fields:
%   dim: is 2 or 3 (2 default). 2 means a 2DPseudocolor plot, 3 means a 3D Pseudocolor plot
%   filter: name of the filter method proc_<filter>. default filtForth
%   smooth: smooth parameter for ERD, default:[]
%   colorBand: twodim-vector for maximal range which is shown, default (-inf inf)
%   colorFlag: if 1 colorBand is used, if 0 all values will be limited in the range (default: 1)
%
% Output: h: Handle to the plot
%         val: the plotted values
%
% Guido Dornhege
% 02.08.2002

%checking input, setting defaults
if nargin<4 error('not enough input arguments'); end

if ~exist('class') | isempty(class)
	class = 1:size(mrk.y,1);
end

if ~exist('frequencies') | isempty(frequencies)
	frequencies = [5; 30];
end

if ~exist('opt') | isempty(opt)
	opt.dim = 2;
end

if ~isstruct(opt)
	opt.dim = opt;
end

if ~isfield(opt,'dim') 
	opt.dim = 2;
end

if ~isfield(opt,'smooth')
	opt.smooth = [];
end
if ~isfield(opt,'filter');
	opt.filter = 'filtForth';
end

if ~isfield(opt,'colorBand');
	opt.colorBand = [-inf inf];
end
if ~isfield(opt,'colorFlag');
	opt.colorFlag = 1;
end

opt.filter = ['proc_' opt.filter];


% calculates windows
if iscell(frequencies)
   if length(frequencies)>3 | length(frequencies)<2
	error('frequencies has wrong format');
   end
   if length(frequencies)==3
	frequencies{1} = frequencies{1}(1):frequencies{3}:frequencies{1}(2);
   end
   if length(frequencies{2}) == 1
	frequencies{2} = [-frequencies{2},frequencies{2}];
   end
   if size(frequencies{2},1) == 1
	frequencies{2} = repmat(frequencies{2},[length(frequencies{1}),1]);
   end
   if size(frequencies{2},2) == 1
	frequencies{2} = [-frequencies{2},frequencies{2}];
   end
   freq = [frequencies{1}+transpose(frequencies{2}(:,1)); frequencies{1}; frequencies{1}+transpose(frequencies{2}(:,2))];

elseif isnumeric(frequencies)
   if size(frequencies,2)==1
	if length(frequencies)==2
	    frequencies = [frequencies;1];
	end
        freq = frequencies(1):frequencies(3):(frequencies(2)-0.5*frequencies(3));
	freq = [freq;freq+0.5*frequencies(3);freq+frequencies(3)];
   else
	freq = [frequencies(1:end-1);0.5*(frequencies(1:end-1)+frequencies(2:end)); frequencies(2:end)];
   end
	
else 
error('frequencies must be array or cell');
end

[dum,a] = find(mrk.y(class,:));
mrk.pos = mrk.pos(:,a);
if isfield(mrk,'toe')
mrk.toe = mrk.toe(:,a);
end
mrk.y = ones(1,length(a));

iva = [min([seg,refIval]),max([seg,refIval])];
for i = 1:size(freq,2)
	fprintf('\r%i/%i',i,size(freq,2));
	band = transpose(freq(:,i));
	mpoint = band(2);
	band = band([1,3]);
	cnt_flt = feval(opt.filter,cnt,band);
	epo = makeSegments(cnt_flt,mrk, iva);
        erd = proc_squareChannels(epo);
	erd= proc_classMean(erd, 1);
	erd= proc_calcERD(erd, refIval, opt.smooth);
       erd = proc_selectIval(erd,seg);
	val(:,:,i) = erd.x;
end
ti = erd.t;
fprintf('\n');

win = [min(mnt.box(1,:)), max(mnt.box(1,:)), min(mnt.box(2,:)),max(mnt.box(2,:))];
if opt.colorFlag==0
val = min(opt.colorBand(2),val);
val = max(opt.colorBand(1),val);
end
cmax = min([max(val(:)),opt.colorBand(2)]);
cmin = max([min(val(:)),opt.colorBand(1)]);
win(2) = win(2)-win(1)+1;
win(4) = win(4)-win(3)+1;
width = 1/win(2);
height = 1/win(4);
nEvents = size(mrk.y,2);
for i = 1:size(mnt.x,1)
   if ~isnan(mnt.box(1,i))
	cc = strmatch(mnt.clab{i},cnt.clab);
	if ~isempty(cc)
       subplot('position',[(mnt.box(1,i)-win(1))*width+0.05*width, (mnt.box(2,i)-win(3))*height+0.02, 0.9*width, 0.75*height]);
       	 mat = squeeze(val(:,cc(1),:));
mat = [mat, 0.5*(cmin+cmax)*ones(size(mat,1),1); 0.5*(cmin+cmax)*ones(1,size(mat,2)+1)];

	if opt.dim==2
	 pcolor([ti,2*ti(end)-ti(end-1)],[freq(1,:),freq(3,end)],mat');
	else
	 surf(ti,freq(2,:),mat');
	end
	shading flat
	title(mnt.clab{i});
	caxis([cmin cmax]);
	axis off
end
   end
end

if size(mnt.box,2)>size(mnt.x,1)
	%legende
	if ~isnan(mnt.box(1,end))
	  subplot('position',[(mnt.box(1,end)-win(1))*width+0.05*width, (mnt.box(2,end)-win(3))*height+0.2*height, 0.9*width, 0.7*height]);
  	j=pcolor(rand(5));   %MATLAB SEI DANK; wir faken rum!!!!
        caxis([cmin cmax]);
        h = colorbar('horiz');
        set(gca,'Visible','off');
	set(j,'Visible','off');
end
end



if isfield(mrk, 'className'),
  evtStr= [vec2str({mrk.className{class}}, [], ' & ') ','];
else
  evtStr= '';
end
  xLimStr= sprintf('[%g %g] ms ', seg(1),seg(2));
  yLimStr= sprintf('[%g %g] Hz', min(freq(1,:)),max(freq(3,:)));

tit= sprintf('%s N=%s,  %s %s', evtStr, vec2str(nEvents,[],'/'), ...
             xLimStr, yLimStr);
if isfield(cnt, 'title'),
  tit= [untex(epo.title) ':  ' tit];
end
h= addTitle(tit);




	