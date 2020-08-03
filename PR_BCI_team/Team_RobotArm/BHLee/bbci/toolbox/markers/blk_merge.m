function blk= blk_merge(blk1, blk2, varargin)

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'sort', 1, ...
                  'className',[]);

if isempty(blk1),
  blk= blk2;
  return;
end

if blk1.fs~=blk2.fs,
  error('sampling rates inconsistent');
end

if ~isfield(blk1, 'y'),
  blk1.y= ones(1, size(blk1.ival,2));
end
if ~isfield(blk2, 'y'),
  blk2.y= ones(1, size(blk2.ival,2));
end

blk.fs= blk1.fs;
blk.ival= cat(2, blk1.ival, blk2.ival);

if isfield(blk1, 'y'),
  s1= size(blk1.y);
  s2= size(blk2.y);
  if isfield(blk1, 'className') & isfield(blk2, 'className'),
    blk.y= [blk1.y, zeros(s1(1), s2(2))];
    blk2y= [zeros(s2(1), s1(2)), blk2.y];
    blk.className= blk1.className;
    for ii = 1:length(blk2.className)
      c = find(strcmp(blk.className,blk2.className{ii}));
      if isempty(c)
        blk.y= cat(1, blk.y, zeros(1,size(blk.y,2)));
        blk.className=  cat(2, blk.className, {blk2.className{ii}});
        c= size(blk.y,1);
      elseif length(c)>1,
        error('multiple classes have the same name');
      end
      blk.y(c,end-size(blk2.y,2)+1:end)= blk2.y(ii,:);
    end
  else
    blk.y= [[blk1.y; zeros(s2(1), s1(2))], [zeros(s1(1), s2(2)); blk2.y]];
  end
end

if ~isempty(opt.className),
  if length(opt.className)~=size(blk.y,1),
    error('length of opt.className does not fit');
  end
  blk.className= opt.className;
end

if opt.sort,
  blk= blk_sortChronologically(blk);
end
