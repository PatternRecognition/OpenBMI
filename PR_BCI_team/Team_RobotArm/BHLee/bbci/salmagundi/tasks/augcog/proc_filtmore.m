function cn = proc_filtmore(cnt,varargin);

cn = proc_filt(cnt,varargin{1}{1},varargin{1}{2});

for i = 2:length(varargin)
  hcn = proc_filt(hcn,varargin{i}{1},varargin{i}{2});
  cn.x = cat(2,cn.x,hcn.x);
end
