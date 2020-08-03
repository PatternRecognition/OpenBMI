function clean_up(varargin)

for i = 1:length(varargin)
  delete varargin{i};
end
