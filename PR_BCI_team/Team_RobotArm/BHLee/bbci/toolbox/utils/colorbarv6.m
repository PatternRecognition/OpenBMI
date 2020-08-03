function h_cb= colorbarv6(varargin)
%COLORBARV6 - Creates a MATLAB V6-style colorbar

vv= ver;
ii= strmatch('MATLAB', {vv.Name}, 'exact');
if str2num(strtok(vv(ii).Version,'.'))>=7,
  h_cb= colorbar('v6', varargin{:});
else
  h_cb= colorbar(varargin{:});
end
