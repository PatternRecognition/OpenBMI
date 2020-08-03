function jvm= jvm_hideFig

a= sscanf(getfield(ver('MATLAB'), 'Release'), '(R%d)');
v= version('-java');
if isempty(strfind(v, 'not enabled')) % && a>=2009,
  jvm.fig= gcf;
  jvm.visible= get(jvm.fig, 'Visible');
  set(jvm.fig, 'Visible','off');
else
  jvm= [];
end
