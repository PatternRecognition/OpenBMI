function out= choose_case(str);

color_groups = {'fdyGk&lt;', 'pJUX!E', 'iSwcz-','TBMqAH','LRvON.'};
alpha= strcat(color_groups{:});
alpha= strrep(alpha, '&lt','<');
for i= 1:length(str),
  j= find(upper(str(i))==upper(alpha));
  out(i)= alpha(j);
end
