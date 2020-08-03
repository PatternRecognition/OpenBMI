available_leads= 'azyxwvu';
[dmy,sl]= getSubjectCode('');
for idx= 1:length(sl),
  i1= ceil(idx/26);
  i2= mod(idx-1, 26);
  code= [available_leads(i1) char('a'+i2)];
  fprintf('%s: %s\n', code, sl{idx});
end
