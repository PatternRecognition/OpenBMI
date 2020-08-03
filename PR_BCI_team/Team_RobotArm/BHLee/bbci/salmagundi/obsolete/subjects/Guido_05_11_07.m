function Guido_05_11_07(typ)

if nargin==0,
  bbci = {'csp','cspmulti','slow','errorpotential'};
else

  switch typ
   case 'csp'
    bbci.train_file= {'Guido_05_11_07/imag_lettGuido'};
    bbci.classDef = {1,2,3;'left','right','foot'};
    bbci.player = 1;
    bbci.setup = 'csp';
    bbci.save_name = 'Guido_05_11_07/imag_Guido';
    bbci.feedback = '1d';
    bbci.classes = {'left','foot'};
   case 'cspmulti'
    bbci.train_file= {'Guido_05_11_07/imag_lettGuido'};
    bbci.classDef = {1,2,3;'left','right','foot'};
    bbci.player = 1;
    bbci.setup = 'cspmulti';
    bbci.save_name = 'Guido_05_11_07/imag_Guido';
    bbci.feedback = '1d';
    bbci.classes = {'left','right','foot'};
   case 'slow'
    bbci.train_file= {'Guido_05_11_07/imag_lettGuido'};
    bbci.classDef = {1,2,3;'left','right','foot'};
    bbci.player = 1;
    bbci.setup = 'imagMRP';
    bbci.save_name = 'Guido_05_11_07/imag_Guido';
    bbci.feedback = '1d';
    bbci.classes = {'left','foot'};  
   case 'errorpotential'
    bbci.train_file= {'Guido_05_11_07/imag_1drfb3Guido'};
    bbci.classDef = {[11,12],[21,22];'hit','miss'};
    bbci.player = 1;
    bbci.setup = 'errorpotential';
    bbci.save_name = 'Guido_05_11_07/imag_Guido';
    bbci.feedback = '1d_error';
    bbci.classes = {'hit','miss'};  
  end
end

assignin('caller','bbci',bbci);
