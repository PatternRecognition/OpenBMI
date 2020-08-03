function testall
%TESTALL Test successively all the functions of the 
%	Time-Frequency Toolbox.
	
%	O. Lemoine  March-September 1996.

%  Choice of the Instantaneous Amplitude 
   amexpo1t 		
   amexpo2t 		
   amgausst  			
   amrectt   		
   amtriant 		
   
%  Choice of the Instantaneous Frequency
   fmconstt  
   fmhypt    
   fmlint    	
   fmodanyt  							
   fmpart   
   fmpowert  
   fmsint    
   gdpowert						

%  Choice of Particular Signals
   altest    							
   anaaskt   
   anabpskt  
   anafskt   
   anapulst 	
   anaqpskt  
   anasingt  							
   anastept  
   atomst    							
   dopnoist 
   dopplert  
   klaudert  							
   mexhatt    					

%  Addition of Noise
   noisecgt  							
   noisecut  							

%  Modification
   scalet    						
   

% Processing Files

%  Time-Domain Processing
   ifestart
   instfret							
   loctimet  
	
%  Frequency-Domain Processing
   fmtt      					
   ifmtt     					
   locfreqt  
   sgrpdlat 					

%  Linear Time-Frequency Processing
   tfrgabot 
   tfrstftt  					


% Bilinear Time-Frequency Processing in the Cohen's Class
   tfrbjt
   tfrbudt	
   tfrcwt
   tfrgrdt			   
   tfrmhst
   tfrmht			   
   tfrmmcet
   tfrpaget			   
   tfrpmht
   tfrppagt			   
   tfrpwvt
   tfrridbt			   
   tfrridht
   tfrridnt			   
   tfrridtt			   
   tfrrit
   tfrspt			   
   tfrspwvt
   tfrwvt
   tfrzamt	
		   
%  Bilinear Time-Frequency Processing in the Affine Class
   tfrbertt  					
   tfrdflat  					
   tfrscalt 					
   tfrspawt    				
   tfruntet   				
   
%  Reassigned TimeFrequency Processing
   tfrrgabt  					
   tfrrmsct   							
   tfrrpmht   					
   tfrrppat  					
   tfrrpwvt   							
   tfrrspt    							
   tfrrspwt  					

%  Ambiguity Functions
   ambifunt 
   ambifuwt 					

%  PostProcessing or Help to the Interpretation
   friedmat 					
   holdert   					
   htlt     
   margtfrt  					
   midpoitt 
   momftfrt  					
   momttfrt  					
   renyit    					
   ridgest   					


% Other 
   dividert
   dwindowt
   integt    					
   integ2dt  					
   izakt      					
   kayttht
   modulot   
   oddt
   sigmergt  					
   zakt	      

