%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Questo programma grafica la distirbuzione di Rice per diversi valori dei parametri.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;

A=4;
N0=1;
T=[1 2 5 10 20]';
NT=length(T);
a=[0:0.001:8]';
NA=length(a);
pdfa=zeros(NA,NT);

for i=1:NT

    sigma2=N0/T(i);
    pdfa(:,i)=a./sigma2.*exp(-(a.^2+A^2)./(2*sigma2)).*besseli(0,a*A/sigma2);
    
end

figure (1);
plot(a,pdfa,'Linewidth',2);
grid on;
legend('T=1','T=2','T=5','T=10','T=20') ;
title('Rice Probability Density Function');
xlabel('a');ylabel('f_{A}(a)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

T=5;
N0=1;
Av=[0 1 2 5 8 10]';
NAmpl=length(Av);
a=[0:0.001:12]';
NA=length(a);
pdfa=zeros(NA,NAmpl);
sigma2=N0/T;

for i=1:NAmpl

    A=Av(i);
    pdfa(:,i)=a./sigma2.*exp(-(a.^2+A.^2)./(2*sigma2)).*besseli(0,a*A/sigma2);
    
end

% Figura rappresentante le ddp dei due stimatori

figure (2);
plot(a,pdfa,'Linewidth',2);
grid on;
legend('A=0', 'A=1','A=2','A=5','A=8','A=10') ;
title('Rice Probability Density Function');
xlabel('a');ylabel('f_{A}(a)')

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 
