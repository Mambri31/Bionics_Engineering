%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab genera una realizzazione di un processo AR(1) 
% stima e traccia il grafico della funzione di autorcorrelazione (ACF) 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

N=4096;              % Numero di campioni generati
Nlag=4000;            % Numero di ritardi per cui si stima la ACF -> max{Nlag} = N-1
mlag=[0:Nlag]';
rho=0.9 ;            % Coefficiente di correlazione ad un passo
varX=1;              % Potenza del processo AR(1) X[n]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

n=[0:N-1]';
varW=varX*(1-rho^2); % Potenza del processo delle innovazioni W[n]
Ntrans=ceil(3*log(10)/abs(log(rho)));
Ntot=Ntrans+N;
W=sqrt(varW)*randn(Ntot,1);
Z=filter(1,[1 -rho]',W);
X=Z(Ntrans+1:Ntot);
Rx=varX*rho.^mlag;

for m=0:Nlag
    
    Rxhatnp(m+1)=mean(X(1:N-m).*X(m+1:N));
    Rxhatp(m+1)=((N-m)/N)*Rxhatnp(m+1);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(1)
stem(mlag,Rxhatnp,'Linewidth',2)
grid on
hold on
stem(mlag,Rx,'r','Linewidth',2)  
xlabel('Time-Lag [m]')               
ylabel('R_{X}[m]')
title('Processo AR(1) Gaussiano')
legend('ACF - Stima non polarizzata','ACF - Teorica')
axis([0 Nlag -0.5 1.5])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(2)
stem(mlag,Rxhatp,'Linewidth',2)
grid on
hold on
stem(mlag,Rx,'r','Linewidth',2) 
xlabel('Time-Lag [m]')               
ylabel('R_{X}[m]')
title('Processo AR(1) Gaussiano')
legend('ACF - Stima polarizzata','ACF - Teorica')
axis([0 Nlag -0.5 1.5])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%
% Autore: 
% Fulvio Gini; 
% Dipartimento di Ingegneria dell'Informazione, Università di Pisa
% Via G. Caruso 16, I-56122, Pisa, Italia
% Tel: +39 050 2217550; Fax: +39 050 2217522
% E-mail: f.gini@ing.unipi.it
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


