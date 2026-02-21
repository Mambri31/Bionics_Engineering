%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Questo script Matlab genera una realizzazione di un processo AR(1) 
% e traccia i grafici della funzione di autorcorrelazione (ACF) 
% e della densità spettrale di potenza (PSD)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

N=100;               % Numero di campioni generati
rho=-0.8 ;            % Coefficiente di correlazione ad un passo
varX=1;              % Potenza del processo AR(1) X[n]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

n=[0:N-1]';
varW=varX*(1-rho^2);                   % Potenza del processo delle innovazioni W[n]
Ntrans=ceil(3*log(10)/abs(log(rho)));  % Si considera la risposta impulsiva nulla quando è minore di 10^-3
Ntot=Ntrans+N;
W=sqrt(varW)*randn(Ntot,1);
Z=filter(1,[1 -rho]',W);
X=Z(Ntrans+1:Ntot);
Rx=varX*rho.^n;
f=[0:0.001:0.5]';
Sx=varW./((abs(1-rho*exp(-j*2*pi*f))).^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(1)
stem(n,X,'Linewidth',2)  
grid on
xlabel('Discrete-Time [n]')               
ylabel('X[n]')
title('Processo AR(1) Gaussiano')
legend('Realizzazione - Processo AR(1)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(2)
stem(n,Rx,'Linewidth',2)  
grid on
xlabel('Time-Lag [m]')               
ylabel('R_{X}[m]')
title('Processo AR(1) Gaussiano')
legend('ACF - Processo AR(1)')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

figure(3)
plot(f,Sx,'Linewidth',2)  
grid on
xlabel('Digital Frequency (f)')               
ylabel('S_{X}(f)')
title('Processo AR(1) Gaussiano')
legend('PSD - Processo AR(1)')

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



